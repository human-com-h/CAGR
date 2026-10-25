"""Append-only cache bridge on frozen Native State B; no document re-prefill."""
import hashlib,time
from v04_v3_native_state_inference import NativeStateGenerator

def suffix_text(tokenizer,instruction,last_token):
    # Derive an additional user turn from the frozen template, not a new system prompt.
    base=[{'role':'system','content':'__CAGR_SYSTEM__'},{'role':'user','content':'__CAGR_QUESTION__'},{'role':'assistant','content':'__CAGR_ANSWER__'}]
    before=tokenizer.apply_chat_template(base,tokenize=False,add_generation_prompt=False)
    after=tokenizer.apply_chat_template(base+[{'role':'user','content':instruction}],tokenize=False,add_generation_prompt=True)
    if not after.startswith(before):raise RuntimeError('Chat template is not prefix-composable')
    end=tokenizer.convert_tokens_to_ids('<|im_end|>')
    if not isinstance(end,int) or end<0:raise RuntimeError('Frozen Qwen assistant terminator missing')
    return ('' if last_token==end else '<|im_end|>')+'\n'+after[len(before):]

class CacheBridgeMixin:
    retain_rag1=False
    retained=None
    bridge_options=None
    def generate(self,system,user,max_tokens,kind,collect_signals=True):
        if kind!='rag1' or not self.retain_rag1:return super().generate(system,user,max_tokens,kind,collect_signals)
        if self.retained is not None:raise RuntimeError('A previous retained cache was not consumed')
        state=self.begin(system,user,max_tokens,kind,collect_signals);self.advance(state,max_tokens);features=self.features(state)
        start=time.perf_counter();answer=self.tokenizer.decode(state['generated'],skip_special_tokens=True).strip();past,pending=state['past'],state['pending']
        super().release(state,'answer_completed');state.update(past=past,pending=pending,source_system=system,source_user=user)
        self.retained=state;state['call']['cache_retained_for_bridge']=True;state['call']['ms']+=1000*(time.perf_counter()-start)
        return answer,features,state['call']

    def drop_retained(self):
        if self.retained is not None:
            self.retained['past']=None;self.retained['pending']=None;self.retained=None

    def bridge_spec(self):
        parent=self.retained
        if parent is None or self.active is not None:raise RuntimeError('Bridge needs one completed retained H1')
        options=self.bridge_options;text=suffix_text(self.tokenizer,options['instruction'],parent['generated'][-1]);ids=self.tokenizer.encode(text,add_special_tokens=False)
        if not 1<=len(ids)<=options['max_suffix_tokens']:raise RuntimeError('Bridge suffix is not the registered short window')
        before=int(parent['past'].get_seq_length());expected=parent['call']['input_tokens']+parent['call']['output_tokens']-1
        if before!=expected:raise RuntimeError('Retained H1 cache length mismatch')
        if before+len(ids)+1>self.config['max_input_tokens']:raise RuntimeError('Bridge effective context exceeds shared context budget')
        return parent,text,ids,before

    def _bridge_inputs(self,pending,ids):
        t=self.torch;suffix=t.tensor([ids],dtype=t.long,device=self.config['device'])
        return t.cat((pending,suffix),dim=1)

    def _raw_inputs(self,ids):return self.torch.tensor([ids],dtype=self.torch.long,device=self.config['device'])

    def run_token_input(self,ids,past,kind,max_tokens,*,input_count,reused_context=0,prompt_sha256=None,setup_started=None):
        if self.active is not None:raise RuntimeError('Another generation cache is active')
        if max_tokens<=0:raise ValueError('Positive output budget required')
        start=setup_started or time.perf_counter();call={'kind':kind,'status':'running','input_tokens':input_count,'output_tokens':0,'ms':0.,'signal_ms':0.,'feature_materialization_ms':0.,
            'prefill_calls':0,'suffix_prefill_calls':0,'reused_context_tokens':reused_context,'forward_trace':[],'forward_attempts':[],'output_token_ids':[],
            'prompt_sha256':prompt_sha256,'do_sample':False,'temperature':None,'token_logit_gaps':[],'completion':None,'attempt':0}
        self.calls.append(call);state={'call':call,'pending':ids,'past':past,'max_tokens':max_tokens,'generated':[],'raw_count':0,'collect':False,'done':False,'signal_flushed':True}
        self.active=state
        try:
            for step in range(max_tokens):
                before=int(state['past'].get_seq_length()) if state['past'] is not None else 0
                attempt={'step':step,'status':'running','cache_before':before};call['forward_attempts'].append(attempt)
                token,pending,cache,stats,ms,event=self._forward(state['pending'],state['past'],False)
                attempt.update(event,status='ok');call['forward_trace'].append(event)
                expected_input=input_count if step==0 else 1
                if event['cache_before']!=before or event['input_tokens']!=expected_input or event['cache_after']!=before+expected_input:raise RuntimeError('Bridge append/decode cache transition changed')
                if bool(event['prefill'])!=(before==0):raise RuntimeError('Bridge prefill label differs from actual cache')
                call['prefill_calls']+=int(event['prefill']);call['suffix_prefill_calls']+=int(step==0 and reused_context>0)
                state['generated'].append(token);state['past'],state['pending']=cache,pending
                call.update(output_tokens=len(state['generated']),output_token_ids=list(state['generated']),cache_final_sequence_length=event['cache_after'])
                if token in self.eos:break
            expected=reused_context+input_count+call['output_tokens']-1
            if call['cache_final_sequence_length']!=expected or call['prefill_calls']!=int(not reused_context):raise RuntimeError('Bridge final cache/prefill invariant failed')
            answer=self.tokenizer.decode(state['generated'],skip_special_tokens=True).strip();call.update(status='ok',stopped_on_eos=state['generated'][-1] in self.eos,completion='bridge_completed',decoded_text=answer)
            return answer,{},call
        except BaseException as exc:
            call.update(status='failed',error_type=type(exc).__name__,inflight_device_cost_unknown=True)
            if call['forward_attempts'] and call['forward_attempts'][-1]['status']=='running':call['forward_attempts'][-1]['status']='failed'
            raise
        finally:
            call['ms']=1000*(time.perf_counter()-start);self.active=None;state['past']=None;state['pending']=None

    def bridge_continue(self,max_tokens,keep_reference=False):
        start=time.perf_counter();parent,text,ids,before=self.bridge_spec()
        if max_tokens>self.bridge_options['max_new_tokens']:raise RuntimeError('Bridge output exceeds registered cap')
        full=None
        if keep_reference:
            prompt=self.tokenizer.apply_chat_template([{'role':'system','content':parent['source_system']},{'role':'user','content':parent['source_user']}],tokenize=False,add_generation_prompt=True)
            full=self.tokenizer.encode(prompt,add_special_tokens=False)+list(parent['generated'])+ids
        pending=self._bridge_inputs(parent['pending'],ids);cache=parent['past'];parent['past']=None;parent['pending']=None;self.retained=None
        digest=hashlib.sha256((parent['call']['prompt_sha256']+'|'+str(parent['generated'])+'|'+text).encode()).hexdigest()
        result=self.run_token_input(pending,cache,'kv_bridge',max_tokens,input_count=1+len(ids),reused_context=before,prompt_sha256=digest,setup_started=start)
        result[2].update(parent_prompt_sha256=parent['call']['prompt_sha256'],parent_output_token_ids=list(parent['generated']),suffix=text,suffix_token_ids=ids,
            parent_pending_tokens_reingested=1,document_prefill_repeated=False,original_document_prefill_calls=parent['call']['prefill_calls'],bridge_max_tokens=max_tokens)
        return result,full

    def raw_bridge_reference(self,ids,max_tokens):
        start=time.perf_counter();tensor=self._raw_inputs(ids)
        return self.run_token_input(tensor,None,'bridge_full_prefix_reference',max_tokens,input_count=len(ids),prompt_sha256=hashlib.sha256(str(ids).encode()).hexdigest(),setup_started=start)

    def abort_active(self):
        self.drop_retained();return super().abort_active()

class KVBridgeGenerator(CacheBridgeMixin,NativeStateGenerator):pass
