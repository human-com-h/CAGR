"""K=5 pause/continue using the SAME KV cache and pending generated token."""
import hashlib, math, time
from run_baselines import Generator

class PausedGenerator(Generator):
    def __init__(self, config, revision):
        if not config.get('local_model_path'):
            raise RuntimeError('v0.3 requires a verified local_model_path; Hub fallback is forbidden.')
        super().__init__(config, revision)
        self.active = None

    def _sync(self):
        self.torch.cuda.synchronize(self.config['device'])

    def _prepare(self, system, user):
        text=self.tokenizer.apply_chat_template([{'role':'system','content':system},{'role':'user','content':user}], tokenize=False,add_generation_prompt=True)
        ids=self.tokenizer(text,return_tensors='pt',add_special_tokens=False)['input_ids'].to(self.config['device'])
        if ids.shape[1]>self.config['max_input_tokens']:
            raise ValueError('Prompt exceeds the frozen 4096-token input budget.')
        self.torch.cuda.reset_peak_memory_stats(self.config['device'])
        return ids, int(ids.shape[1]), hashlib.sha256(text.encode()).hexdigest()

    def _forward(self, ids, past, collect):
        torch=self.torch
        self._sync()
        forward_start=time.perf_counter()
        before=int(past.get_seq_length()) if past is not None else 0
        length=int(ids.shape[1])
        positions=torch.arange(before,before+length,device=ids.device,dtype=torch.int64)
        mask=torch.ones((1,before+length),dtype=torch.long,device=ids.device)
        with torch.inference_mode():
            result=self.model(input_ids=ids,past_key_values=past,use_cache=True,attention_mask=mask,cache_position=positions)
            logits=result.logits[:,-1,:]
            self._sync()
            forward_ms=1000*(time.perf_counter()-forward_start)
            stats=None
            signal_ms=0.0
            if collect:
                self._sync()
                start=time.perf_counter()
                logp=logits.float().log_softmax(-1)
                probs=logp.exp()
                h=-(probs*logp).sum(-1)
                ptop=probs.topk(2,-1).values
                ztop=logits.float().topk(2,-1).values
                stats=(float(h.item()),float((ptop[0,0]-ptop[0,1]).item()),float((ztop[0,0]-ztop[0,1]).item()))
                self._sync()
                signal_ms=1000*(time.perf_counter()-start)
                if not all(math.isfinite(x) for x in stats):
                    raise RuntimeError('Non-finite Pilot signal; preserve failure and stop.')
            pending=logits.argmax(-1).reshape(1,1)
            token=int(pending.item())
        past=result.past_key_values
        event={'input_tokens':length,'cache_before':before,'cache_after':int(past.get_seq_length()),'prefill':before==0,'forward_ms':forward_ms}
        return token,pending,past,stats,signal_ms,event

    def begin(self, system, user, max_tokens, kind, collect_signals=True):
        if self.active is not None:
            raise RuntimeError('Release or complete the current cache before another generation.')
        if max_tokens<=0:
            raise ValueError('A positive output budget is required.')
        start=time.perf_counter()
        call={'kind':kind,'status':'running','input_tokens':0,'output_tokens':0,'ms':0.0,'signal_ms':0.0,
              'forward_trace':[],'forward_attempts':[],'output_token_ids':[],'prefill_calls':0,'completion':None}
        self.calls.append(call)  # failures during tokenization remain in the ledger
        try:
            ids,n,digest=self._prepare(system,user)
            call.update(input_tokens=n,prompt_sha256=digest)
            state={'call':call,'pending':ids,'past':None,'max_tokens':max_tokens,'generated':[],
                   'entropies':[],'margins':[],'gaps':[],'collect':collect_signals,'done':False}
            call.update(token_entropies=state['entropies'],token_probability_margins=state['margins'],token_logit_gaps=state['gaps'])
            self.active=state
            call['ms']=1000*(time.perf_counter()-start)
            return state
        except Exception as exc:
            call.update(status='failed',ms=1000*(time.perf_counter()-start),error_type=type(exc).__name__)
            raise

    def advance(self, state, target_steps):
        if state is not self.active:
            raise RuntimeError('This cache is no longer active.')
        start=time.perf_counter()
        call=state['call']
        try:
            while not state['done'] and len(state['generated'])<min(target_steps,state['max_tokens']):
                step=len(state['generated'])
                attempt={'step':step,'input_tokens':int(state['pending'].shape[1]) if hasattr(state['pending'],'shape') else len(state['pending']),
                         'cache_before':int(state['past'].get_seq_length()) if state['past'] is not None else 0,'status':'running'}
                call['forward_attempts'].append(attempt)
                token,pending,past,stats,ms,event=self._forward(state['pending'],state['past'],state['collect'] and step<self.config['pilot_k'])
                attempt.update(event,status='ok')
                call['forward_trace'].append(event)
                call['prefill_calls']+=int(event['prefill'])
                # A continuation must pass ONLY the pending token, never the whole prompt.
                if event['cache_before'] and event['input_tokens']!=1:
                    raise RuntimeError('Detected repeated prefill during continuation.')
                state['past'],state['pending']=past,pending
                state['generated'].append(token)
                call['output_token_ids']=list(state['generated'])
                call['output_tokens']=len(state['generated'])
                call['signal_ms']+=ms
                if stats is not None:
                    state['entropies'].append(stats[0]);state['margins'].append(stats[1]);state['gaps'].append(stats[2])
                state['done']=token in self.eos or len(state['generated'])==state['max_tokens']
                call['cache_final_sequence_length']=event['cache_after']
            self._sync()
            call['ms']+=1000*(time.perf_counter()-start)
            call['pilot_output_tokens']=min(self.config['pilot_k'],call['output_tokens'])
            if call['output_tokens']<=self.config['pilot_k']:
                call['pilot_prefix_ms']=call['ms']
            call.update(peak_gpu_allocated_bytes=self.torch.cuda.max_memory_allocated(self.config['device']),
                        peak_gpu_reserved_bytes=self.torch.cuda.max_memory_reserved(self.config['device']))
        except Exception as exc:
            try: self._sync()
            except Exception: pass
            if call['forward_attempts'] and call['forward_attempts'][-1]['status']=='running':
                call['forward_attempts'][-1].update(status='failed',error_type=type(exc).__name__)
            call.update(status='failed',error_type=type(exc).__name__,ms=call['ms']+1000*(time.perf_counter()-start))
            self.active=None
            state['past']=None;state['pending']=None
            raise
        return state

    def features(self, state):
        start=time.perf_counter()
        def mean(x): return sum(x)/len(x) if x else None
        result={'prefix_entropy':mean(state['entropies']),'probability_margin':mean(state['margins']),
                'logit_gap':mean(state['gaps']),'effective_k':len(state['entropies']),
                'answer_length_tokens':sum(t not in self.eos for t in state['generated']),
                'token_entropies':list(state['entropies']),'token_probability_margins':list(state['margins']),
                'token_logit_gaps':list(state['gaps'])}
        elapsed=1000*(time.perf_counter()-start)
        state['call']['feature_materialization_ms']=state['call'].get('feature_materialization_ms',0.0)+elapsed
        state['call']['ms']+=elapsed
        return result

    def release(self, state, completion):
        call=state['call']
        if call['status']=='failed':
            return
        expected=call['input_tokens']+call['output_tokens']-1
        if call['cache_final_sequence_length']!=expected or call['prefill_calls']!=1:
            raise RuntimeError('KV length/prefill audit failed.')
        call.update(status='ok',completion=completion,stopped_on_eos=bool(state['generated'][-1] in self.eos))
        self.active=None
        state['past']=None;state['pending']=None

    def pilot(self, system, user, max_tokens, kind='pilot'):
        state=self.begin(system,user,max_tokens,kind)
        self.advance(state,self.config['pilot_k'])
        state['call']['cache_pause_sequence_length']=state['call']['cache_final_sequence_length']
        state['call']['pause_output_token_ids']=list(state['generated'])
        state['call']['pilot_prefix_ms']=state['call']['ms']
        return state,self.features(state)

    def complete(self, state):
        self.advance(state,state['max_tokens'])
        features=self.features(state)
        answer=self.tokenizer.decode(state['generated'],skip_special_tokens=True).strip()
        self.release(state,'answer_completed')
        return answer,features,state['call']

    def generate(self, system, user, max_tokens, kind, collect_signals=True):
        state=self.begin(system,user,max_tokens,kind,collect_signals)
        # Identical loop, no routing pause. Used for same-environment fixed baselines.
        return self.complete(state)

    def abort_active(self):
        if self.active is not None:
            call=self.active['call']
            call.update(status='failed',error_type='PipelineFailureAfterPilot')
            self.active['past']=None;self.active['pending']=None
            self.active=None
