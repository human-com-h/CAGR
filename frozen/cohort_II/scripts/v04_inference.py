"""Candidate feature adapter; same v0.3 cache mechanics, explicit new feature schema."""
import math,time
from v03_inference import PausedGenerator
from execution_policy import guard_model_execution
from v04_features import GAP_SCHEMA,summarize_gap,torch_step_signals

class LightweightGenerator(PausedGenerator):
    def __init__(self,config,revision,mode='gap',top_k=64):
        guard_model_execution()
        if mode not in {'gap','local_entropy'}:raise ValueError('Light features require an explicit mode')
        self.signal_mode=mode;self.signal_top_k=top_k
        self.revision=revision;self.sampling_temperature=None
        self.feature_schema=GAP_SCHEMA if mode=='gap' else f'local_entropy_{top_k}_v1'
        super().__init__(config,revision)

    def begin(self,*args,**kwargs):
        if getattr(self,'signal_collection_enabled',True) is False:
            if len(args)>=5:args=(*args[:4],False,*args[5:])
            else:kwargs['collect_signals']=False
        state=super().begin(*args,**kwargs);call=state['call']
        call.update(do_sample=self.sampling_temperature is not None,temperature=self.sampling_temperature)
        call.pop('token_entropies');call.pop('token_probability_margins')
        call['feature_schema']=self.feature_schema
        call['token_gap_exp_uncertainty']=state['margins']
        if self.signal_mode=='local_entropy':call['token_local_entropies']=state['entropies']
        return state

    def _forward(self,ids,past,collect):
        torch=self.torch;self._sync();start=time.perf_counter()
        before=int(past.get_seq_length()) if past is not None else 0;length=int(ids.shape[1])
        positions=torch.arange(before,before+length,device=ids.device,dtype=torch.int64)
        mask=torch.ones((1,before+length),dtype=torch.long,device=ids.device)
        with torch.inference_mode():
            result=self.model(input_ids=ids,past_key_values=past,use_cache=True,attention_mask=mask,cache_position=positions)
            logits=result.logits[:,-1,:];self._sync();forward_ms=1000*(time.perf_counter()-start)
            # topk's index tie-breaking is not argmax's; retain the frozen greedy choice.
            sampling_ms=0.
            if self.sampling_temperature is None:pending=logits.argmax(-1).reshape(1,1)
            else:
                self._sync();sampling_start=time.perf_counter()
                pending=torch.multinomial((logits.float()/self.sampling_temperature).softmax(-1),1)
                self._sync();sampling_ms=1000*(time.perf_counter()-sampling_start)
            token=int(pending.item())
            stats=None;signal_ms=0.
            if collect:
                self._sync();t=time.perf_counter()
                stats=torch_step_signals(logits,self.signal_mode,self.signal_top_k)
                self._sync();signal_ms=1000*(time.perf_counter()-t)
        past=result.past_key_values
        return token,pending,past,stats,signal_ms,{'input_tokens':length,'cache_before':before,'cache_after':int(past.get_seq_length()),'prefill':before==0,'forward_ms':forward_ms,'sampling_ms':sampling_ms}

    def generate_sample(self,system,user,max_tokens,kind,temperature=1.):
        if not math.isfinite(temperature) or temperature<=0:raise ValueError('Positive sampling temperature required')
        if self.active is not None:raise RuntimeError('Cannot sample while a Pilot cache is active')
        self.sampling_temperature=temperature
        try:return self.generate(system,user,max_tokens,kind,collect_signals=False)
        finally:self.sampling_temperature=None

    def features(self,state):
        t=time.perf_counter();gaps=state['gaps']
        out=summarize_gap(gaps) if gaps else {'feature_schema':self.feature_schema,'mean_logit_gap':None,'gap_uncertainty_of_mean':None,'targ_mean_exp_gap':None}
        out.update(feature_schema=self.feature_schema,effective_k=len(gaps),answer_length_tokens=sum(x not in self.eos for x in state['generated']),token_logit_gaps=list(gaps))
        if self.signal_mode=='local_entropy':out.update(local_entropy=sum(state['entropies'])/len(state['entropies']) if state['entropies'] else None,local_entropy_support=self.signal_top_k)
        elapsed=1000*(time.perf_counter()-t);call=state['call'];call['ms']+=elapsed
        call['feature_materialization_ms']=call.get('feature_materialization_ms',0.)+elapsed
        return out
