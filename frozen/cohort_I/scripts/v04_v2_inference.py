"""Stage-end signal materialization; v1 source and routing semantics stay frozen.

No torch import or model execution on import. GPU equivalence is a required
runtime gate, not something established by the local NumPy/fake-device tests.
"""
import math,time
from v04_inference import LightweightGenerator

def materialize_top2(torch,pairs):
    """One host transfer for all K positions; float32 subtraction matches v1.

    A single batch exp preserves v1's diagnostic per-position uncertainty.
    The router's exp(mean gap) and TARG mean(exp gap) still use the same Python
    summarize_gap function. No exp/stack/host copy is done inside capture().
    """
    top=torch.stack(pairs,dim=0).float()
    gap=top[:,0]-top[:,1]
    packed=torch.stack((gap,torch.exp(-gap/3.)),dim=1)
    return packed.cpu().tolist()

class StageSignalBuffer:
    def __init__(self,torch,capacity=5):self.torch=torch;self.capacity=capacity;self.pairs=[];self.flushed=False
    def capture(self,logits):
        if self.flushed or len(self.pairs)>=self.capacity:raise RuntimeError('Signal window exceeded or already materialized')
        self.pairs.append(self.torch.topk(logits,2,dim=-1,sorted=True).values.reshape(-1,2)[0])
    def materialize(self):
        if self.flushed:return []
        self.flushed=True
        if not self.pairs:return []
        values=materialize_top2(self.torch,self.pairs);self.pairs.clear()
        if any(not math.isfinite(g+u) or g<0 for g,u in values):raise RuntimeError('Invalid batched gap')
        return values

class BatchedLightweightGenerator(LightweightGenerator):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.signal_mode!='gap':raise ValueError('v2 equivalence package supports only the frozen gap schema')
        # Allocate/reuse the five event pairs during recorded model setup,
        # rather than allocating timing objects inside every generated token.
        self._signal_timers=[(self.torch.cuda.Event(enable_timing=True),self.torch.cuda.Event(enable_timing=True)) for _ in range(self.config['pilot_k'])]
        for a,b in self._signal_timers:a.record();b.record()
    def _sync(self):
        # The greedy pending.item() remains necessary for EOS/greedy token choice.
        # Signal collection adds NO per-token cuda.synchronize().
        return None

    def pilot(self,*args,**kwargs):
        state,features=super().pilot(*args,**kwargs)
        state['call']['pilot_prefix_ms']=state['call']['ms']
        return state,features

    def begin(self,*args,**kwargs):
        state=super().begin(*args,**kwargs)
        state['v2_buffer']=StageSignalBuffer(self.torch,self.config['pilot_k'])
        state['v2_signal_events']=[];state['v2_signal_enqueue_ms']=0.
        state['call'].update(signal_collection='stage_end_batch_v2',signal_materialization_batches=0,
            signal_added_explicit_device_syncs=0,forward_ms_scope='individual GPU forward interval not isolated; aggregate generation wall time is fully measured',
            signal_accounting='conservative CPU enqueue + GPU topk event intervals + one blocking batch host materialization; final pending GPU tail may overlap the batch wait; signal is a generation subset, never added again to E2E')
        return state

    def _forward(self,ids,past,collect):
        torch=self.torch;state=self.active;before=int(past.get_seq_length()) if past is not None else 0;length=int(ids.shape[1])
        positions=torch.arange(before,before+length,device=ids.device,dtype=torch.int64)
        mask=torch.ones((1,before+length),dtype=torch.long,device=ids.device)
        with torch.inference_mode():
            result=self.model(input_ids=ids,past_key_values=past,use_cache=True,attention_mask=mask,cache_position=positions)
            logits=result.logits[:,-1,:]
            sample_start=time.perf_counter()
            if self.sampling_temperature is None:pending=logits.argmax(-1).reshape(1,1)
            else:pending=torch.multinomial((logits.float()/self.sampling_temperature).softmax(-1),1)
            token=int(pending.item())  # common greedy/sampling transfer; never a signal transfer
            sampling_ms=1000*(time.perf_counter()-sample_start) if self.sampling_temperature is not None else 0.
            if collect:
                start=time.perf_counter();a,b=self._signal_timers[len(state['v2_buffer'].pairs)]
                a.record();state['v2_buffer'].capture(logits);b.record()
                state['v2_signal_enqueue_ms']+=1000*(time.perf_counter()-start);state['v2_signal_events'].append((a,b))
        past=result.past_key_values
        event={'input_tokens':length,'cache_before':before,'cache_after':int(past.get_seq_length()),'prefill':before==0,'forward_ms':None,'sampling_ms':sampling_ms}
        return token,pending,past,None,0.,event

    def features(self,state):
        try:return self._features_batch(state)
        except BaseException:
            state['call']['signal_inflight_cost']='unknown_on_failure_not_zero'
            raise

    def _features_batch(self,state):
        call=state['call'];buffer=state['v2_buffer']
        if not buffer.flushed:
            start=time.perf_counter();n=len(buffer.pairs)
            values=buffer.materialize()  # one blocking CPU transfer, only at phase end
            if values:
                for gap,uncertainty in values:state['gaps'].append(gap);state['margins'].append(uncertainty)
                call['signal_materialization_batches']+=1
            gpu_ms=sum(a.elapsed_time(b) for a,b in state['v2_signal_events'])
            elapsed=1000*(time.perf_counter()-start)
            call['signal_enqueue_cpu_ms']=state['v2_signal_enqueue_ms'];call['signal_topk_gpu_ms']=gpu_ms
            call['signal_batch_host_wall_ms']=elapsed if n else 0.
            # Conservative component accounting, declared above. Actual E2E and
            # feature-on/off pairs remain the wall-clock measures, with no subtraction.
            call['signal_ms']+=state['v2_signal_enqueue_ms']+gpu_ms+(elapsed if n else 0.)
            call['ms']+=elapsed
        state['v2_signal_events'].clear()
        return super().features(state)

    def advance(self,state,target_steps):
        try:return super().advance(state,target_steps)
        except BaseException:
            state['call']['signal_inflight_cost']='unknown_on_failure_not_zero'
            raise
