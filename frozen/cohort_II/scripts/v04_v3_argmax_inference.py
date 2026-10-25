"""One vocabulary scan for frozen greedy argmax AND top2; deferred host records."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,time
from v04_inference import LightweightGenerator
from v04_v3_inference import K,V,TIMER_MASK,TOKEN_MASK
from v04_v3_clockfix_inference import CausalSignalPool,causal_union_ns

def load_native(torch,root):
    from torch.utils.cpp_extension import load,CUDA_HOME
    nvcc=Path(CUDA_HOME or '/missing')/'bin/nvcc'
    if not nvcc.is_file() or not shutil.which('ninja') or not shutil.which('c++'):raise RuntimeError('Existing locked offline compiler required; no download fallback')
    build=root/'logs/v3_argmax_native_build';build.mkdir(parents=True,exist_ok=True)
    os.environ['MAX_JOBS']='1';os.environ['TORCH_CUDA_ARCH_LIST']='7.5'
    sources=[Path(__file__).parent/n for n in ['v04_v3_argmax_bind.cpp','v04_v3_argmax_cuda.cu']]
    (build/'toolchain.json').write_text(json.dumps({'nvcc':str(nvcc),'ninja':shutil.which('ninja'),'cxx':shutil.which('c++'),
        'nvcc_version':subprocess.run([str(nvcc),'--version'],capture_output=True,text=True,timeout=20).stdout,'torch':torch.__version__,
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}),encoding='utf-8')
    return load(name='cagr_v3_argmax_top2',sources=[str(p) for p in sources],extra_cflags=['-O2'],extra_cuda_cflags=['-O2','--fmad=false'],build_directory=str(build),verbose=False)

def materialize_record(raw,capture_cpu_ns=0):
    a,b,z,s,e,decode=raw;ns,bounds=causal_union_ns(a,b,z,s,e)
    return {'cpu_start_ns':a,'cpu_end_ns':b,'shared_read_done_ns':z,'gpu_start_ns':s,'gpu_end_ns':e,
        'work_union_upper_ns':ns,'causal_offset_lower_ns':bounds[0],'causal_offset_upper_ns':bounds[1],
        'decode_cpu_ns':decode,'capture_cpu_ns':capture_cpu_ns,'shared_token_waits':1,'extra_device_syncs':0,'native_entry_calls':1,'native_kernel_launches':2,
        'accounting':'per-dispatch causal union upper; dictionaries deferred to stage end; minimum ALL block starts'}

class ArgmaxPool(CausalSignalPool):
    def __init__(self,torch,native,device,eos):
        super().__init__(torch,native,device,eos)
        self.meta=torch.empty(K*8,dtype=torch.int64,device=device)
        self.indices=torch.empty((K,8),dtype=torch.int32,device=device)
        self.invalid=torch.empty((K,8),dtype=torch.int32,device=device)
        self.pending=torch.empty((1,1),dtype=torch.int64,device=device)

    def fused_step(self,logits,n,limit,mode):
        if mode not in ['A','B'] or not 1<=n<=limit<=K:raise RuntimeError('Invalid frozen K5 signal step')
        a=time.perf_counter_ns()
        self.native.dispatch(logits,self.pending,self.pairs,self.partials,self.indices,self.invalid,self.meta,self.eos,self.packet,
            self.mapped_features,self.mapped_starts,n,limit,mode=='A')
        b=time.perf_counter_ns();value=int(self.packet.item())  # The ONE original greedy-token wait.
        z=time.perf_counter_ns();s=int(self.starts_view[n-1]);token=value&TOKEN_MASK;e=s+(((value>>18)-(s&TIMER_MASK))&TIMER_MASK)
        decode=time.perf_counter_ns()-z
        return token,self.pending,(a,b,z,s,e,decode)

class ArgmaxGenerator(LightweightGenerator):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.signal_mode!='gap' or self.config['pilot_k']!=K:raise RuntimeError('Frozen gap/K5 required')
        self.native=load_native(self.torch,Path(__file__).resolve().parents[1]);self.pool=ArgmaxPool(self.torch,self.native,self.config['device'],self.eos)
        self.pool.calibrate();self.signal_variant='A';self.production_event_records=0

    def _sync(self):return None

    def begin(self,*args,**kwargs):
        state=super().begin(*args,**kwargs);start=time.perf_counter()
        state['raw_signal']=[None]*K;state['raw_capture_cpu']=[0]*K;state['raw_count']=0;state['signal_flushed']=False
        state['call'].update(signal_collection='v3_fused_argmax_top2_'+self.signal_variant,
            signal_materialization_batches=0,signal_added_explicit_device_syncs=0,signal_cuda_event_records=0,
            signal_clock_bounds_ns=list(self.pool.bounds),signal_buffer_bytes=K*V*2,signal_partial_buffer_bytes=320,
            signal_index_buffer_bytes=160,signal_invalid_buffer_bytes=160,signal_clock_scratch_bytes=320,
            signal_pending_buffer_bytes=8,signal_native_entry_calls_per_position=1,signal_native_kernel_launches_per_position=2,
            signal_accounting='per-dispatch causal CPU/GPU union upper + exclusive deferred record/materialization work',
            signal_clock_bounds_scope='diagnostic only; every dispatch independently enclosed at stage end',
            forward_ms_scope='generation total wall measured; per-forward interval not isolated',
            greedy_selection='fused argmax lowest index; original argmax retained outside observed greedy steps',
            signal_host_records='K5 scalar tuples; full dictionaries at stage end',signal_full_vocab_argmax_calls_on_observed_steps=0)
        ms=1000*(time.perf_counter()-start);state['call']['ms']+=ms
        if state['collect']:state['call']['signal_ms']+=ms
        return state

    def _forward(self,ids,past,collect):
        torch=self.torch;state=self.active;before=int(past.get_seq_length()) if past is not None else 0;length=int(ids.shape[1])
        positions=torch.arange(before,before+length,device=ids.device,dtype=torch.int64);mask=torch.ones((1,before+length),dtype=torch.long,device=ids.device)
        with torch.inference_mode():
            result=self.model(input_ids=ids,past_key_values=past,use_cache=True,attention_mask=mask,cache_position=positions)
            logits=result.logits[:,-1,:];sample_start=time.perf_counter()
            if collect:
                if self.sampling_temperature is not None:raise RuntimeError('Observed frozen path must be greedy')
                n=state['raw_count']+1;token,pending,raw=self.pool.fused_step(logits,n,min(K,state['max_tokens']),self.signal_variant)
                state['raw_signal'][n-1]=raw;state['raw_count']=n
                if token>=V:raise RuntimeError('Non-finite FP16 vocabulary input rejected by fused argmax/top2')
                state['raw_capture_cpu'][n-1]=time.perf_counter_ns()-(raw[2]+raw[5])
            else:
                if self.sampling_temperature is None:pending=logits.argmax(-1).reshape(1,1)
                else:pending=torch.multinomial((logits.float()/self.sampling_temperature).softmax(-1),1)
                token=int(pending.item())
            sampling_ms=1000*(time.perf_counter()-sample_start) if self.sampling_temperature is not None else 0.
        return token,pending,result.past_key_values,None,0.,{'input_tokens':length,'cache_before':before,'cache_after':int(result.past_key_values.get_seq_length()),'prefill':before==0,'forward_ms':None,'sampling_ms':sampling_ms}

    def _flush(self,state,include_features=True):
        call=state['call'];count=state['raw_count']
        if state['signal_flushed']:return
        if not count:state['signal_flushed']=True;return
        start=time.perf_counter();records=[materialize_record(r,c) for r,c in zip(state['raw_signal'][:count],state['raw_capture_cpu'][:count])]
        call['signal_work_intervals']=records
        call['signal_ms']+=sum(r['work_union_upper_ns']+r['decode_cpu_ns']+r['capture_cpu_ns'] for r in records)/1e6
        if count and include_features:
            for gap,u in self.pool.values(count):state['gaps'].append(gap);state['margins'].append(u)
            call['signal_materialization_batches']=1
        state['signal_flushed']=True
        ms=1000*(time.perf_counter()-start);call['signal_deferred_record_and_values_ms']=ms;call['signal_ms']+=ms;call['ms']+=ms

    def features(self,state):
        self._flush(state)
        return super().features(state)

    def pilot(self,*args,**kwargs):
        state,features=super().pilot(*args,**kwargs);state['call']['pilot_prefix_ms']=state['call']['ms'];return state,features

    def advance(self,state,target_steps):
        try:return super().advance(state,target_steps)
        except BaseException:
            try:self._flush(state,False)
            finally:state['call'].update(signal_accounting_complete=False,incomplete_signal_cost='unknown_not_zero if a native dispatch did not complete; complete generation wall retained')
            raise
