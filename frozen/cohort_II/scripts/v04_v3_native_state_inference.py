"""Persistent native K5 buffers and records; unchanged validated GPU kernels."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,time
from v04_inference import LightweightGenerator
from v04_v3_inference import K,V
from v04_v3_argmax_inference import ArgmaxPool

def load_native(torch,root):
    from torch.utils.cpp_extension import load,CUDA_HOME
    nvcc=Path(CUDA_HOME or '/missing')/'bin/nvcc'
    if not nvcc.is_file() or not shutil.which('ninja') or not shutil.which('c++'):raise RuntimeError('Existing locked offline compiler required; no download fallback')
    build=root/'logs/v3_native_state_build';build.mkdir(parents=True,exist_ok=True)
    os.environ['MAX_JOBS']='1';os.environ['TORCH_CUDA_ARCH_LIST']='7.5'
    sources=[Path(__file__).parent/n for n in ['v04_v3_native_state_bind.cpp','v04_v3_native_state_cuda.cu']]
    dependencies=sources+[Path(__file__).parent/n for n in ['v04_v3_native_state_host.h','v04_v3_argmax_cuda.cu']]
    (build/'toolchain.json').write_text(json.dumps({'nvcc':str(nvcc),'ninja':shutil.which('ninja'),'cxx':shutil.which('c++'),
        'nvcc_version':subprocess.run([str(nvcc),'--version'],capture_output=True,text=True,timeout=20).stdout,'torch':torch.__version__,
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies}}),encoding='utf-8')
    return load(name='cagr_v3_persistent_native_state',sources=[str(p) for p in sources],extra_cflags=['-O2'],extra_cuda_cflags=['-O2','--fmad=false'],build_directory=str(build),verbose=False)

class NativeStatePool(ArgmaxPool):
    def __init__(self,torch,native,device,eos):
        super().__init__(torch,native,device,eos)
        self.boundaries=torch.empty(K,dtype=torch.int64) # CPU-only boundary, never a device read.
        self.boundary_view=self.boundaries.numpy()
        self.state=native.NativeState([self.pending,self.pairs,self.partials,self.indices,self.invalid,self.meta,self.eos,self.packet,
            self.window,self.host_features,self.host_starts,self.boundaries])
        # Verify native/Python monotonic epochs by bracketing, not by assuming
        # std::chrono's unspecified epoch. Paid once during initialization.
        a=time.perf_counter_ns();b=native.host_clock_ns();c=time.perf_counter_ns()
        if not a<=b<=c:raise RuntimeError('Native/Python monotonic clock epoch differs')
        self.clock_epoch_check={'python_before_ns':a,'native_ns':b,'python_after_ns':c,'same_epoch':True}

    def fused_step(self,logits,n,limit,mode):
        # Compatibility for the frozen 120-case GPU gate ONLY. Formal inference
        # calls the bound native method directly and never materializes raw tuples.
        if mode not in ['A','B'] or not 1<=n<=limit<=K:raise RuntimeError('Invalid frozen K5 signal step')
        token=self.state.step(logits,n,limit,mode=='A',time.perf_counter_ns())
        self.boundary_view[n-1]=time.perf_counter_ns()
        return token,self.pending,self.state.last_raw()

class NativeStateGenerator(LightweightGenerator):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.signal_mode!='gap' or self.config['pilot_k']!=K:raise RuntimeError('Frozen gap/K5 required')
        self.native=load_native(self.torch,Path(__file__).resolve().parents[1]);self.pool=NativeStatePool(self.torch,self.native,self.config['device'],self.eos)
        self.native_step=self.pool.state.step # Bind once; avoid repeated method lookup.
        self.pool.calibrate();self.signal_variant='A';self.production_event_records=0

    def _sync(self):return None

    def begin(self,*args,**kwargs):
        state=super().begin(*args,**kwargs);start=time.perf_counter()
        state['raw_count']=0;state['signal_flushed']=False
        state['call'].update(signal_collection='v3_fused_argmax_top2_'+self.signal_variant,
            signal_materialization_batches=0,signal_added_explicit_device_syncs=0,signal_cuda_event_records=0,
            signal_clock_bounds_ns=list(self.pool.bounds),signal_buffer_bytes=K*V*2,signal_partial_buffer_bytes=320,
            signal_index_buffer_bytes=160,signal_invalid_buffer_bytes=160,signal_clock_scratch_bytes=320,
            signal_pending_buffer_bytes=8,signal_native_entry_calls_per_position=1,signal_native_kernel_launches_per_position=2,
            signal_accounting='per-dispatch causal CPU/GPU union upper + exclusive deferred record/materialization work',
            signal_clock_bounds_scope='diagnostic only; every dispatch independently enclosed at stage end',
            forward_ms_scope='generation total wall measured; per-forward interval not isolated',
            greedy_selection='fused argmax lowest index; original argmax retained outside observed greedy steps',
            signal_host_records='persistent native K5 fixed records; stage-end dictionaries',signal_full_vocab_argmax_calls_on_observed_steps=0,
            signal_fixed_buffers_bound_once=True,signal_native_clock_epoch_check=self.pool.clock_epoch_check,
            signal_host_boundary_buffer_bytes=40,signal_fixed_buffer_tensor_args_per_position=0)
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
                n=state['raw_count']+1
                token=self.native_step(logits,n,min(K,state['max_tokens']),self.signal_variant=='A',time.perf_counter_ns())
                pending=self.pool.pending;state['raw_count']=n
                # This CPU-only fixed slot includes pybind return + Python
                # capture/metadata in native stage-end capture_cpu_ns. No
                # features/packet readback or GPU event is performed here.
                self.pool.boundary_view[n-1]=time.perf_counter_ns()
                if token>=V:raise RuntimeError('Non-finite FP16 vocabulary input rejected by fused argmax/top2')
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
        start=time.perf_counter();batch=self.pool.state.finalize(count,include_features)
        call['signal_work_intervals']=batch['records'];call['signal_ms']+=batch['charged_ns']/1e6
        if include_features:
            for gap,u in batch['values']:state['gaps'].append(gap);state['margins'].append(u)
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
