"""K5 device-side EOS dispatch and shared greedy-token readback; no events.

The tiny CUDA helper supplies bounded FP16 copy/top2, phase-end FP32 gap/exp
and device timestamps. Strict bitwise GPU gates must match frozen PyTorch
signals. No model arithmetic, prompt, head or threshold is changed.
"""
from pathlib import Path
import hashlib,math,os,shutil,time
from v04_inference import LightweightGenerator

K=5;V=151936;TIMER_BITS=44;TIMER_MASK=(1<<TIMER_BITS)-1;TOKEN_MASK=(1<<18)-1

def decode_packet(value,start_ns):
    if not isinstance(value,int) or not 0<=value<1<<62:raise RuntimeError('Invalid packed token value')
    token=value&TOKEN_MASK;end_low=(value>>18)&TIMER_MASK
    end=start_ns+((end_low-(start_ns&TIMER_MASK))&TIMER_MASK)
    if token>=V or end-start_ns>=10800*10**9:raise RuntimeError('Invalid GPU token/timing packet')
    return token,end

def union_upper_ns(cpu_start,cpu_end,gpu_start,gpu_end,offset_bounds):
    """Upper bound on UNION of CPU/GPU work, never their overlapping sum."""
    if cpu_end<cpu_start or gpu_end<gpu_start:raise ValueError('Negative clock interval')
    lo,hi=offset_bounds
    if lo>hi:raise ValueError('Invalid host/device clock enclosure')
    def union(offset):
        a,b=gpu_start+offset,gpu_end+offset
        overlap=max(0,min(cpu_end,b)-max(cpu_start,a))
        return cpu_end-cpu_start+gpu_end-gpu_start-overlap
    return max(union(lo),union(hi))

def load_native(torch,root):
    from torch.utils.cpp_extension import load,CUDA_HOME
    nvcc=Path(CUDA_HOME or '/missing')/'bin/nvcc'
    if not nvcc.is_file() or not shutil.which('ninja') or not shutil.which('c++'):
        raise RuntimeError('Locked offline image needs its existing nvcc/ninja/C++ tools for v3; no install or download fallback is allowed.')
    build=root/'logs/v3_native_build';build.mkdir(parents=True,exist_ok=True)
    os.environ['MAX_JOBS']='1';os.environ['TORCH_CUDA_ARCH_LIST']='7.5'
    source=Path(__file__).parent
    import json,subprocess
    (build/'toolchain.json').write_text(json.dumps({'nvcc':str(nvcc),'ninja':shutil.which('ninja'),'cxx':shutil.which('c++'),
        'nvcc_version':subprocess.run([str(nvcc),'--version'],capture_output=True,text=True,timeout=20).stdout,
        'torch':torch.__version__,'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [source/'v04_v3_signal_bind.cpp',source/'v04_v3_signal_cuda.cu']}}),encoding='utf-8')
    return load(name='cagr_v3_signal',sources=[str(source/'v04_v3_signal_bind.cpp'),str(source/'v04_v3_signal_cuda.cu')],
        extra_cflags=['-O2'],extra_cuda_cflags=['-O2','--fmad=false'],build_directory=str(build),verbose=False)

class SignalPool:
    def __init__(self,torch,native,device,eos):
        self.t=torch;self.native=native;self.device=device
        self.window=torch.zeros((K,V),dtype=torch.float16,device=device)
        self.pairs=torch.zeros((K,2),dtype=torch.float16,device=device)
        self.meta=torch.zeros(3,dtype=torch.int64,device=device)
        self.packet=torch.zeros(1,dtype=torch.int64,device=device)
        self.eos=torch.tensor(sorted(eos),dtype=torch.int64,device=device)
        self.host_features=torch.empty((K,2),dtype=torch.float32,pin_memory=True)
        self.host_starts=torch.empty(K,dtype=torch.int64,pin_memory=True)
        self.features_view=self.host_features.numpy();self.starts_view=self.host_starts.numpy()
        # GPU writes this pinned mailbox only at terminal K/EOS. No feature
        # copies or exp for intermediate positions, no extra host wait.
        self.mapped_features=native.mapped(self.host_features)
        self.mapped_starts=native.mapped(self.host_starts)
        self.bounds=None;self.clock_records=[]

    def calibrate(self):
        torch=self.t;out=torch.empty(1,dtype=torch.int64,device=self.device);intervals=[];start=time.perf_counter()
        for _ in range(32):
            before=time.perf_counter_ns();self.native.stamp(out);stamp=int(out.item());after=time.perf_counter_ns()
            intervals.append((before-stamp,after-stamp))
        bounds=(max(v[0] for v in intervals),min(v[1] for v in intervals))
        if bounds[0]>bounds[1]:raise RuntimeError('GPU globaltimer/host monotonic clocks did not admit a common offset; stop instead of reporting zero GPU work')
        record={'offset_lower_ns':bounds[0],'offset_upper_ns':bounds[1],'uncertainty_ns':bounds[1]-bounds[0],'ms':1000*(time.perf_counter()-start),'scope':'separate paid clock calibration, no model calls'}
        self.clock_records.append(record)
        if self.bounds is None:self.bounds=bounds
        elif not (max(self.bounds[0],bounds[0])<=min(self.bounds[1],bounds[1])):
            raise RuntimeError('Host/device clock enclosure drifted; preserve raw timing and stop acceptance')
        return record

    def step(self,logits,pending,n,limit,mode):
        if mode not in ['A','B'] or not 1<=n<=limit<=K:raise RuntimeError('Invalid bounded production signal step')
        cpu_start=time.perf_counter_ns()
        self.native.capture(logits,pending,self.window,self.meta,self.mapped_starts,n,limit,mode=='A')
        if mode=='B':self.pairs[n-1].copy_(self.t.topk(logits,2,dim=-1,sorted=True).values.reshape(-1,2)[0])
        self.native.reduce(self.window,self.pairs,self.meta,self.eos,self.mapped_features,mode=='A')
        self.native.finish(self.meta,self.packet)
        cpu_end=time.perf_counter_ns()
        value=int(self.packet.item())  # REPLACES pending.item(): one common token wait.
        decode_start=time.perf_counter_ns();gpu_start=int(self.starts_view[n-1]);token,gpu_end=decode_packet(value,gpu_start)
        ns=union_upper_ns(cpu_start,cpu_end,gpu_start,gpu_end,self.bounds)
        record={'cpu_start_ns':cpu_start,'cpu_end_ns':cpu_end,'gpu_start_ns':gpu_start,'gpu_end_ns':gpu_end,
            'work_union_upper_ns':ns,'decode_cpu_ns':time.perf_counter_ns()-decode_start,'shared_token_waits':1,'extra_device_syncs':0}
        return token,record

    def values(self,count):
        values=self.features_view[:count].tolist()  # CPU only; existing token read completed the async updates.
        if any(g<0 or not math.isfinite(g+u) for g,u in values):raise RuntimeError('Non-finite v3 feature')
        return values

class ZeroReboundGenerator(LightweightGenerator):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.signal_mode!='gap' or self.config['pilot_k']!=K:raise RuntimeError('Frozen gap/K5 required')
        root=Path(__file__).resolve().parents[1];self.native=load_native(self.torch,root)
        self.pool=SignalPool(self.torch,self.native,self.config['device'],self.eos)
        self.pool.calibrate();self.signal_variant='A';self.production_event_records=0

    def _sync(self):return None

    def begin(self,*args,**kwargs):
        state=super().begin(*args,**kwargs);start=time.perf_counter()
        state['v3_records']=[];state['v3_flushed']=False
        state['call'].update(signal_collection='v3_shared_greedy_wait_'+self.signal_variant,
            signal_materialization_batches=0,signal_added_explicit_device_syncs=0,signal_cuda_event_records=0,
            signal_clock_bounds_ns=list(self.pool.bounds),signal_buffer_bytes=K*V*2,
            signal_accounting='bounded CPU/GPU interval union + exclusive CPU materialization; GPU timing ends in the common token read packet; no overlapping CPU/GPU addition',
            forward_ms_scope='generation total wall time measured; per-forward GPU interval not isolated')
        elapsed=1000*(time.perf_counter()-start);state['call']['ms']+=elapsed
        if state['collect'] and getattr(self,'signal_collection_enabled',True):state['call']['signal_ms']+=elapsed
        return state

    def _forward(self,ids,past,collect):
        torch=self.torch;state=self.active;before=int(past.get_seq_length()) if past is not None else 0;length=int(ids.shape[1])
        positions=torch.arange(before,before+length,device=ids.device,dtype=torch.int64);mask=torch.ones((1,before+length),dtype=torch.long,device=ids.device)
        with torch.inference_mode():
            result=self.model(input_ids=ids,past_key_values=past,use_cache=True,attention_mask=mask,cache_position=positions)
            logits=result.logits[:,-1,:];sample_start=time.perf_counter()
            if self.sampling_temperature is None:pending=logits.argmax(-1).reshape(1,1)
            else:pending=torch.multinomial((logits.float()/self.sampling_temperature).softmax(-1),1)
            signal_ms=0.
            if collect:
                n=len(state['v3_records'])+1
                token,record=self.pool.step(logits,pending,n,min(K,state['max_tokens']),self.signal_variant)
                state['v3_records'].append(record);signal_ms=(record['work_union_upper_ns']+record['decode_cpu_ns'])/1e6
            else:token=int(pending.item())
            sampling_ms=1000*(time.perf_counter()-sample_start) if self.sampling_temperature is not None else 0.
        past=result.past_key_values
        return token,pending,past,None,signal_ms,{'input_tokens':length,'cache_before':before,'cache_after':int(past.get_seq_length()),'prefill':before==0,'forward_ms':None,'sampling_ms':sampling_ms}

    def features(self,state):
        start=time.perf_counter();call=state['call'];count=len(state['v3_records'])
        if count and not state['v3_flushed']:
            for gap,uncertainty in self.pool.values(count):state['gaps'].append(gap);state['margins'].append(uncertainty)
            call['signal_materialization_batches']=1;call['signal_work_intervals']=list(state['v3_records']);state['v3_flushed']=True
            elapsed=1000*(time.perf_counter()-start);call['signal_ms']+=elapsed;call['ms']+=elapsed
        return super().features(state)

    def pilot(self,*args,**kwargs):
        state,features=super().pilot(*args,**kwargs);state['call']['pilot_prefix_ms']=state['call']['ms'];return state,features

    def advance(self,state,target_steps):
        try:return super().advance(state,target_steps)
        except BaseException:
            state['call'].update(signal_work_intervals=list(state['v3_records']),signal_accounting_complete=False,
                incomplete_signal_cost='unknown_not_zero; preserved generation wall includes failed work')
            raise
