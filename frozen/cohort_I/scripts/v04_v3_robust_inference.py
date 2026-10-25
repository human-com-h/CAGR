"""Two native kernels per observed position; same forward and signal semantics."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,time
from v04_inference import LightweightGenerator
from v04_v3_inference import K,decode_packet
from v04_v3_clockfix_inference import ClockSafeGenerator,CausalSignalPool,causal_union_ns


def load_native(torch,root):
    from torch.utils.cpp_extension import load,CUDA_HOME
    nvcc=Path(CUDA_HOME or '/missing')/'bin/nvcc'
    if not nvcc.is_file() or not shutil.which('ninja') or not shutil.which('c++'):raise RuntimeError('Existing locked offline compiler tools required; no download fallback')
    build=root/'logs/v3_fused_native_build';build.mkdir(parents=True,exist_ok=True)
    os.environ['MAX_JOBS']='1';os.environ['TORCH_CUDA_ARCH_LIST']='7.5'
    paths=[Path(__file__).parent/n for n in ['v04_v3_fused_signal_bind.cpp','v04_v3_fused_signal_cuda.cu']]
    (build/'toolchain.json').write_text(json.dumps({'nvcc':str(nvcc),'ninja':shutil.which('ninja'),'cxx':shutil.which('c++'),
        'nvcc_version':subprocess.run([str(nvcc),'--version'],capture_output=True,text=True,timeout=20).stdout,'torch':torch.__version__,
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}),encoding='utf-8')
    return load(name='cagr_v3_fused_signal',sources=[str(p) for p in paths],extra_cflags=['-O2'],extra_cuda_cflags=['-O2','--fmad=false'],build_directory=str(build),verbose=False)


class FusedPool(CausalSignalPool):
    def __init__(self,torch,native,device,eos):
        super().__init__(torch,native,device,eos)
        self.meta=torch.empty(K*8,dtype=torch.int64,device=device)

    def step(self,logits,pending,n,limit,mode):
        if mode not in ['A','B'] or not 1<=n<=limit<=K:raise RuntimeError('Invalid bounded production signal step')
        cpu_start=time.perf_counter_ns()
        self.native.dispatch(logits,pending,self.window,self.pairs,self.partials,self.meta,self.eos,self.packet,
            self.mapped_features,self.mapped_starts,n,limit,mode=='A')
        cpu_end=time.perf_counter_ns();value=int(self.packet.item())
        read_done=time.perf_counter_ns();gpu_start=int(self.starts_view[n-1]);token,gpu_end=decode_packet(value,gpu_start)
        ns,bounds=causal_union_ns(cpu_start,cpu_end,read_done,gpu_start,gpu_end)
        return token,{'cpu_start_ns':cpu_start,'cpu_end_ns':cpu_end,'shared_read_done_ns':read_done,
            'gpu_start_ns':gpu_start,'gpu_end_ns':gpu_end,'work_union_upper_ns':ns,
            'causal_offset_lower_ns':bounds[0],'causal_offset_upper_ns':bounds[1],
            'decode_cpu_ns':time.perf_counter_ns()-read_done,'shared_token_waits':1,'extra_device_syncs':0,
            'native_entry_calls':1,'native_kernel_launches':2,
            'accounting':'per-dispatch causal union upper; minimum across ALL block starts, finish follows entire work kernel'}


class RobustGenerator(ClockSafeGenerator):
    def __init__(self,*args,**kwargs):
        LightweightGenerator.__init__(self,*args,**kwargs)
        if self.signal_mode!='gap' or self.config['pilot_k']!=K:raise RuntimeError('Frozen gap/K5 required')
        self.native=load_native(self.torch,Path(__file__).resolve().parents[1]);self.pool=FusedPool(self.torch,self.native,self.config['device'],self.eos)
        self.pool.calibrate();self.signal_variant='A';self.production_event_records=0

    def begin(self,*args,**kwargs):
        state=super().begin(*args,**kwargs);start=time.perf_counter()
        state['call'].update(signal_native_kernel_launches_per_position=2,signal_clock_scratch_bytes=320,
            signal_collection='v3_two_kernel_causal_'+self.signal_variant)
        ms=1000*(time.perf_counter()-start);state['call']['ms']+=ms
        if state['collect']:state['call']['signal_ms']+=ms
        return state
