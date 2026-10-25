"""Same model/feature semantics; parallel GPU scans and one native dispatch."""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,time
from v04_inference import LightweightGenerator
from v04_v3_inference import SignalPool,ZeroReboundGenerator,decode_packet,union_upper_ns,K,V
PARTS=8

def load_parallel_native(torch,root):
    from torch.utils.cpp_extension import load,CUDA_HOME
    nvcc=Path(CUDA_HOME or '/missing')/'bin/nvcc'
    if not nvcc.is_file() or not shutil.which('ninja') or not shutil.which('c++'):raise RuntimeError('Existing locked offline compiler tools required; no download fallback')
    build=root/'logs/v3_parallel_native_build';build.mkdir(parents=True,exist_ok=True)
    os.environ['MAX_JOBS']='1';os.environ['TORCH_CUDA_ARCH_LIST']='7.5'
    source=Path(__file__).parent;paths=[source/'v04_v3_parallel_signal_bind.cpp',source/'v04_v3_parallel_signal_cuda.cu']
    (build/'toolchain.json').write_text(json.dumps({'nvcc':str(nvcc),'ninja':shutil.which('ninja'),'cxx':shutil.which('c++'),
        'nvcc_version':subprocess.run([str(nvcc),'--version'],capture_output=True,text=True,timeout=20).stdout,'torch':torch.__version__,
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}),encoding='utf-8')
    return load(name='cagr_v3_parallel_signal',sources=[str(p) for p in paths],extra_cflags=['-O2'],extra_cuda_cflags=['-O2','--fmad=false'],build_directory=str(build),verbose=False)

class ParallelSignalPool(SignalPool):
    def __init__(self,torch,native,device,eos):
        super().__init__(torch,native,device,eos)
        self.partials=torch.empty((K,PARTS,2),dtype=torch.float32,device=device)

    def step(self,logits,pending,n,limit,mode):
        if mode not in ['A','B'] or not 1<=n<=limit<=K:raise RuntimeError('Invalid bounded production signal step')
        cpu_start=time.perf_counter_ns()
        self.native.dispatch(logits,pending,self.window,self.pairs,self.partials,self.meta,self.eos,self.packet,
            self.mapped_features,self.mapped_starts,n,limit,mode=='A')
        cpu_end=time.perf_counter_ns()
        value=int(self.packet.item())  # Same shared greedy/EOS read; no extra wait.
        decode_start=time.perf_counter_ns();gpu_start=int(self.starts_view[n-1]);token,gpu_end=decode_packet(value,gpu_start)
        ns=union_upper_ns(cpu_start,cpu_end,gpu_start,gpu_end,self.bounds)
        record={'cpu_start_ns':cpu_start,'cpu_end_ns':cpu_end,'gpu_start_ns':gpu_start,'gpu_end_ns':gpu_end,'work_union_upper_ns':ns,
            'decode_cpu_ns':time.perf_counter_ns()-decode_start,'shared_token_waits':1,'extra_device_syncs':0,
            'native_entry_calls':1,'native_kernel_launches':4 if mode=='A' else 3}
        return token,record

class ParallelGenerator(ZeroReboundGenerator):
    def __init__(self,*args,**kwargs):
        # Initialize the original model once, and compile only the new helper.
        LightweightGenerator.__init__(self,*args,**kwargs)
        if self.signal_mode!='gap' or self.config['pilot_k']!=K:raise RuntimeError('Frozen gap/K5 required')
        root=Path(__file__).resolve().parents[1];self.native=load_parallel_native(self.torch,root)
        self.pool=ParallelSignalPool(self.torch,self.native,self.config['device'],self.eos)
        self.pool.calibrate();self.signal_variant='A';self.production_event_records=0

    def begin(self,*args,**kwargs):
        state=super().begin(*args,**kwargs);start=time.perf_counter()
        state['call'].update(signal_collection='v3_parallel_shared_greedy_wait_'+self.signal_variant,
            signal_partial_buffer_bytes=K*PARTS*2*4,signal_native_entry_calls_per_position=1)
        elapsed=1000*(time.perf_counter()-start);state['call']['ms']+=elapsed
        if state['collect']:state['call']['signal_ms']+=elapsed
        return state
