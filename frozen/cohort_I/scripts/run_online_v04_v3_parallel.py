"""Independent engineering repair; unchanged route/microbenchmark/statistical gates."""
import time,zipfile
import run_online_v04_v3_mountfix as prior
from v04_v3_parallel_inference import ParallelGenerator
from v04_v3_parallel_checks import parallel_numeric_gate
import v04_v3_benchmark as benchmark
from v03_assets import read,write

def archive(root=prior.entry.base.ROOT,label='snapshot'):
    path=prior.archive(root,label)
    with zipfile.ZipFile(path,'a',zipfile.ZIP_DEFLATED) as z:
        names=set(z.namelist())
        for name in read(root/'package_manifest.json')['files']:
            if name not in names:z.write(root/name,name)
    return path

def main():
    prior.entry.ZeroReboundGenerator=ParallelGenerator
    benchmark.numeric_gate=parallel_numeric_gate
    prior.main()

if __name__=='__main__':
    start=time.perf_counter();state={'status':'running'}
    try:main();state['status']='ok'
    except BaseException as exc:
        state.update(status='failed',error_type=type(exc).__name__)
        write(prior.entry.base.ROOT/'logs/fatal.json',{'error_type':type(exc).__name__,'error':str(exc)});raise
    finally:state['ms']=1000*(time.perf_counter()-start);write(prior.entry.base.ROOT/'logs/runtime_process.json',state)
