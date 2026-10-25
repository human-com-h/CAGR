"""Independent clock-safe engineering retry with fixed repeated metrology."""
import time,zipfile
import run_online_v04_v3_parallel as prior
import run_online_v04_v3 as v3
import v04_v3_benchmark as benchmark
from v04_v3_clockfix_inference import ClockSafeGenerator
from v04_v3_clockfix_pairs import paired_path,analyze
from v03_assets import read,write


class BudgetedClockSafeGenerator(ClockSafeGenerator):
    def __init__(self,*args,**kwargs):
        self.clockfix_process_deadline=PROCESS_DEADLINE
        super().__init__(*args,**kwargs)


def archive(root=v3.base.ROOT,label='snapshot'):
    path=prior.archive(root,label)
    with zipfile.ZipFile(path,'a',zipfile.ZIP_DEFLATED) as z:
        names=set(z.namelist())
        for name in read(root/'package_manifest.json')['files']:
            if name not in names:z.write(root/name,name)
    return path


def main():
    global PROCESS_DEADLINE
    PROCESS_DEADLINE=time.monotonic()+10800
    # Install after all frozen adapters, before their main binds base entries.
    v3.ZeroReboundGenerator=BudgetedClockSafeGenerator
    benchmark.numeric_gate=prior.parallel_numeric_gate
    v3.paired_path=paired_path
    # analyze calls the original v3 function, not itself.
    original_analyze=v3.analyze
    import v04_v3_clockfix_pairs as pairs_module
    pairs_module.original_analyze=original_analyze
    v3.analyze=analyze
    prior.prior.main()


if __name__=='__main__':
    start=time.perf_counter();state={'status':'running'}
    try:main();state['status']='ok'
    except BaseException as exc:
        state.update(status='failed',error_type=type(exc).__name__)
        write(v3.base.ROOT/'logs/fatal.json',{'error_type':type(exc).__name__,'error':str(exc)});raise
    finally:state['ms']=1000*(time.perf_counter()-start);write(v3.base.ROOT/'logs/runtime_process.json',state)
