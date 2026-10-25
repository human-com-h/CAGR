"""Independent fused greedy/top2 entry; unchanged complete 20-to-200 controller."""
from pathlib import Path
import time
import run_online_v04_v3 as v3
import run_online_v04_v3_mountfix as mounts
import run_online_v04_v3_clockfix as archived_entry
from v04_v3_argmax_inference import ArgmaxGenerator
from v04_v3_argmax_benchmark import startup,telemetry
from v04_v3_robust_pairs import paired_path,analyze
from v04_v3_robust_control import TechnicalGateRejected,guarded_partition
from v03_assets import read,write,sha

PROCESS_DEADLINE=None


class Generator(ArgmaxGenerator):
    def __init__(self,*args,**kwargs):
        self.robust_process_deadline=PROCESS_DEADLINE
        super().__init__(*args,**kwargs)


def run(partition,g,*args,**kwargs):
    try:return guarded_partition(v3.wrapped_run,v3.base.ROOT,partition,g,*args,**kwargs)
    finally:telemetry(v3.base.ROOT,'partition_'+partition+'_exit')


def archive(root=v3.base.ROOT,label='snapshot'):return archived_entry.archive(root,label)


def install():
    # One explicit composition point; do not call older monkey-patching mains.
    base=v3.base;v3.startup=startup
    base.LightweightGenerator=Generator;base.run_method=v3.wrapped_method;base.run=run
    base.analyze=analyze;base.archive=archive;base.paired_path=paired_path;base.order=v3.order
    base.prepare_mounted_assets=mounts.prepare_assets


def main():
    global PROCESS_DEADLINE
    PROCESS_DEADLINE=time.monotonic()+10800;install();start=time.perf_counter()
    result={'runtime_status':'running','scientific_acceptance':'pending','held_out_test_used':False}
    try:
        v3.base.main()
        summary_path=v3.base.ROOT/'results/online-v04/calibration/summary.json'
        if summary_path.is_file():
            final=read(summary_path)['final_freeze']
            result.update(runtime_status='completed',scientific_acceptance='passed_main_freeze_criteria' if final['method_frozen'] else 'not_met_final_freeze_criteria',
                method_frozen=final['method_frozen'],calibration_started=True,calibration_completed=True,final_freeze=final,
                summary_sha256=sha(summary_path))
        else:
            result.update(runtime_status='completed',scientific_acceptance='technical_stage_only_not_method_accepted',method_frozen=False,calibration_started=False)
    except TechnicalGateRejected as exc:
        result.update(runtime_status='completed_controlled_gate_stop',scientific_acceptance='rejected',rejected_stage=exc.stage,evidence=exc.evidence,
            calibration_started=False,method_frozen=False)
        print('VALIDATION REJECTED: unchanged technical gates; calibration NOT executed. See logs/final_status.json.',flush=True)
    except BaseException as exc:
        result.update(runtime_status='failed',scientific_acceptance='not_accepted',error_type=type(exc).__name__)
        write(v3.base.ROOT/'logs/fatal.json',{'error_type':type(exc).__name__,'error':str(exc)});raise
    finally:
        result['ms']=1000*(time.perf_counter()-start);write(v3.base.ROOT/'logs/final_status.json',result)
        write(v3.base.ROOT/'logs/runtime_process.json',result)
    return result


if __name__=='__main__':main()
