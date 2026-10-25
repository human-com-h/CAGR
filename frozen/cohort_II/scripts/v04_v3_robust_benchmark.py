"""Fixed warmup and variance-planned balanced ABC; never retry-until-pass."""
import copy,json,subprocess,time
from v03_assets import write
from v04_v3_gates import double_drop
from v04_v3_benchmark import legacy_clone,call_view
from v04_v3_parallel_checks import parallel_numeric_gate
from v04_features import torch_step_signals
from v04_v3_robust_control import TechnicalGateRejected

ORDERS=[['A','B','C'],['B','C','A'],['C','A','B'],['C','B','A'],['B','A','C'],['A','C','B']]
CYCLES=20


def telemetry(root,label):
    start=time.perf_counter();r={'label':label,'scope':'paid preparation diagnostic, never subtracted from E2E'}
    try:
        p=subprocess.run(['nvidia-smi','--query-gpu=index,name,clocks.current.sm,clocks.current.memory,temperature.gpu,power.draw,utilization.gpu,memory.used','--format=csv'],capture_output=True,text=True,timeout=10)
        r.update(returncode=p.returncode,output=p.stdout,error=p.stderr)
    except Exception as exc:r.update(error_type=type(exc).__name__)
    r['ms']=1000*(time.perf_counter()-start)
    with (root/'logs/v3_gpu_telemetry.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(r)+'\n')
    return r


def numeric_gate(g):
    start=time.perf_counter();out=parallel_numeric_gate(g);torch=g.torch;extra=[]
    torch.manual_seed(42)
    for mode in ['A','B']:
        for count in range(1,6):
            for repeat in range(3):
                logits=torch.randn((count,151936),device=g.config['device'],dtype=torch.float16)
                logits[:,18991]=10;logits[:,18992]=10 if repeat==0 else 9.75
                ref=[torch_step_signals(logits[i:i+1]) for i in range(count)]
                for i in range(count):
                    token=next(iter(g.eos)) if repeat==2 and i==count-1 else 17
                    actual,_=g.pool.step(logits[i:i+1],torch.tensor([token],device=g.config['device'],dtype=torch.int64),i+1,5 if repeat==2 else count,mode)
                    if actual!=token:raise RuntimeError('Fused packet changed supplied greedy/EOS token')
                if g.pool.values(count)!=[[v[2],v[1]] for v in ref]:raise RuntimeError('Fused previous/current-row copy/reduce race or numeric mismatch')
                extra.append({'mode':mode,'count':count,'repeat':repeat,'bitwise_equal':True})
    out['fused_window_stress']=extra;out['ms']=1000*(time.perf_counter()-start);return out


def startup(g,config,samples,root,append):
    start=time.perf_counter();report={'status':'running','selected':None,'GPU_measured':True,'fixed_cycles':CYCLES,'measured_ABC_calls':720,'paid_warmup_calls':36}
    path=root/'logs/v3_startup.json';write(path,report);telemetry(root,'before_startup')
    try:
        c=legacy_clone(g);report['numeric']=numeric_gate(g);observations=[]
        for phase,cycles in [('preregistered_warmup',1),('measurement',CYCLES)]:
            for sample in samples[:2]:
                for repeat in range(cycles*len(ORDERS)):
                    order=ORDERS[repeat%len(ORDERS)];group={}
                    for mode in order:
                        if time.monotonic()>g.robust_process_deadline:raise RuntimeError('Frozen remote budget exhausted in ABC benchmark')
                        target=c if mode=='C' else g
                        if mode!='C':g.signal_variant=mode
                        target.calls=[];target.torch.manual_seed(42);target.torch.cuda.manual_seed_all(42);before=time.perf_counter()
                        try:target.generate(config['generation']['answer_system'],'Question: '+sample['question'],config['model']['max_answer_tokens'],'v3_ABC_real_generation')
                        finally:
                            wall_ms=1000*(time.perf_counter()-before)
                            append(root/'logs/v3_startup_calls.jsonl',{'sample_id':sample['sample_id'],'repeat':repeat,'mode':mode,'order':order,'phase':phase,
                                'scope':'one paid technical execution; warmup designated before observations, never outcome-filtered','generation_calls':target.calls,'attempt_wall_ms':wall_ms})
                        call=dict(target.calls[-1]);call['micro_attempt_wall_ms']=wall_ms;group[mode]=call
                    if not (call_view(group['A'])==call_view(group['B'])==call_view(group['C'])):raise RuntimeError('Real ABC tokens/features/cache differ')
                    if phase=='measurement':observations.append(group)
        def costs(v):return {'signal_ms':v['signal_ms']+v.get('feature_materialization_ms',0.),'generation_ms':v['ms'],'end_to_end_ms':v['micro_attempt_wall_ms']}
        report['variants']={mode:double_drop([{'candidate':costs(o[mode]),'v2':costs(o['C'])} for o in observations]) for mode in ['A','B']}
        report['selected']=next((m for m in ['A','B'] if report['variants'][m]['passed']),None)
        report['status']='passed' if report['selected'] else 'rejected'
        if report['selected'] is None:raise TechnicalGateRejected('startup',{'gate':'logs/v3_startup.json','scientific_acceptance':False,'calibration_started':False})
        g.signal_variant=report['selected'];g.calls=[];return c,report
    except TechnicalGateRejected:raise
    except BaseException as exc:report.update(status='failed',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        report['ms']=1000*(time.perf_counter()-start);report['clock_calibrations']=g.pool.clock_records;write(path,report);telemetry(root,'after_startup')
