"""Paid T4 A/B/C startup gate; C events never enter a formal method run."""
import copy,time
from v03_assets import write
from v04_v2_inference import BatchedLightweightGenerator
from v04_v3_gates import double_drop
from v04_features import torch_step_signals

def legacy_clone(g):
    c=object.__new__(BatchedLightweightGenerator);c.__dict__=dict(g.__dict__)
    c.calls=[];c.active=None
    c._signal_timers=[(g.torch.cuda.Event(enable_timing=True),g.torch.cuda.Event(enable_timing=True)) for _ in range(5)]
    for a,b in c._signal_timers:a.record();b.record()
    return c

def call_view(call):
    return {k:call.get(k) for k in ['prompt_sha256','input_tokens','output_token_ids','token_logit_gaps','token_gap_exp_uncertainty','prefill_calls','cache_final_sequence_length']}

def cost(row):
    return {'signal_ms':row['cost']['routing_total_ms'],'generation_ms':sum(c['ms'] for c in row['generation_calls']),
        'end_to_end_ms':row['cost']['end_to_end_ms']}

def numeric_gate(g):
    torch=g.torch;torch.manual_seed(42);start=time.perf_counter();checks=[]
    for mode in ['A','B']:
        for count in [1,3,5]:
            for early_eos in [False,True]:
                logits=torch.randn((count,151936),device=g.config['device'],dtype=torch.float16)
                logits[:,0]=10;logits[:,1]=10 if count==1 else 9.75
                reference=[torch_step_signals(logits[i:i+1]) for i in range(count)]
                for i in range(count):
                    token=next(iter(g.eos)) if early_eos and i==count-1 else 17
                    pending=torch.tensor([token],dtype=torch.int64,device=g.config['device'])
                    actual,_=g.pool.step(logits[i:i+1],pending,i+1,5 if early_eos else count,mode)
                    if actual!=token:raise RuntimeError('EOS packet changed token')
                values=g.pool.values(count)
                if values!=[[v[2],v[1]] for v in reference]:raise RuntimeError('Native FP32 gap/exp not bitwise identical to locked PyTorch')
                checks.append({'mode':mode,'count':count,'early_eos':early_eos,'bitwise_equal':True})
    return {'checks':checks,'ms':1000*(time.perf_counter()-start),'scope':'synthetic device signals; not QA latency'}

def startup(g,config,samples,root,append):
    start=time.perf_counter();report={'status':'running','selected':None,'GPU_measured':True}
    path=root/'logs/v3_startup.json';write(path,report)
    try:
        c=legacy_clone(g)
        report['numeric']=numeric_gate(g);observations=[]
        # Same real model/prompts/budget, balanced ABC orders, no synthetic
        # 200ms denominator. First two frozen smoke inputs, never test input.
        orders=[['A','B','C'],['B','C','A'],['C','A','B'],['C','B','A'],['B','A','C'],['A','C','B']]
        for sample in samples[:2]:
            for repeat,order in enumerate(orders):
                group={}
                for mode in order:
                    target=c if mode=='C' else g
                    if mode!='C':g.signal_variant=mode
                    target.calls=[];target.torch.manual_seed(42);target.torch.cuda.manual_seed_all(42)
                    before=time.perf_counter()
                    try:
                        target.generate(config['generation']['answer_system'],'Question: '+sample['question'],config['model']['max_answer_tokens'],'v3_ABC_real_generation')
                    finally:
                        wall_ms=1000*(time.perf_counter()-before)
                        append(root/'logs/v3_startup_calls.jsonl',{'sample_id':sample['sample_id'],'repeat':repeat,'mode':mode,'order':order,
                            'scope':'paid technical only; one execution per logged group','generation_calls':copy.deepcopy(target.calls),'attempt_wall_ms':wall_ms})
                    call=copy.deepcopy(target.calls[-1]);call['micro_attempt_wall_ms']=wall_ms;group[mode]=call
                if not (call_view(group['A'])==call_view(group['B'])==call_view(group['C'])):raise RuntimeError('Real ABC tokens/features/cache differ')
                observations.append(group)
        report['variants']={}
        for mode in ['A','B']:
            def costs(v):return {'signal_ms':v['signal_ms']+v.get('feature_materialization_ms',0.),'generation_ms':v['ms'],'end_to_end_ms':v['micro_attempt_wall_ms']}
            pairs=[{'candidate':costs(o[mode]),'v2':costs(o['C'])} for o in observations]
            report['variants'][mode]=double_drop(pairs)
        # Predetermined engineering priority; no answer quality or head tuning.
        report['selected']=next((m for m in ['A','B'] if report['variants'][m]['passed']),None)
        report['status']='passed' if report['selected'] else 'failed'
        if report['selected'] is None:raise RuntimeError('T4 ABC double-drop gate failed; 20/200 blocked')
        g.signal_variant=report['selected'];g.calls=[];return c,report
    except BaseException as exc:
        report.update(status='failed',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        report['ms']=1000*(time.perf_counter()-start);report['clock_calibrations']=g.pool.clock_records;write(path,report)
