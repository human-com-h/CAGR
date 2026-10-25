"""Independent, once-authorized v3 entry; no submission API or test loader."""
from pathlib import Path
import json,time,zipfile
import numpy as np
import run_online_v04 as base
from v04_runtime import run_method as execute
from v03_assets import read,write,sha
from v04_v3_inference import ZeroReboundGenerator
from v04_v3_gates import ROUTED,scientific_view,scientific_sha256,double_drop,freeze_gate
from v04_v3_benchmark import startup,cost

PARTITION=None;SAMPLES={};REFERENCES={};VERIFIED={};SCOPE='normal';LEGACY=None;STARTUP=None;COMPARISONS=[]
OLD_RUN=base.run;OLD_ANALYZE=base.analyze;OLD_ARCHIVE=base.archive;OLD_PAIR=base.paired_path;OLD_ORDER=base.order
V2_ORDER=['direct','always_rag','always_multihop','binary_gate','targ_style','sugar_l_style_budgeted','cagr','cagr_in_situ','cagr_doc_bridge','cagr_compact']

def archive(root=base.ROOT,label='snapshot'):
    path=OLD_ARCHIVE(root,label)
    with zipfile.ZipFile(path,'a',zipfile.ZIP_DEFLATED) as z:
        for directory in ['data/v1_equivalence','data/v3_equivalence']:
            for p in (root/directory).glob('*.json'):z.write(p,p.relative_to(root))
        if (root/'docs/protocol_v04_v3.md').exists():z.write(root/'docs/protocol_v04_v3.md','docs/protocol_v04_v3.md')
    return path

def order(index,methods):return [m for m in OLD_ORDER(index,V2_ORDER) if m in methods]

def wrapped_method(method,question,generator,retriever,config,budget,parameters,**kwargs):
    row=execute(method,question,generator,retriever,config,budget,parameters,**kwargs)
    if SCOPE=='normal' and kwargs.get('technical_plan') is None:
        start=time.perf_counter();sample=SAMPLES[question];key=(sample['sample_id'],method)
        if scientific_sha256(row)!=REFERENCES[PARTITION][key]:
            write(base.ROOT/'logs/v3_equivalence_failure.json',{'partition':PARTITION,'key':list(key),'paid_attempt':row})
            raise RuntimeError('Strict v2/v3 tokens/features/decisions/retrieval trace equality failed')
        VERIFIED[PARTITION][key]=len([d for d in row['control_trace'] if d['kind']=='decision_value'])
        for call in row['generation_calls']:
            if call.get('signal_cuda_event_records')!=0 or call.get('signal_added_explicit_device_syncs')!=0:raise RuntimeError('Formal v3 path added event/sync')
            if call.get('signal_buffer_bytes')!=5*151936*2:raise RuntimeError('Formal signal buffer differs from K5 FP16')
        base.append(base.ROOT/'logs/v3_equivalence_check_costs.jsonl',{'partition':PARTITION,'key':list(key),'ms':1000*(time.perf_counter()-start),'scope':'post-prediction verification, not routing; no model calls'})
    return row

def paired_path(sample,g,retriever,config,study,budget,router,index):
    global SCOPE
    previous=SCOPE;SCOPE='technical_pair'
    baseline=None
    def legacy_run():
        LEGACY.calls=[];before=time.perf_counter()
        try:
            row=execute('cagr',sample['question'],LEGACY,retriever,config,budget,router,study=study)
            base.validate_row(row,budget);return row
        finally:
            base.append(base.ROOT/'logs/v3_legacy_comparisons.jsonl',{'sample_id':sample['sample_id'],'generation_calls':list(LEGACY.calls),
                'attempt_wall_ms':1000*(time.perf_counter()-before),'scope':'additional paid C technical comparator; not normal method cost'})
    try:
        if PARTITION=='smoke' and index%2:baseline=legacy_run()
        result=OLD_PAIR(sample,g,retriever,config,study,budget,router,index)
        if PARTITION=='smoke':
            if baseline is None:baseline=legacy_run()
            actual=result['pairs']['True']
            if scientific_view(actual)!=scientific_view(baseline):raise RuntimeError('Same-T4 C comparator tokens/features/retrieval/decisions differ')
            pair={'sample_id':sample['sample_id'],'dataset':sample['dataset'],'candidate':cost(actual),'v2':cost(baseline),
                'order':'C-before' if index%2 else 'C-after','baseline_row':baseline,'scope':'extra C execution; not duplicated in base technical totals'}
            COMPARISONS.append(pair);base.append(base.ROOT/'logs/v3_technical_double_drop.jsonl',pair)
        return result
    finally:SCOPE=previous

def analyze(scored,technical,methods):
    result=OLD_ANALYZE(scored,technical,methods)
    result['v3_unique_main']='cagr';result['archived_development_arms']=['cagr_compact','cagr_doc_bridge']
    if PARTITION=='smoke':
        measured=double_drop(COMPARISONS)
        overhead=all(result['overhead'][ds]['cagr']['full_components']['upper_one_sided95']<.005 and
            result['overhead'][ds]['paired_e2e']['upper_one_sided95']<.005 for ds in result['methods'])
        measured['per_dataset_complete_and_independent_on_off_passed']=overhead
        measured['passed']=bool(measured['passed'] and overhead and STARTUP['status']=='passed')
        write(base.ROOT/'logs/v3_20_double_drop.json',measured);result['v3_20_double_drop']=measured
    result['full_cost_totals']['v3_extra_technical']={
        'ABC_generations':'logs/v3_startup_calls.jsonl: each line one execution; all A/B/C calls including failures',
        'C_comparators':'logs/v3_technical_double_drop.jsonl: baseline_row is one extra C execution; legacy_comparisons is duplicate failure-resilient call evidence, do not sum twice',
        'native_compile_and_clock':'logs/preparation.json model_load includes offline native compile/pinned allocation/initial clock; v3_clock.json later paid clock checks',
        'diagnostics':'all raw CPU/GPU work intervals and clock bounds retained; overlap union never subtracted from end_to_end_ms'}
    return result

def wrapped_run(partition,g,*args,**kwargs):
    global PARTITION,SAMPLES,LEGACY,STARTUP
    PARTITION=partition;VERIFIED[partition]={};SAMPLES={s['question']:s for s in base.samples(partition)}
    reference=read(base.ROOT/f'data/v3_equivalence/{partition}.json')
    REFERENCES[partition]={(r['sample_id'],r['method']):r['scientific_sha256'] for r in reference['rows']}
    if partition=='smoke':LEGACY,STARTUP=startup(g,args[1],base.samples(partition),base.ROOT,base.append)
    else:
        base.require_smoke();proof=read(base.ROOT/'results/online-v04/smoke/v3_gate.json')
        if proof['status']!='passed':raise RuntimeError('v3 T4/20 double-drop hard gate blocks 200')
    OLD_RUN(partition,g,*args,**kwargs)
    out=base.ROOT/f'results/online-v04/{partition}'
    if set(VERIFIED[partition])!=set(REFERENCES[partition]):raise RuntimeError('Frozen method coverage incomplete')
    routed=[v for key,v in VERIFIED[partition].items() if key[1] in ROUTED]
    write(out/'v3_equivalence.json',{'status':'passed','all_method_question_pairs':len(VERIFIED[partition]),'routed_pairs':len(routed),
        'routed_decisions':sum(routed),'exact_tokens_features_decisions_queries_documents_scores':True})
    clock=g.pool.calibrate();base.append(base.ROOT/'logs/v3_clock.jsonl',{'partition':partition,**clock})
    if partition=='smoke':passed=read(base.ROOT/'logs/v3_20_double_drop.json')['passed']
    else:
        passed=read(base.ROOT/'results/online-v04/smoke/v3_gate.json')['status']=='passed'
        pairs=sum(sum(key[1] in ROUTED for key in values) for values in VERIFIED.values())
        decisions=sum(sum(v for key,v in values.items() if key[1] in ROUTED) for values in VERIFIED.values())
        if (pairs,decisions)!=(880,1058):raise RuntimeError('Expected exactly 880 routed pairs / 1058 decisions')
        summary=read(out/'summary.json');scored=base.read_jsonl(out/'scored.jsonl')
        latency=lambda m:float(np.mean([r['cost']['end_to_end_ms'] for r in scored if r['method']==m]))
        summary['final_freeze']=freeze_gate(summary,latency('cagr'),latency('always_rag'),True,passed,partition)
        main=[r for r in scored if r['method']=='cagr'];tiers=np.bincount([r['tier'] for r in main],minlength=3).tolist()
        idx={(r['sample_id'],r['method']):r for r in scored}
        delta=[r['quality']['f1']-idx[r['sample_id'],'always_rag']['quality']['f1'] for r in main if r['tier']==2]
        outcomes=[sum(v>1e-12 for v in delta),sum(abs(v)<=1e-12 for v in delta),sum(v<-1e-12 for v in delta)]
        if tiers!=[118,63,19] or outcomes!=[2,17,0] or sum(r['cost']['retrieval_calls'] for r in main)!=101:raise RuntimeError('Frozen branch/quality/retrieval invariant changed')
        summary['final_freeze'].update(tiers=tiers,upgrade_positive_tie_negative=outcomes,retrieval_reduction=.495)
        write(out/'summary.json',summary)
    write(out/'v3_gate.json',{'status':'passed' if passed else 'failed','partition':partition,'test_executed':False})
    receipt=read(out/'acceptance.json')
    for name in ['summary.json','v3_equivalence.json','v3_gate.json']:receipt['evidence_sha256'][name]=sha(out/name)
    write(out/'acceptance.json',receipt)
    if not passed:raise RuntimeError('20 technical no-rebound/overhead gate failed; 200 blocked')

def main():
    base.LightweightGenerator=ZeroReboundGenerator;base.run_method=wrapped_method;base.run=wrapped_run
    base.analyze=analyze;base.archive=archive;base.paired_path=paired_path;base.order=order
    base.main()

if __name__=='__main__':
    start=time.perf_counter();state={'status':'running'}
    try:main();state['status']='ok'
    except BaseException as exc:
        state.update(status='failed',error_type=type(exc).__name__);write(base.ROOT/'logs/fatal.json',{'error_type':type(exc).__name__,'error':str(exc)});raise
    finally:state['ms']=1000*(time.perf_counter()-start);write(base.ROOT/'logs/runtime_process.json',state)
