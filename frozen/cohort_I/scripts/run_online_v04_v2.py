"""Independent v2 bindings around the frozen runner, no cloud submission API.

All overrides are explicit, process-local and used only by this new entry.
The v1 files/Notebook/protocol are not edited. Common budgets and accounting
remain in the shared frozen runner/runtime.
"""
from pathlib import Path
import json,time,zipfile
import numpy as np
import run_online_v04 as base
from v03_assets import read,write,sha
from v04_v2_inference import BatchedLightweightGenerator,StageSignalBuffer
from v04_v2_runtime import run_method as execute
from v04_features import torch_step_signals

PARTITION=None;VERIFIED={};REFERENCES={};SAMPLES={}
REFERENCE_SCOPE='normal'
OLD_RUN=base.run;OLD_ANALYZE=base.analyze;OLD_ARCHIVE=base.archive
OLD_PAIRED_PATH=base.paired_path

def wrapped_paired_path(*args,**kwargs):
    """Keep feature-on/off technical runs outside the OOF policy oracle.

    The frozen runner intentionally uses the full fitted router for these
    paired timing runs. Normal calibration predictions use their held-out
    fold heads. The technical pair still checks identical tokens, contexts
    and paths internally; it must not be compared to a different fitted head.
    """
    global REFERENCE_SCOPE
    previous=REFERENCE_SCOPE;REFERENCE_SCOPE='technical_pair'
    try:return OLD_PAIRED_PATH(*args,**kwargs)
    finally:REFERENCE_SCOPE=previous

def archive(root=base.ROOT,label='snapshot'):
    path=OLD_ARCHIVE(root,label)
    with zipfile.ZipFile(path,'a',zipfile.ZIP_DEFLATED) as z:
        for p in (root/'data/v1_equivalence').glob('*.json'):z.write(p,p.relative_to(root))
    return path

def wrapped_method(method,question,generator,retriever,config,budget,parameters,**kwargs):
    row=execute(method,question,generator,retriever,config,budget,parameters,**kwargs)
    if row['status'] not in {'ok','budget_limited_answer'}:return row
    # The caller supplies sample_id after this function; the question key below
    # is a technical index only, built from the current authorized partition.
    if REFERENCE_SCOPE=='normal' and method in {'cagr','cagr_in_situ','binary_gate','targ_style'} and kwargs.get('technical_plan') is None:
        verification_start=time.perf_counter()
        if PARTITION not in REFERENCES:
            REFERENCES[PARTITION]={(r['sample_id'],r['method']):r for r in read(base.ROOT/f'data/v1_equivalence/{PARTITION}.json')['rows']}
        sample=SAMPLES[question]
        expected=REFERENCES[PARTITION][sample['sample_id'],method]
        def reject(message):
            write(base.ROOT/'logs/v2_equivalence_failure.json',{'partition':PARTITION,'sample_id':sample['sample_id'],'method':method,'reason':message,'paid_attempt':row})
            raise RuntimeError('Strict v1/v2 equivalence failed: '+message)
        for key in ['answer','tier','context_ids']:
            if row[key]!=expected[key]:reject(key)
        if len(row['generation_calls'])!=len(expected['generation_calls']):reject('call count')
        for actual,ref in zip(row['generation_calls'],expected['generation_calls']):
            for key in ['kind','prompt_sha256','input_tokens','output_token_ids','token_logit_gaps','token_gap_exp_uncertainty']:
                if actual.get(key)!=ref.get(key):reject(key)
        decisions=[d for d in row['control_trace'] if d['kind']=='decision_value']
        if decisions!=expected['decisions']:reject('frozen decision values')
        VERIFIED[PARTITION][sample['sample_id'],method]=len(decisions)
        base.append(base.ROOT/'logs/v2_equivalence_check_costs.jsonl',{'partition':PARTITION,'sample_id':sample['sample_id'],'method':method,'ms':1000*(time.perf_counter()-verification_start),'scope':'post-prediction technical verification; no model calls, not router latency'})
    return row

def gpu_signal_gate(g):
    torch=g.torch;start=time.perf_counter();torch.manual_seed(42);positions=0
    for count in [1,3,5]:
        # Include tied maxima and a known FP16 subtraction case.
        logits=torch.randn((count,151936),device=g.config['device'],dtype=torch.float16)
        logits[:,0]=10;logits[:,1]=10 if count==1 else 9.75
        old=[torch_step_signals(logits[i:i+1]) for i in range(count)]
        buffer=StageSignalBuffer(torch)
        for i in range(count):buffer.capture(logits[i:i+1])
        new=buffer.materialize()
        if [(v[2],v[1]) for v in old]!=[tuple(v) for v in new]:raise RuntimeError('GPU batched gap/diagnostic uncertainty differs from v1')
        positions+=count
    write(base.ROOT/'logs/v2_gpu_signal_gate.json',{'status':'passed','positions':positions,'same_FP16_top2_FP32_gap_and_exp':True,'ms':1000*(time.perf_counter()-start),'scope':'separate technical GPU work, no LLM calls, not normal inference latency'})

def wrapped_run(partition,*args,**kwargs):
    global PARTITION,SAMPLES
    PARTITION=partition;VERIFIED[partition]={}
    if partition=='smoke':gpu_signal_gate(args[0])
    else:
        base.require_smoke()
        receipt=read(base.ROOT/'results/online-v04/smoke/v2_equivalence.json')
        if receipt['status']!='passed':raise RuntimeError('20 strict equivalence gate required before 200')
    SAMPLES={s['question']:s for s in base.samples(partition)}
    OLD_RUN(partition,*args,**kwargs)
    expected=REFERENCES[partition]
    if set(VERIFIED[partition])!=set(expected):raise RuntimeError('Not all expected v1 paths were verified')
    decisions=sum(VERIFIED[partition].values());out=base.ROOT/f'results/online-v04/{partition}';p=out/'v2_equivalence.json'
    write(p,{'status':'passed','method_question_pairs':len(expected),'frozen_decisions_100_percent_equal':decisions,'tokens_features_contexts_100_percent_equal':True,'scope':'four frozen routed methods; new query arms have new answers and separate scientific metrics'})
    receipt=read(out/'acceptance.json');receipt['evidence_sha256'][p.name]=sha(p);write(out/'acceptance.json',receipt)
    if partition=='calibration' and sum(sum(v.values()) for v in VERIFIED.values())!=1058:raise RuntimeError('Expected exactly 1058 original decisions')

def analyze(scored,technical,methods):
    result=OLD_ANALYZE(scored,technical,methods);idx={(r['sample_id'],r['method']):r for r in scored}
    for ds in result['methods']:
        ids=sorted({r['sample_id'] for r in scored if r['dataset']==ds})
        for method in ['cagr_doc_bridge','cagr_compact']:
            result['quality_mean_floor'][ds][method]={k:result['methods'][ds][method][k]>=result['methods'][ds]['always_rag'][k]-.01-1e-12 for k in ['em','f1']}
            result['overhead'][ds][method]={'full_components':base.bootstrap([idx[s,method]['cost']['routing_total_ms']/idx[s,method]['cost']['end_to_end_ms'] for s in ids])}
            for other in ['binary_gate','targ_style','always_rag','always_multihop','sugar_l_style_budgeted']:
                result['paired'][ds][method+'_minus_'+other]={metric:base.bootstrap([idx[s,method][key][metric]-idx[s,other][key][metric] for s in ids]) for key,metric in [('quality','f1'),('quality','em'),('cost','end_to_end_ms'),('cost','retrieval_calls')]}
    result['v2_scope']='same frozen heads/thresholds; new query arms are independently generated, not inherited quality'
    result['freeze_eligibility']={}
    for method in ['cagr_compact','cagr','cagr_doc_bridge']:
        entries=[r for r in scored if r['method']==method]
        mean_ms=float(np.mean([r['cost']['end_to_end_ms'] for r in entries]))
        floor=all(all(result['quality_mean_floor'][ds][method].values()) for ds in result['methods'])
        overhead=all(result['overhead'][ds][method]['full_components']['upper_one_sided95']<.005 for ds in result['methods'])
        result['freeze_eligibility'][method]={'quality_mean_floor':floor,'complete_overhead_upper_below_0_5_percent':overhead,'mean_latency_ms':mean_ms,'latency_below_580ms':mean_ms<580,'stretch_below_550ms':mean_ms<550,'eligible_after_complete_200_equivalence':PARTITION=='calibration' and floor and overhead and mean_ms<580,'test_unsealed':False}
    return result

def main():
    base.LightweightGenerator=BatchedLightweightGenerator;base.run_method=wrapped_method;base.run=wrapped_run;base.analyze=analyze;base.archive=archive;base.paired_path=wrapped_paired_path
    base.main()

if __name__=='__main__':
    start=time.perf_counter();state={'status':'running'}
    try:main();state['status']='ok'
    except BaseException as exc:
        state.update(status='failed',error_type=type(exc).__name__);write(base.ROOT/'logs/fatal.json',{'error_type':type(exc).__name__,'error':str(exc)});raise
    finally:state['ms']=1000*(time.perf_counter()-start);write(base.ROOT/'logs/runtime_process.json',state)
