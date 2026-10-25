"""Prospective fine v2: technical20 -> formal200, one deployment on/off path."""
from pathlib import Path
import json,time,platform,random,collections,hashlib,sys
import numpy as np,yaml
from v03_assets import read,write,sha
from run_online_v04 import append,samples,parameters,fingerprint,cache_check,bootstrap
from run_baselines import Retriever,answer_metrics
from v04_runtime import run_method
from v2_fine_runtime import run_fine,selected_plan,science
from kv_bridge_inference import KVBridgeGenerator
from route23_runtime import validate_row as validate_route23_row
from test1000_io_assets import prepare_mounted_assets
from v04_v3_mounts import checked_resolved_mounts
from v04_v3_native_state_checks import numeric_gate
from v03_dependencies import check_versions
from execution_policy import validate_online_policy
from run_route23_collection import archive
from runtime_policy_session import RuntimePolicySession
ROOT=Path(__file__).resolve().parents[1]
METHODS=['direct','always_rag','always_multihop','matched_binary_v2','targ_style','sugar_l_style_budgeted','cagr_v1','cagr_v2','cagr_v2_full','matched_binary_v2_full','targ_style_full']


def validate_execution(row,budget):
    # A frozen budget-adapted SUGAR stop is an executed answer, not a backend failure.
    if row['status'] not in ['ok','budget_limited_answer']:raise RuntimeError('Execution failed: '+str(row.get('error')))
    check={**row,'status':'ok'};validate_route23_row(check,budget)
    assert row['cost']['total_input_tokens']==sum(c['input_tokens'] for c in row['generation_calls'])
    assert row['cost']['total_output_tokens']==sum(c['output_tokens'] for c in row['generation_calls'])


def dispatch(method,s,g,retriever,config,study,budget,fine,router,baselines,part,plan=None):
    fold=s.get('fold') if part=='calibration' and not method.endswith('_full') else None
    if method.startswith(('cagr_v2','matched_binary_v2')):return run_fine(s['question'],s['dataset'],g,retriever,config,budget,fine,study,fold,method.startswith('matched_binary'),plan)
    name='cagr' if method=='cagr_v1' else 'targ_style' if method=='targ_style_full' else method
    p=parameters(name,router,baselines,fold)
    return run_method(name,s['question'],g,retriever,config,budget,p,study=study,fold=fold if name=='cagr' else None)


def verify_expected(row,e):
    view={k:science(row)[k] for k in ['answer','context_ids','calls','queries']}
    digest=hashlib.sha256(json.dumps(view,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
    assert digest==e['science_sha256'],'Frozen answer/context/prompt/Token/query drift'
    calls=row['generation_calls']
    if calls and calls[0]['kind']=='pilot':assert calls[0]['token_logit_gaps']==e['pilot_gaps'],'Pilot feature drift'
    if e['H1_gaps'] is not None:
        c=next(c for c in calls if c['kind']=='rag1');assert c['token_logit_gaps']==e['H1_gaps'],'H1 feature drift'
    if e['fine_plan'] is not None:assert selected_plan(row)==e['fine_plan']


def report_partition(rows,pairs,part):
    indexed={(r['sample_id'],r['method']):r for r in rows};report={'methods':{},'overhead':{},'quality':{},'scope':'measured same-card development normal executions','partition':part}
    for ds in ['hotpotqa','triviaqa','overall']:
        ids=sorted({r['sample_id'] for r in rows if ds=='overall' or r['dataset']==ds});report['methods'][ds]={}
        for method in METHODS:
            rr=[indexed[sid,method] for sid in ids];report['methods'][ds][method]={**{k:float(np.mean([r['quality'][k] for r in rr])) for k in ['em','f1']},
                'end_to_end_ms':float(np.mean([r['cost']['end_to_end_ms'] for r in rr])), 'retrieval_calls':float(np.mean([r['cost']['retrieval_calls'] for r in rr])),
                'tiers':np.bincount([r['tier'] for r in rr],minlength=3).tolist()}
        report['overhead'][ds]={};report['quality'][ds]={}
        for method in ['cagr_v2','cagr_v2_full']:
            comp=bootstrap([indexed[sid,method]['cost']['routing_total_ms']/indexed[sid,method]['cost']['end_to_end_ms'] for sid in ids])
            report['overhead'][ds][method]={'full_components':comp,'passed':comp['upper_one_sided95']<.005}
            rr=report['methods'][ds];report['quality'][ds][method]={k:rr[method][k]>=rr['always_rag'][k]-.01-1e-12 for k in ['em','f1']}
        vals=[r['question_mean_ratio'] for r in pairs if r['method']=='cagr_v2_full' and r['sample_id'] in ids]
        ind=bootstrap(vals)
        report['overhead'][ds]['independent_deployment_full']={'independent_on_off':ind,
            'passed':ind['upper_one_sided95']<.005,'hard_gate':part=='calibration',
            'question_count':len(vals),'pairs_per_question':4 if part=='smoke' else 12,
            'physical_pairs':len(vals)*(4 if part=='smoke' else 12)}
    complete=all(vv[m]['passed'] for ds,vv in report['overhead'].items() if ds!='overall' for m in ['cagr_v2','cagr_v2_full'])
    independent=all(vv['independent_deployment_full']['passed'] for ds,vv in report['overhead'].items() if ds!='overall')
    quality=all(all(v.values()) for ds,vv in report['quality'].items() if ds!='overall' for v in vv.values())
    comparison=True
    for ds in ['hotpotqa','overall']:
        rr=report['methods'][ds]
        for method,other in [('cagr_v2','matched_binary_v2'),('cagr_v2_full','matched_binary_v2_full')]:
            targ='targ_style_full' if method.endswith('_full') else 'targ_style'
            comparison &= all(rr[method][k]>=rr[other][k]-1e-12 and rr[method][k]>=rr[targ][k]-1e-12 for k in ['em','f1'])
    rr=report['methods']['overall'];anchors={}
    for m in ['cagr_v2','cagr_v2_full']:
        raw=rr[m]['end_to_end_ms'];rag=rr['always_rag']['end_to_end_ms'];targ=rr['targ_style']['end_to_end_ms']
        adjusted=raw/rag*588.64
        anchors[m]={'raw_mean_ms':raw,'same_batch_RAG_mean_ms':rag,'same_batch_OOF_TARG_mean_ms':targ,
                    'RAG_anchor_ms':588.64,'normalized_mean_ms':adjusted,'raw_below550':raw<550,
                    'normalized_below550':adjusted<550,'raw_below_both_comparators':raw<rag and raw<targ,
                    'passed':(raw<550 or adjusted<550) and raw<rag and raw<targ,
                    'same_batch_Full_TARG_difference_ms':raw-rr['targ_style_full']['end_to_end_ms']}
    latency=all(v['passed'] for v in anchors.values());report['latency_anchors']=anchors
    upgrades={}
    for method in ['cagr_v2','cagr_v2_full']:
        binary='matched_binary_v2_full' if method.endswith('_full') else 'matched_binary_v2';ids=sorted({r['sample_id'] for r in rows})
        delta=[indexed[sid,method]['quality']['f1']-indexed[sid,binary]['quality']['f1'] for sid in ids]
        up=[delta[i] for i,sid in enumerate(ids) if indexed[sid,method]['tier']==2]
        nonup=[delta[i] for i,sid in enumerate(ids) if indexed[sid,method]['tier']!=2]
        upgrades[method]={'upgrades':len(up),'positive':sum(x>1e-12 for x in up),'tie':sum(abs(x)<=1e-12 for x in up),
                          'negative':sum(x < -1e-12 for x in up),'net':float(np.mean(delta)),
                          'nonupgrade_quality_unchanged':all(abs(x)<=1e-12 for x in nonup)}
    positive=all(v['upgrades']==5 and v['positive']==2 and v['tie']==3 and v['negative']==0 and v['net']>0 and v['nonupgrade_quality_unchanged'] for v in upgrades.values())
    branches=(rr['cagr_v2']['tiers']==[101,94,5] and rr['cagr_v2_full']['tiers']==[100,95,5])
    report['gates']={'full_components':complete,'independent_on_off':independent,'independent_is_hard_gate':part=='calibration',
                     'RAG_quality':quality,'comparison_quality':bool(comparison),'latency_mean':latency,
                     'exact_two_positive_three_tie_zero_harm':positive,'frozen_branch_distribution':branches,
                     'technical_pass':complete,
                     'method_frozen':part=='calibration' and complete and independent and quality and comparison and latency and positive and branches}
    report['upgrade_effect']=upgrades;return report


def run_partition(part,g,retriever,config,study,online,fine,router,baselines,oracle,deadline,backend=dispatch):
    qs=samples(part);out=ROOT/'results/fine_validation'/part;out.mkdir(parents=True,exist_ok=False);normal=[];pairs=[];budget=online['common_budget'];repeats=4 if part=='smoke' else 12
    def execute(s,method,scope,ordinal,enabled=True,plan=None):
        if time.monotonic()>deadline:raise RuntimeError('Registered wall guard; no retry')
        g.calls=[];g.signal_collection_enabled=enabled;g.torch.manual_seed(42);g.torch.cuda.manual_seed_all(42);start=time.perf_counter();row=None
        eid=part+':'+s['sample_id']+':'+scope+':'+method+':'+str(ordinal)
        try:
            row=backend(method,s,g,retriever,config,study,budget,fine,router,baselines,part,plan)
            row.update(execution_id=eid,sample_id=s['sample_id'],dataset=s['dataset'],method=method,scope=scope,features_enabled=enabled,attempt_wall_ms=1000*(time.perf_counter()-start))
            row['outer_framework_ms']=row['attempt_wall_ms']-row['cost']['end_to_end_ms']
            append(out/'executions.jsonl',row);validate_execution(row,budget)
            if scope=='normal' and method in oracle[part][s['sample_id']]:verify_expected(row,oracle[part][s['sample_id']][method])
            return row
        except BaseException as exc:
            if row is None:append(out/'failed_attempts.jsonl',{'execution_id':eid,'generation_calls':g.calls,'attempt_wall_ms':1000*(time.perf_counter()-start),'error':type(exc).__name__,'unreturned_backend_cost':'unknown_not_zero'})
            raise
        finally:g.signal_collection_enabled=True;g.drop_retained();g.retain_rag1=False
    for i,s in enumerate(qs):
        # Charge retrieval prewarming separately, before all methods for this question.
        _,warm=retriever.search(s['question']);append(ROOT/'logs/shared_retrieval_warmup.jsonl',{'sample_id':s['sample_id'],'partition':part,'scope':'shared_preparation_not_policy',**warm})
        methods=list(METHODS);random.Random(42).shuffle(methods);shift=i%len(methods);methods=methods[shift:]+methods[:shift];by_method={}
        for method in methods:
            r=execute(s,method,'normal',0);normal.append(r);by_method[method]=r
        for method in ['cagr_v2_full']:
            reference=by_method[method];plan=selected_plan(reference);ratios=[];ids=[]
            for rep in range(repeats):
                order=['on','off'] if (i+rep)%2==0 else ['off','on'];group={}
                for mode in order:
                    group[mode]=execute(s,method,'pair_'+mode,rep,mode=='on',plan if mode=='off' else None);ids.append(group[mode]['execution_id'])
                assert science(group['on'])==science(group['off'])==science(reference),'on/off changes science'
                assert selected_plan(group['on'])==selected_plan(group['off'])==plan,'on/off changes decisions'
                if method in oracle[part][s['sample_id']]:verify_expected(group['on'],oracle[part][s['sample_id']][method])
                ratios.append((group['on']['cost']['end_to_end_ms']-group['off']['cost']['end_to_end_ms'])/group['off']['cost']['end_to_end_ms'])
            p={'sample_id':s['sample_id'],'dataset':s['dataset'],'method':method,'repeats':repeats,'ratios':ratios,'question_mean_ratio':float(np.mean(ratios)),'execution_ids':ids};pairs.append(p);append(out/'pairs.jsonl',p)
        if part=='smoke':
            for tier in [0,1,2]:
                forced={'first':tier>0,'action':'rewrite' if tier==2 else 'stop'};execute(s,'cagr_v2_full','forced_tier_'+str(tier),0,False,forced)
        print(json.dumps({'partition':part,'completed':i+1,'total':len(qs)}),flush=True)
        if (i+1)%20==0:print('Checkpoint:',archive(ROOT,part+'-'+str(i+1)),flush=True)
    # Reference labels enter only after all normal predictions in this partition exist.
    commit={'normal_rows':len(normal),'questions':len(qs),'executions_sha256':sha(out/'executions.jsonl'),'all_predictions_committed_before_reference':True};write(out/'prediction_commit.json',commit)
    refs={r['sample_id']:r['answers'] for r in [json.loads(l) for l in (ROOT/f'data/references/{part}.jsonl').read_text().splitlines()]}
    scored=[{**r,'quality':answer_metrics(r['answer'],refs[r['sample_id']],r['dataset'])} for r in normal]
    for r in scored:append(out/'scored.jsonl',r)
    report=report_partition(scored,pairs,part);write(out/'summary.json',report);return report


def run_all(g,retriever,config,study,online,fine,router,baselines,oracle,deadline,backend=dispatch):
    smoke=run_partition('smoke',g,retriever,config,study,online,fine,router,baselines,oracle,deadline,backend)
    if not smoke['gates']['technical_pass']:return {'runtime_status':'completed_controlled_smoke_stop','method_frozen':False,'calibration_started':False,'smoke_gates':smoke['gates']}
    cal=run_partition('calibration',g,retriever,config,study,online,fine,router,baselines,oracle,deadline,backend)
    return {'runtime_status':'validation_completed','method_frozen':cal['gates']['method_frozen'],'calibration_started':True,'calibration_completed':True,'gates':cal['gates']}


def main():
    started=time.perf_counter();deadline=time.monotonic()+42000;status={'runtime_status':'running','method_frozen':False};retriever=None;session=None
    try:
        if not ROOT.resolve().is_relative_to(Path('/kaggle/working').resolve()):raise RuntimeError('Frozen Kaggle runtime only')
        config=yaml.safe_load((ROOT/'configs/experiment.yaml').read_text());online=yaml.safe_load((ROOT/'configs/online_v04.yaml').read_text());study=yaml.safe_load((ROOT/'configs/upgrade_v02.yaml').read_text());fingerprint()
        design=yaml.safe_load((ROOT/'configs/fine_gpu_v2.yaml').read_text())
        assert design['protocol']=='cagr-v2-fine-gpu-v2' and design['pairs_per_question']=={'smoke':4,'calibration':12}
        assert design['independent_method']=='cagr_v2_full' and design['RAG_latency_anchor_ms']==588.64 and design['latency_limit_ms']==550
        assert sha(ROOT/'configs/cagr_v2_fine_candidate.json')==design['candidate_sha256'] and design['seed']==42
        session=RuntimePolicySession(ROOT);session.__enter__();validate_online_policy(config);check_versions(ROOT/'configs/requirements-runtime-v03.txt');import torch
        env={'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,'gpus':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],'seed':42,'device':'cuda:0'};write(ROOT/'logs/environment.json',env)
        rt=online['runtime'];assert len(env['gpus'])==2 and all('T4' in n for n in env['gpus']) and env['torch']==rt['torch_required'] and env['cuda']==rt['cuda_required'] and platform.python_version().rsplit('.',1)[0]==rt['python_major_minor']
        torch.cuda.set_device(0);torch.manual_seed(42);torch.cuda.manual_seed_all(42);ledger=[];model,_,working=prepare_mounted_assets(ROOT,checked_resolved_mounts(ROOT),ledger);config['model']['local_model_path']=str(model)
        record={'kind':'model_load_and_unchanged_Native_B_build','status':'running'};ledger.append(record);t=time.perf_counter()
        try:g=KVBridgeGenerator(config['model'],read(ROOT/'configs/model_asset_lock.json')['revision']);g.signal_variant='B';g.bridge_options=yaml.safe_load((ROOT/'configs/cagr_v2_route23.yaml').read_text())['bridge'];record['status']='ok'
        finally:record['ms']=1000*(time.perf_counter()-t);write(ROOT/'logs/preparation.json',ledger)
        retriever=Retriever(working,config['retrieval']);g.calls=[];g.generate(config['generation']['answer_system'],'Question: What is 1 plus 1?',8,'warmup',False);write(ROOT/'logs/warmup.json',{'generation_calls':g.calls})
        write(ROOT/'logs/native_numeric.json',numeric_gate(g))
        for s in samples('smoke'):append(ROOT/'logs/cache_preflight.jsonl',cache_check(s,g,config))
        status.update(run_all(g,retriever,config,study,online,read(ROOT/'configs/cagr_v2_fine_candidate.json'),read(ROOT/'configs/router_v04_pareto.json'),read(ROOT/'configs/baselines_v04_online.json'),read(ROOT/'data/fine_validation/expectations.json'),deadline))
        if status['method_frozen']:write(ROOT/'results/fine_validation/method_freeze.json',{'method_frozen':True,'candidate_id':'fine_382','candidate_sha256':sha(ROOT/'configs/cagr_v2_fine_candidate.json'),
            'package_manifest_sha256':sha(ROOT/'package_manifest.json'),'protocol':'docs/protocol_v2_fine_gpu_v2.md','gates':status['gates'],'heldout_v2_run':False,'test_threshold_tuning':False})
    except BaseException as exc:
        status.update(runtime_status='failed',error_type=type(exc).__name__,error=str(exc));write(ROOT/'logs/fatal.json',status);raise
    finally:
        if retriever is not None:retriever.connection.close()
        if session is not None and session.original is not None:
            session.__exit__(*sys.exc_info());write(ROOT/'logs/policy_session.json',session.record)
        status.update(total_wall_ms=1000*(time.perf_counter()-started),new_heldout_used=False,automatic_retries=0);write(ROOT/'logs/final_status.json',status);print('Final archive:',archive(ROOT,'fine-final'),flush=True)
if __name__=='__main__':main()
