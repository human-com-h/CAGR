"""One offline T4 development collection: smoke20 -> Cal200 complete action matrix."""
from pathlib import Path
import importlib.metadata as md,json,platform,time,zipfile
import numpy as np,yaml
from v03_assets import read,write,sha
from run_online_v04 import append,fingerprint,samples,parameters,cache_check
from v04_runtime import SharedRuntime,run_method
from run_baselines import Retriever,answer_metrics
from kv_bridge_inference import KVBridgeGenerator
from route23_runtime import collect_bridge,collect_in_situ_increment,collect_rewrite_control,validate_row,run_candidate,path_identity
from v04_v3_native_state_checks import numeric_gate
from v04_v3_mounts import checked_resolved_mounts
from test1000_io_assets import prepare_mounted_assets
from v03_dependencies import check_versions
from execution_policy import validate_online_policy
from fit_route23 import fit_candidates
from analyze_route23_regimes import analyze_regimes
ROOT=Path(__file__).resolve().parents[1]

def archive(root=ROOT,label='snapshot'):
    target=Path('/kaggle/working')/f'cagr-route23-{label}-{time.time_ns()}-results.zip'
    with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED) as z:
        names=set()
        for folder in ['results','logs']:
            for p in (root/folder).rglob('*'):
                if p.is_file() and '__pycache__' not in p.parts:z.write(p,p.relative_to(root));names.add(p.relative_to(root).as_posix())
        for name in [*read(root/'package_manifest.json')['files'],'package_manifest.json']:
            if name not in names:z.write(root/name,name)
    return target

def require_signature(row,expected):
    if row['context_ids']!=expected['context_ids']:raise RuntimeError('Frozen corpus retrieval/context identity changed')
    got=[{k:c[k] for k in ['kind','prompt_sha256','input_tokens','output_token_ids']} for c in row['generation_calls']]
    if got!=expected['calls']:raise RuntimeError('Frozen prompt/greedy Token changed; cannot reuse old potential answers')

def matrix_row(sample,bridge,control,increment,known,baseline_threshold):
    calls=bridge['generation_calls'];pilot=next(c for c in calls if c['kind']=='pilot');rag=next(c for c in calls if c['kind']=='rag1');kv=next(c for c in calls if c['kind']=='kv_bridge');final=next((c for c in calls if c['kind']=='multihop_final'),None)
    ret1,ret2=[r['ms'] for r in bridge['retrieval_trace']];base=bridge['collection_H1_phase_ms']
    multi=control['always_multihop'];mc=multi['generation_calls'];rewrite=next(c for c in mc if c['kind']=='query_rewrite');mf=next((c for c in mc if c['kind']=='multihop_final'),None);mr=multi['retrieval_trace'][1]['ms']
    direct=control['direct'];actions={
        'direct':{'quality':direct['quality'],'latency_proxy_ms':direct['cost']['end_to_end_ms']+.5,'retrieval_ms':0.,'retrievals':0,'current_generation_measured':True},
        'stop':{'quality':control['always_rag']['quality'],'latency_proxy_ms':base+.5,'retrieval_ms':ret1,'retrievals':1,'current_generation_measured':True},
        'kv_bridge':{'quality':bridge['quality'],'latency_proxy_ms':bridge['cost']['end_to_end_ms']+.5,'retrieval_ms':ret1+ret2,'retrievals':2,'current_generation_measured':True},
        'rewrite':{'quality':multi['quality'],'latency_proxy_ms':base+multi['cost']['end_to_end_ms']-multi['collection_H1_phase_ms']+.5,'retrieval_ms':ret1+mr,'retrievals':2,'current_generation_measured':True}}
    if increment:
        value=increment['quality'];inc=increment['cost']['end_to_end_ms'];rms=increment['cost']['retrieval_ms']
    else:value=known['known_quality'];inc=known['known_increment_ms'];rms=known['known_second_retrieval_ms']
    actions['in_situ']={'quality':value,'latency_proxy_ms':base+inc+.5,'retrieval_ms':ret1+rms,'retrievals':2,'current_generation_measured':bool(increment),
        'provenance':'current missing-H2 increment' if increment else 'verified historical same-prompt final answer; old component timing proxy'}
    targ_action='stop' if bridge['stage0_features']['targ_mean_exp_gap']>baseline_threshold else 'direct'
    return {'sample_id':sample['sample_id'],'dataset':sample['dataset'],'fold':sample['fold'],'stage0_features':bridge['stage0_features'],'stage1_features':bridge['stage1_features'],
        'actions':actions,'TARG_action':targ_action,'TARG_actual_ms':control['targ_style']['cost']['end_to_end_ms'],'collection_base_ms':base,
        'TARG_retrieval_ms':control['targ_style']['cost']['retrieval_ms'],
        'Always_RAG_actual_ms':control['always_rag']['cost']['end_to_end_ms'],'Always_RAG_retrieval_ms':control['always_rag']['cost']['retrieval_ms'],
        'TARG_arm_proxy_ms':{'direct':direct['cost']['end_to_end_ms']+.5,'stop':control['always_rag']['cost']['end_to_end_ms']+pilot['pilot_prefix_ms']+.5},
        'latency_scope':'paid component recombination, no new policy E2E; H1 shared collection costs not duplicated in physical ledger'}

def collect_partition(partition,g,retriever,config,study,online,cfg,deadline):
    questions=samples(partition);ref={r['sample_id']:r for r in read(ROOT/'data/route23/archived_signatures.json')};coverage={r['sample_id']:r for r in read(ROOT/'data/route23/in_situ_coverage.json')};gold={r['sample_id']:r for r in [json.loads(s) for s in (ROOT/f'data/references/{partition}.jsonl').read_text().splitlines()]}
    router=read(ROOT/'configs/router_v04_pareto.json');baselines=read(ROOT/'configs/baselines_v04_online.json');out=ROOT/'results/route23'/partition;out.mkdir(parents=True,exist_ok=False);matrix=[]
    def execute(s,name,fn):
        if time.monotonic()>deadline:raise RuntimeError('Registered wall guard reached; partial costs retained')
        g.calls=[];g.torch.manual_seed(42);g.torch.cuda.manual_seed_all(42);start=time.perf_counter();value=None
        try:
            value=fn();row=value[0] if isinstance(value,tuple) else value
            row.update(sample_id=s['sample_id'],dataset=s['dataset'],partition=partition,collection_arm=name,execution_id=partition+':'+s['sample_id']+':'+name,
                attempt_wall_ms=1000*(time.perf_counter()-start),quality=answer_metrics(row['answer'],gold[s['sample_id']]['answers'],s['dataset']))
            append(out/'executions.jsonl',row);validate_row(row,online['common_budget']);return value
        except BaseException as exc:
            if value is None:append(out/'failed_attempts.jsonl',{'sample_id':s['sample_id'],'arm':name,'generation_calls':g.calls,'attempt_wall_ms':1000*(time.perf_counter()-start),'error_type':type(exc).__name__,'unreturned_cost':'unknown_not_zero'})
            raise
        finally:g.drop_retained();g.retain_rag1=False
    for i,s in enumerate(questions):
        fold=s.get('fold') if partition=='calibration' else None;expected=ref[s['sample_id']];control={}
        arms=['direct','always_rag','always_multihop','targ_style','kv_bridge'];shift=i%len(arms);arms=arms[shift:]+arms[:shift]
        for method in arms:
            if method=='kv_bridge':bridge,initial=execute(s,method,lambda:collect_bridge(s['question'],g,retriever,config,online['common_budget'],study,partition=='smoke'))
            elif method=='always_multihop':control[method]=execute(s,method,lambda:collect_rewrite_control(s['question'],g,retriever,config,online['common_budget'],study))
            else:control[method]=execute(s,method,lambda m=method:run_method(m,s['question'],g,retriever,config,online['common_budget'],parameters(m,router,baselines,fold),study=study,fold=fold))
            if method in expected['arms']:require_signature(control[method],expected['arms'][method])
        parent=next(c for c in bridge['generation_calls'] if c['kind']=='rag1');rag=control['always_rag']['generation_calls'][0]
        if parent['output_token_ids']!=rag['output_token_ids'] or parent['prompt_sha256']!=rag['prompt_sha256'] or bridge['initial_context_ids']!=control['always_rag']['context_ids']:raise RuntimeError('Retaining H1 changed original science')
        gaps=expected['original_native_H1_gaps']
        if gaps and parent['token_logit_gaps']!=gaps:raise RuntimeError('Frozen Native H1 feature values changed')
        if partition=='smoke':
            def reference():
                rt=SharedRuntime(g,retriever,config,online['common_budget'],study=study);g.raw_bridge_reference(initial['reference_prefix'],cfg['bridge']['max_new_tokens']);return rt.finish('bridge_full_prefix_reference',g.calls[-1]['decoded_text'])
            full=execute(s,'bridge_full_prefix_reference',reference)
            if full['generation_calls'][0]['output_token_ids']!=initial['bridge_tokens']:raise RuntimeError('Append-only bridge differs from identical full Token prefix reference')
            increment=execute(s,'in_situ_increment',lambda:collect_in_situ_increment(s['question'],initial,g,retriever,config,online['common_budget'],study));known=None
        else:
            known=coverage[s['sample_id']];increment=None
            if known['needs_new_H2']:increment=execute(s,'in_situ_increment',lambda:collect_in_situ_increment(s['question'],initial,g,retriever,config,online['common_budget'],study))
            if increment and (increment['retrieval_trace'][0]['query']!=known['expected_query'] or increment['context_ids']!=known['expected_merged_doc_ids']):raise RuntimeError('Missing-H2 query/context changed from registered target')
        threshold=parameters('targ_style',router,baselines,fold)['threshold']
        if partition=='calibration':
            item=matrix_row(s,bridge,control,increment,known,threshold);append(out/'matrix.jsonl',item);matrix.append(item)
        print(json.dumps({'stage':partition,'completed':i+1,'total':len(questions),'missing_H2_new':bool(increment)}),flush=True)
        if (i+1)%20==0:print('Checkpoint:',archive(ROOT,partition+'-'+str(i+1)),flush=True)
    write(out/'commit.json',{'partition':partition,'questions':len(questions),'executions_sha256':sha(out/'executions.jsonl'),'all_validation_passed':True,'new_validation_used':False})
    return matrix

def main():
    start=time.perf_counter();deadline=time.monotonic()+42000;status={'runtime_status':'running','method_frozen':False,'fresh_validation_used':False};retriever=None
    try:
        if not ROOT.resolve().is_relative_to(Path('/kaggle/working').resolve()):raise RuntimeError('Kaggle frozen-package entry only')
        config=yaml.safe_load((ROOT/'configs/experiment.yaml').read_text());study=yaml.safe_load((ROOT/'configs/upgrade_v02.yaml').read_text());online=yaml.safe_load((ROOT/'configs/online_v04.yaml').read_text());cfg=yaml.safe_load((ROOT/'configs/cagr_v2_route23.yaml').read_text())
        validate_online_policy(config);fingerprint();check_versions(ROOT/'configs/requirements-runtime-v03.txt');import torch
        env={'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,'gpus':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            'device':'cuda:0','seed':42,'model_revision':read(ROOT/'configs/model_asset_lock.json')['revision'],'native_variant':'B'};write(ROOT/'logs/route23_environment.json',env)
        runtime=online['runtime']
        if len(env['gpus'])!=2 or any('T4' not in n for n in env['gpus']) or env['torch']!=runtime['torch_required'] or env['cuda']!=runtime['cuda_required'] or platform.python_version().rsplit('.',1)[0]!=runtime['python_major_minor']:raise RuntimeError('Frozen same-T4 environment differs')
        torch.cuda.set_device(0);torch.manual_seed(42);torch.cuda.manual_seed_all(42);ledger=[];model,_,working=prepare_mounted_assets(ROOT,checked_resolved_mounts(ROOT),ledger);config['model']['local_model_path']=str(model)
        record={'kind':'model_load_and_Native_B_build','status':'running'};ledger.append(record);before=time.perf_counter()
        try:g=KVBridgeGenerator(config['model'],env['model_revision']);g.signal_variant='B';g.bridge_options=cfg['bridge'];record['status']='ok'
        finally:record['ms']=1000*(time.perf_counter()-before);write(ROOT/'logs/preparation.json',ledger)
        retriever=Retriever(working,config['retrieval']);g.calls=[];before=time.perf_counter();g.generate(config['generation']['answer_system'],'Question: What is 1 plus 1?',8,'warmup',False);write(ROOT/'logs/route23_warmup.json',{'generation_calls':g.calls,'ms':1000*(time.perf_counter()-before)})
        write(ROOT/'logs/route23_numeric.json',numeric_gate(g))
        for s in samples('smoke'):append(ROOT/'logs/route23_cache_preflight.jsonl',cache_check(s,g,config))
        collect_partition('smoke',g,retriever,config,study,online,cfg,deadline)
        matrix=collect_partition('calibration',g,retriever,config,study,online,cfg,deadline);assert len(matrix)==200
        if sum(r['actions']['in_situ']['current_generation_measured'] for r in matrix)!=69:raise RuntimeError('Missing69 coverage differs')
        before=time.perf_counter();candidate,selection=fit_candidates(matrix,cfg,read(ROOT/'configs/baselines_v04_online.json'),ROOT/'results/route23/selection');write(ROOT/'logs/route23_fit_cost.json',{'ms':1000*(time.perf_counter()-before),'new_model_calls':0})
        analyze_regimes(matrix,candidate,cfg,ROOT/'results/route23/selection')
        status.update(runtime_status='collection_and_development_selection_completed',complete_matrix_questions=200,real_bridge_paths=200,new_missing_H2=69,joint_proxy_candidates=selection['joint_proxy_candidates'],
            full_policy_GPU_validation_required=True,method_frozen=False)
    except BaseException as exc:
        status.update(runtime_status='failed',error_type=type(exc).__name__,error=str(exc));write(ROOT/'logs/route23_fatal.json',status);raise
    finally:
        if retriever is not None:retriever.connection.close()
        status['total_wall_ms']=1000*(time.perf_counter()-start);write(ROOT/'logs/final_status.json',status);write(ROOT/'logs/runtime_process.json',status)
        print('Final archive:',archive(ROOT,'final'),flush=True)
if __name__=='__main__':main()
