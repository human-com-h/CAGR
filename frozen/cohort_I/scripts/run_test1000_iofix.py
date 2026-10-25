"""Prediction-only held-out shard. No reference loader or scoring entry here."""
from pathlib import Path
import argparse,datetime,importlib.metadata as md,json,platform,time,zipfile
import yaml
from test1000_contract import (METHODS,append,read,write,sha,jsonl,validate_policy,samples,fingerprint,parameters,method_order,pair_order,
    validate_execution,scientific_sha256,path_view,commit_predictions)
from v04_runtime import run_method
from v04_v3_native_state_inference import NativeStateGenerator
from v04_v3_native_state_checks import numeric_gate
from v04_v3_mounts import checked_resolved_mounts
from test1000_io_assets import prepare_mounted_assets
from v03_dependencies import check_versions
from run_baselines import Retriever
from run_online_v04 import cache_check
ROOT=Path(__file__).resolve().parents[1]
KAGGLE_WORKING=Path('/kaggle/working')
class WallGuard(RuntimeError):pass
def archive(root=ROOT,label='prediction-snapshot'):
    root=Path(root);target=Path('/kaggle/working')/f'cagr-test1000-{root.name}-{label}-{time.time_ns()}-results.zip'
    with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED) as z:
        names=set()
        for directory in ['results','logs']:
            for p in (root/directory).rglob('*'):
                if p.is_file() and '__pycache__' not in p.parts:z.write(p,p.relative_to(root));names.add(p.relative_to(root).as_posix())
        for name in [*read(root/'package_manifest.json')['files'],'package_manifest.json']:
            if name not in names:z.write(root/name,name)
    return target
def run_shard(root,shard,g,retriever,config,study,online,deployment,deadline,session_id,execute=run_method):
    root=Path(root);questions=samples(root,shard);out=root/'results/test1000'
    if out.exists():raise RuntimeError('Never overwrite/resume a test shard automatically')
    out.mkdir(parents=True);budget=online['common_budget']
    def guard():
        if time.monotonic()>deadline:raise WallGuard('Registered operating guard; partial results retained, no quality-dependent continuation')
    def invoke(sample,method,eid,scope,enabled=True,plan=None):
        guard();g.calls=[];g.signal_collection_enabled=enabled;before=time.perf_counter();row=None
        write(root/'logs/inflight_test.json',{'execution_id':eid,'status':'running','unreturned_cost':'unknown_not_zero'})
        try:
            g.torch.manual_seed(42);g.torch.cuda.manual_seed_all(42)
            row=execute(method,sample['question'],g,retriever,config,budget,parameters(method,deployment),study=study,fold=None,technical_plan=plan if not enabled else None)
            row.update(execution_id=eid,sample_id=sample['sample_id'],dataset=sample['dataset'],shard=shard,global_index=sample['global_index'],session_id=session_id,
                scope=scope,parameter_scope='all_200_development',fold=None,attempt=1,retry=False,
                attempt_wall_ms=1000*(time.perf_counter()-before),saved_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            append(out/('predictions.jsonl' if scope=='held_out_normal' else 'technical_executions.jsonl'),row)
            validate_execution(row,budget);return row
        except BaseException as exc:
            if row is None:append(root/'logs/failed_test_executions.jsonl',{'execution_id':eid,'scope':scope,'generation_calls':g.calls,
                'attempt_wall_ms':1000*(time.perf_counter()-before),'unreturned_cost':'unknown_not_zero','error_type':type(exc).__name__})
            raise
        finally:
            g.signal_collection_enabled=True;write(root/'logs/inflight_test.json',{'execution_id':eid,'status':row['status'] if row else 'incomplete'})
    for i,s in enumerate(questions):
        guard();before=time.perf_counter();warm=None
        try:_,warm=retriever.search(s['question'])
        finally:append(root/'logs/test_shared_retrieval_warmup.jsonl',{'sample_id':s['sample_id'],'scope':'shared_preparation_not_method',
            'status':warm.get('status','ok') if warm else 'failed','ms':1000*(time.perf_counter()-before),'trace':warm})
        main=None
        for method in method_order(s['global_index']):
            r=invoke(s,method,f'{shard}:{s["sample_id"]}:normal:{method}','held_out_normal')
            if method=='cagr':main=r
        plan=[False,False]
        for d in main['control_trace']:
            if d['kind']=='decision_value':plan[d['stage']]=bool(d['selected'])
        for repeat in range(2):
            order=pair_order(s['global_index'],repeat);group={}
            for mode in order:group[mode]=invoke(s,'cagr',f'{shard}:{s["sample_id"]}:technical:{repeat}:{mode}','held_out_technical_not_policy',mode=='on',plan)
            if scientific_sha256(group['on'])!=scientific_sha256(main) or path_view(group['on'])!=path_view(group['off']):raise RuntimeError('Test on/off changed frozen deployment route/Token/retrieval')
            off=group['off']['cost']['end_to_end_ms']
            if off<=0:raise RuntimeError('Invalid independent timing denominator')
            append(out/'paired_observations.jsonl',{'sample_id':s['sample_id'],'dataset':s['dataset'],'shard':shard,'repeat':repeat,'order':order,
                'execution_ids':{m:r['execution_id'] for m,r in group.items()},'ratio':(group['on']['cost']['end_to_end_ms']-off)/off})
        print(json.dumps({'shard':shard,'questions_completed':i+1,'total':500}),flush=True)
        if (i+1)%100==0:print('Saved checkpoint:',archive(root,'checkpoint-'+str(i+1)),flush=True)
    return commit_predictions(root,shard,questions,budget)
def main():
    start=time.perf_counter();begin=time.monotonic();p=argparse.ArgumentParser();p.add_argument('--shard',type=int,choices=[1,2],required=True);a=p.parse_args()
    result={'runtime_status':'running','test_shard':a.shard,'scoring_started':False};retriever=None
    try:
        if not ROOT.resolve().is_relative_to(KAGGLE_WORKING.resolve()):raise RuntimeError('GPU test entry only in frozen Kaggle working directory')
        validate_policy(ROOT,a.shard);fp=fingerprint(ROOT)
        config=yaml.safe_load((ROOT/'configs/experiment.yaml').read_text());online=yaml.safe_load((ROOT/'configs/test1000.yaml').read_text());study=yaml.safe_load((ROOT/'configs/upgrade_v02.yaml').read_text());deploy=read(ROOT/'configs/deployment_test1000.json')
        check_versions(ROOT/'configs/requirements-runtime-v03.txt');import torch
        env={'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,'gpus':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            'device':'cuda:0','batch':1,'concurrency':1,'fingerprint':fp,'shard':a.shard,'signal_variant':'B','head_scope':'all_200_development',
            'gpu_uuid':str(getattr(torch.cuda.get_device_properties(0),'uuid','unavailable')),
            'dependencies':{n:md.version(n) for n in ['transformers','accelerate','huggingface_hub','pyarrow','pyyaml','numpy','tokenizers']}}
        write(ROOT/'logs/test_environment.json',env);rt=online['runtime']
        if len(env['gpus'])!=2 or any('T4' not in x for x in env['gpus']) or env['torch']!=rt['torch_required'] or env['cuda']!=rt['cuda_required'] or platform.python_version().rsplit('.',1)[0]!=rt['python_major_minor']:raise RuntimeError('Frozen T4 runtime differs')
        torch.cuda.set_device(0);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        ledger=[];model,_,working=prepare_mounted_assets(ROOT,checked_resolved_mounts(ROOT),ledger);config['model']['local_model_path']=str(model)
        before=time.perf_counter();record={'kind':'model_load_and_native_build','status':'running'};ledger.append(record)
        try:g=NativeStateGenerator(config['model'],read(ROOT/'configs/model_asset_lock.json')['revision']);g.signal_variant='B';record['status']='ok'
        except BaseException as exc:record.update(status='failed',error_type=type(exc).__name__);raise
        finally:record['ms']=1000*(time.perf_counter()-before);write(ROOT/'logs/preparation.json',ledger)
        before=time.perf_counter();g.calls=[];g.generate(config['generation']['answer_system'],'Question: What is 1 plus 1?',8,'warmup',collect_signals=False)
        write(ROOT/'logs/test_model_warmup.json',{'generation_calls':g.calls,'ms':1000*(time.perf_counter()-before),'scope':'paid_preparation'})
        write(ROOT/'logs/test_native_numeric.json',numeric_gate(g))
        for sample in jsonl(ROOT/'data/preflight/smoke.jsonl'):
            append(ROOT/'logs/test_cache_preflight.jsonl',cache_check(sample,g,config))
        retriever=Retriever(working,config['retrieval']);receipt=run_shard(ROOT,a.shard,g,retriever,config,study,online,deploy,begin+online['operating_guard_seconds'],fp)
        result.update(runtime_status='all_predictions_committed',prediction_commit_sha256=sha(ROOT/'results/test1000/prediction_commit.json'),normal_rows=receipt['normal_rows'],reference_semantics_opened=False)
    except WallGuard as exc:result.update(runtime_status='incomplete_wall_guard',error=str(exc));raise
    except BaseException as exc:result.update(runtime_status='failed',error_type=type(exc).__name__);write(ROOT/'logs/test_fatal.json',{'error_type':type(exc).__name__,'error':str(exc)});raise
    finally:
        if retriever is not None:retriever.connection.close()
        result['ms']=1000*(time.perf_counter()-start);write(ROOT/'logs/test_prediction_status.json',result)
    return result
if __name__=='__main__':main()
