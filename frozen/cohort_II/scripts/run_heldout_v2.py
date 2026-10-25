"""Pure prediction shard: frozen Full heads, eight methods and two fixed pairs."""
from pathlib import Path
import argparse,datetime,json,platform,sys,time,zipfile
import yaml
from v03_assets import read,write,sha
from heldout_v2_contract import (METHODS,append,rows,samples,method_order,pair_orders,deployment,validate_policy,
    fingerprint,validate_execution,commit_predictions,science,selected_plan)
from v04_runtime import run_method
from v2_fine_runtime import run_fine
from kv_bridge_inference import KVBridgeGenerator
from runtime_policy_session import RuntimePolicySession
from v04_v3_mounts import checked_resolved_mounts
from test1000_io_assets import prepare_mounted_assets
from v04_v3_native_state_checks import numeric_gate
from run_online_v04 import cache_check
from run_baselines import Retriever
from v03_dependencies import check_versions
from execution_policy import validate_online_policy
ROOT=Path(__file__).resolve().parents[1]


def archive(root=ROOT,label='snapshot',destination=Path('/kaggle/working')):
    root=Path(root);target=Path(destination)/f'cagr-heldout-v2-{root.name}-{label}-{time.time_ns()}-results.zip'
    with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED) as z:
        names=set()
        for folder in ['logs','results']:
            for p in (root/folder).rglob('*'):
                if p.is_file() and '__pycache__' not in p.parts:z.write(p,p.relative_to(root));names.add(p.relative_to(root).as_posix())
        for n in [*read(root/'package_manifest.json')['files'],'package_manifest.json']:
            if n not in names:z.write(root/n,n)
    return target


def dispatch(method,s,g,retriever,config,study,budget,d,plan=None):
    if method in ['cagr_v2','matched_binary_v2']:
        return run_fine(s['question'],s['dataset'],g,retriever,config,budget,d['fine'],study,fold=None,binary=method=='matched_binary_v2',plan=plan)
    name='cagr' if method=='cagr_v1' else method
    p=d['baselines'][method] if method in ['targ_style','sugar_l_style_budgeted'] else d['router']
    return run_method(name,s['question'],g,retriever,config,budget,p,study=study,fold=None)


def run_shard(root,shard,g,retriever,config,study,cfg,d,deadline,session_id,backend=dispatch):
    root=Path(root);qs=samples(root,shard);out=root/'results/heldout_v2';out.mkdir(parents=True,exist_ok=False);budget=cfg['common_budget']
    def guard():
        if time.monotonic()>deadline:raise RuntimeError('42000s operating guard; preserve partial costs; no retry')
    def execute(s,m,scope,repeat=None,mode=None,plan=None):
        guard();enabled=mode!='off';g.calls=[];g.signal_collection_enabled=enabled;g.torch.manual_seed(42);g.torch.cuda.manual_seed_all(42)
        eid=f'{shard}:{s["sample_id"]}:normal:{m}' if scope=='held_out_normal' else f'{shard}:{s["sample_id"]}:technical:{repeat}:{mode}'
        write(root/'logs/inflight_prediction.json',{'execution_id':eid,'status':'running','unreturned_cost':'unknown_not_zero'})
        start=time.perf_counter();row=None
        try:
            row=backend(m,s,g,retriever,config,study,budget,d,plan)
            row.update(execution_id=eid,sample_id=s['sample_id'],dataset=s['dataset'],method=m,scope=scope,shard=shard,
                global_index=s['global_index'],session_id=session_id,parameter_scope='full_200_development',fold=None,
                features_enabled=enabled,attempt=1,retry=False,attempt_wall_ms=1000*(time.perf_counter()-start))
            row['outer_framework_ms']=row['attempt_wall_ms']-row['cost']['end_to_end_ms']
            append(out/('predictions.jsonl' if scope=='held_out_normal' else 'technical_executions.jsonl'),row)
            validate_execution(row,budget);return row
        except BaseException as exc:
            if row is None:append(root/'logs/failed_executions.jsonl',dict(execution_id=eid,generation_calls=g.calls,attempt_wall_ms=1000*(time.perf_counter()-start),error_type=type(exc).__name__,unreturned_cost='unknown_not_zero'))
            raise
        finally:
            g.signal_collection_enabled=True;g.drop_retained();g.retain_rag1=False
            write(root/'logs/inflight_prediction.json',{'execution_id':eid,'status':row['status'] if row else 'incomplete'})
    for i,s in enumerate(qs):
        guard();start=time.perf_counter();warm=None
        try:_,warm=retriever.search(s['question'])
        finally:append(root/'logs/shared_retrieval_warmup.jsonl',dict(sample_id=s['sample_id'],scope='shared_preparation_not_policy',status='ok' if warm else 'failed',ms=1000*(time.perf_counter()-start),trace=warm))
        reference=None
        for m in method_order(s['global_index']):
            r=execute(s,m,'held_out_normal')
            if m=='cagr_v2':reference=r
        plan=selected_plan(reference)
        for repeat,order in enumerate(pair_orders(s['global_index'],2)):
            group={}
            for mode in order:group[mode]=execute(s,'cagr_v2','held_out_technical_not_policy',repeat,mode,plan if mode=='off' else None)
            if science(group['on'])!=science(group['off']) or science(group['on'])!=science(reference) or selected_plan(group['on'])!=selected_plan(group['off']) or selected_plan(group['on'])!=plan:raise RuntimeError('Frozen on/off scientific/decision equivalence failed')
            denominator=group['off']['cost']['end_to_end_ms']
            if denominator<=0:raise RuntimeError('Nonpositive independent denominator')
            append(out/'paired_observations.jsonl',dict(sample_id=s['sample_id'],dataset=s['dataset'],shard=shard,repeat=repeat,order=list(order),
                execution_ids={k:v['execution_id'] for k,v in group.items()},ratio=(group['on']['cost']['end_to_end_ms']-denominator)/denominator))
        print(json.dumps(dict(shard=shard,questions_completed=i+1,total=500)),flush=True)
        if (i+1)%100==0:print('Checkpoint:',archive(root,'checkpoint-'+str(i+1)),flush=True)
    return commit_predictions(root,shard,budget)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--shard',type=int,choices=[1,2],required=True);args=ap.parse_args()
    started=time.perf_counter();begin=time.monotonic();status=dict(runtime_status='running',shard=args.shard,reference_values_opened=False,automatic_retries=0);session=None;retriever=None
    try:
        if not ROOT.resolve().is_relative_to(Path('/kaggle/working').resolve()):raise RuntimeError('Prediction entry only in frozen Kaggle working directory')
        if any((ROOT/n).exists() for n in ['data/references','scoring_only','_scoring']):raise RuntimeError('Reference assets forbidden in inference root')
        session=RuntimePolicySession(ROOT);session.__enter__();validate_policy(ROOT,args.shard);fp=fingerprint(ROOT);cfg,d=deployment(ROOT)
        config=yaml.safe_load((ROOT/'configs/experiment.yaml').read_text());study=yaml.safe_load((ROOT/'configs/upgrade_v02.yaml').read_text())
        validate_online_policy(config);check_versions(ROOT/'configs/requirements-runtime-v03.txt');import torch
        env=dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            seed=42,device='cuda:0',batch=1,concurrency=1,shard=args.shard,fingerprint=fp,head_scope='full_200_development',gpu_uuid=str(getattr(torch.cuda.get_device_properties(0),'uuid','unavailable')))
        write(ROOT/'logs/environment.json',env);rt=cfg['runtime']
        if len(env['gpus'])!=2 or not all('T4' in n for n in env['gpus']) or env['torch']!=rt['torch_required'] or env['cuda']!=rt['cuda_required'] or platform.python_version().rsplit('.',1)[0]!=rt['python_major_minor']:raise RuntimeError('Frozen T4 environment differs')
        torch.cuda.set_device(0);torch.manual_seed(42);torch.cuda.manual_seed_all(42);ledger=[]
        model,_,working=prepare_mounted_assets(ROOT,checked_resolved_mounts(ROOT),ledger);config['model']['local_model_path']=str(model)
        t=time.perf_counter();record={'kind':'model_load_and_unchanged_Native_B_build','status':'running'};ledger.append(record)
        try:
            g=KVBridgeGenerator(config['model'],read(ROOT/'configs/model_asset_lock.json')['revision']);g.signal_variant='B'
            g.bridge_options=yaml.safe_load((ROOT/'configs/cagr_v2_route23.yaml').read_text())['bridge'];record['status']='ok'
        finally:record['ms']=1000*(time.perf_counter()-t);write(ROOT/'logs/preparation.json',ledger)
        g.calls=[];g.generate(config['generation']['answer_system'],'Question: What is 1 plus 1?',8,'warmup',False);write(ROOT/'logs/warmup.json',{'generation_calls':g.calls})
        write(ROOT/'logs/native_numeric.json',numeric_gate(g))
        for s in rows(ROOT/'data/preflight/smoke.jsonl'):append(ROOT/'logs/cache_preflight.jsonl',cache_check(s,g,config))
        retriever=Retriever(working,config['retrieval'])
        receipt=run_shard(ROOT,args.shard,g,retriever,config,study,cfg,d,begin+42000,fp)
        status.update(runtime_status='all_predictions_committed',normal_rows=receipt['normal_rows'],technical_executions=receipt['technical_executions'],paired_observations=receipt['paired_observations'],prediction_commit_sha256=sha(ROOT/'results/heldout_v2/prediction_commit.json'))
    except BaseException as exc:
        status.update(runtime_status='failed_or_incomplete',error_type=type(exc).__name__,error=str(exc));write(ROOT/'logs/fatal.json',status);raise
    finally:
        if retriever is not None:retriever.connection.close()
        if session is not None and session.original is not None:session.__exit__(*sys.exc_info());write(ROOT/'logs/policy_session.json',session.record)
        status['program_wall_ms']=1000*(time.perf_counter()-started);write(ROOT/'logs/final_status.json',status)
        print('Final prediction archive:',archive(ROOT,'prediction-final'),flush=True)


if __name__=='__main__':main()
