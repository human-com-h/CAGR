"""Offline Kaggle cal20->200 runner; no submit, network, downloads or test flag."""
from pathlib import Path
import argparse,datetime,hashlib,importlib.metadata as md,json,os,platform,random,sys,time,zipfile
import numpy as np,yaml
from execution_policy import validate_online_policy
from v03_assets import read,write,sha,prepare_mounted_assets
from v03_dependencies import check_versions
from run_baselines import Retriever,Generator,read_jsonl,answer_metrics
from v04_inference import LightweightGenerator
from v04_runtime import run_method
ROOT=Path(__file__).resolve().parents[1]

def append(path,row):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())

def fingerprint(root=ROOT):
    inv=read(root/'package_manifest.json')['files']
    for name,item in inv.items():
        if sha(root/name)!=item['sha256']:raise RuntimeError('Frozen payload changed: '+name)
    return hashlib.sha256(json.dumps(inv,sort_keys=True).encode()).hexdigest()

def samples(partition,root=ROOT):
    manifest=read(root/'data/splits/manifest.json')
    if set(manifest['partitions'])!={'smoke','calibration'}:raise RuntimeError('Only 20/200 runtime IDs are permitted')
    p=root/'data/splits'/(partition+'.jsonl');e=manifest['partitions'][partition]
    if sha(p)!=e['questions_sha256']:raise RuntimeError('Frozen question SHA mismatch')
    a=read_jsonl(p)
    if len(a)!=e['count'] or len({r['sample_id'] for r in a})!=len(a):raise RuntimeError('Frozen sample count differs')
    if any(set(r)-{'sample_id','dataset','source_split','question','selection_sha256','fold'} for r in a):raise RuntimeError('Unexpected labels/future values in model input')
    return a

def order(index,methods):
    a=list(methods);random.Random(42+index//len(a)).shuffle(a);shift=index%len(a);return a[shift:]+a[:shift]

def parameters(method,router,baselines,fold=None):
    if method not in {'targ_style','sugar_l_style_budgeted'}:return router
    p=dict(baselines[method])
    if fold is not None:p.update(p['calibration_fold_parameters'][str(fold)])
    return p

def validate_row(row,budget):
    if row['status'] not in {'ok','budget_limited_answer'}:raise RuntimeError('Saved method failure: '+str(row.get('error')))
    c=row['cost'];calls=row['generation_calls']
    if c['untracked_backend_cost']:raise RuntimeError('Backend costs not fully tracked')
    if c['total_input_tokens']!=sum(x['input_tokens'] for x in calls) or c['total_output_tokens']!=sum(x['output_tokens'] for x in calls):raise RuntimeError('Call ownership mismatch')
    if c['total_input_tokens']>budget['max_total_input_tokens'] or c['total_output_tokens']>budget['max_total_output_tokens'] or c['retrieval_calls']>budget['max_retrieval_calls']:raise RuntimeError('Common budget exceeded')
    for call in calls:
        if call['status']!='ok' or call['prefill_calls']!=1 or call['cache_final_sequence_length']!=call['input_tokens']+call['output_tokens']-1:raise RuntimeError('Single-prefill/cache audit failed')
        if any(e['input_tokens']!=1 for e in call['forward_trace'][1:]):raise RuntimeError('Repeated prompt prefill')
    return True

def require_smoke(root=ROOT):
    out=root/'results/online-v04/smoke';p=out/'acceptance.json'
    if not p.exists():raise RuntimeError('20 technical questions must pass before calibration')
    r=read(p)
    if r['status']!='passed' or r['questions']!=20 or r['fingerprint']!=fingerprint(root):raise RuntimeError('Smoke acceptance fingerprint differs')
    for name,h in r['evidence_sha256'].items():
        if sha(out/name)!=h:raise RuntimeError('Smoke evidence changed')
    return r

def cache_check(sample,g,config):
    g.calls=[];system=config['generation']['answer_system'];user='Question: '+sample['question'];limit=config['model']['max_answer_tokens']
    answer,_,reference=g.generate(system,user,limit,'uninterrupted_gap_reference')
    state,_=g.pilot(system,user,limit,'pause_continue_check');resumed,_,split=g.complete(state)
    if answer!=resumed or reference['output_token_ids']!=split['output_token_ids'] or split['prefill_calls']!=1:raise RuntimeError('K5 cache continuation token mismatch')
    # Frozen legacy greedy reference, with no full-vocabulary statistics.
    legacy,_,call=Generator.generate(g,system,user,limit,'frozen_greedy_reference',collect_signals=False)
    if legacy!=answer or call['output_token_ids']!=split['output_token_ids']:raise RuntimeError('Light signal adapter changed greedy tokens')
    return {'sample_id':sample['sample_id'],'kind':'cache_check','scope':'technical_not_policy','status':'ok','generation_calls':list(g.calls),
        'same_tokens':True,'resumed_beyond_k5':split['output_tokens']>5}

def paired_path(sample,g,retriever,config,study,budget,router,index):
    def save_subrun(row,label):
        row['technical_cost_id']=sample['sample_id']+'-'+label
        append(ROOT/'logs/technical_pair_subruns.jsonl',{'sample_id':sample['sample_id'],'label':label,'scope':'same_cost_id_as_nested_technical_record_do_not_sum_twice','row':row})
    g.calls=[];normal=run_method('cagr',sample['question'],g,retriever,config,budget,router,study=study);save_subrun(normal,'selection');validate_row(normal,budget)
    plan=[False,False]
    for t in normal['control_trace']:
        if t['kind']=='decision_value' and t['stage'] in [0,1]:plan[t['stage']]=t['selected']
    enabled_order=[False,True] if index%2 else [True,False];pairs={}
    for enabled in enabled_order:
        g.calls=[];g.signal_collection_enabled=enabled
        try:
            row=run_method('cagr',sample['question'],g,retriever,config,budget,router,study=study,
                technical_plan=None if enabled else plan)
        finally:g.signal_collection_enabled=True
        # Enabled side must include actual feature/decision work while using
        # the same already selected path; normal routing is deterministic.
        save_subrun(row,'features-'+str(enabled));validate_row(row,budget);pairs[str(enabled)]=row
    a,b=pairs['True'],pairs['False']
    if a['answer']!=b['answer'] or a['context_ids']!=b['context_ids'] or [x['output_token_ids'] for x in a['generation_calls']]!=[x['output_token_ids'] for x in b['generation_calls']]:raise RuntimeError('Paired feature collection changed outputs or path')
    return {'sample_id':sample['sample_id'],'dataset':sample['dataset'],'kind':'full_path_feature_pair','scope':'technical_not_policy','status':'ok','selected_plan':plan,
        'order':enabled_order,'pairs':pairs,'selection_reference':normal,'paired_e2e_increment_ms':a['cost']['end_to_end_ms']-b['cost']['end_to_end_ms'],
        'paired_ratio':(a['cost']['end_to_end_ms']-b['cost']['end_to_end_ms'])/b['cost']['end_to_end_ms']}

def bootstrap(a,replicates=20000):
    v=np.asarray(a,dtype=float);rng=np.random.default_rng(42);boot=v[rng.integers(0,len(v),(replicates,len(v)))].mean(1)
    return {'n':len(v),'mean':float(v.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist(),'upper_one_sided95':float(np.quantile(boot,.95))}

def analyze(scored,technical,methods):
    indexed={(r['sample_id'],r['method']):r for r in scored};datasets=sorted({r['dataset'] for r in scored});report={'scope':'same-environment 20/200 development validation','methods':{},'paired':{},'quality_mean_floor':{},'overhead':{},'GPU_measured':True,'test_executed':False}
    for ds in datasets:
        ids=sorted({r['sample_id'] for r in scored if r['dataset']==ds});report['methods'][ds]={};report['paired'][ds]={};report['quality_mean_floor'][ds]={}
        for method in methods:
            a=[indexed[sid,method] for sid in ids];c=[r['cost'] for r in a]
            report['methods'][ds][method]={k:float(np.mean([r['quality'][k] for r in a])) for k in ['em','f1']}
            report['methods'][ds][method].update({k:float(np.mean([r[k] for r in c])) for k in ['end_to_end_ms','total_input_tokens','total_output_tokens','retrieval_calls','routing_total_ms']})
            report['methods'][ds][method].update(p50_ms=float(np.median([r['end_to_end_ms'] for r in c])),p95_ms=float(np.quantile([r['end_to_end_ms'] for r in c],.95)),tier_counts=np.bincount([r['tier'] for r in a],minlength=3).tolist(),budget_limited=sum(r['status']=='budget_limited_answer' for r in a))
        for main in ['cagr','cagr_in_situ']:
            report['quality_mean_floor'][ds][main]={k:report['methods'][ds][main][k]>=report['methods'][ds]['always_rag'][k]-.01-1e-12 for k in ['em','f1']}
            for other in ['binary_gate','targ_style','always_rag','always_multihop','sugar_l_style_budgeted']:
                key=main+'_minus_'+other;report['paired'][ds][key]={}
                keys=['em','f1'] if other in {'binary_gate','targ_style'} else ['end_to_end_ms','retrieval_calls']
                for metric in keys:
                    values=[indexed[sid,main]['quality' if metric in {'em','f1'} else 'cost'][metric]-indexed[sid,other]['quality' if metric in {'em','f1'} else 'cost'][metric] for sid in ids]
                    report['paired'][ds][key][metric]=bootstrap(values)
                if other not in {'binary_gate','targ_style'}:
                    v=np.array([indexed[sid,main]['cost']['end_to_end_ms'] for sid in ids]);w=np.array([indexed[sid,other]['cost']['end_to_end_ms'] for sid in ids]);rng=np.random.default_rng(42);draw=rng.integers(0,len(ids),(20000,len(ids)));ratio=w[draw].mean(1)/v[draw].mean(1)
                    report['paired'][ds][key]['comparator_over_main_speedup']={'ratio_of_means':float(w.mean()/v.mean()),'paired_ci95':np.quantile(ratio,[.025,.975]).tolist()}
        report['overhead'][ds]={}
        for main in ['cagr','cagr_in_situ']:
            components=[indexed[sid,main]['cost']['routing_total_ms']/indexed[sid,main]['cost']['end_to_end_ms'] for sid in ids]
            report['overhead'][ds][main]={'full_components':bootstrap(components)}
        pairs=[r['paired_ratio'] for r in technical if r.get('kind')=='full_path_feature_pair' and r.get('dataset')==ds]
        if pairs:report['overhead'][ds]['paired_e2e']=bootstrap(pairs)
    normal_calls=[c for r in scored for c in r['generation_calls']];tech_calls=[];tech_retrievals=[]
    for r in technical:
        if r.get('kind')=='full_path_feature_pair':
            units=[r['selection_reference'],*r['pairs'].values()]
            tech_calls.extend(c for u in units for c in u['generation_calls']);tech_retrievals.extend(t for u in units for t in u['retrieval_trace'])
        else:tech_calls.extend(r.get('generation_calls',[]));tech_retrievals.extend(r.get('retrieval_trace',[]))
    def totals(calls,retrievals):
        return {'generation_calls':len(calls),'total_input_tokens':sum(c['input_tokens'] for c in calls),'total_output_tokens':sum(c['output_tokens'] for c in calls),
            'generation_ms':sum(c['ms'] for c in calls),'retrieval_calls':len(retrievals),'retrieval_ms':sum(t['ms'] for t in retrievals),'failed_generation_calls':sum(c['status']=='failed' for c in calls)}
    report['full_cost_totals']={'normal_methods':totals(normal_calls,[t for r in scored for t in r['retrieval_trace']]),'separate_technical_checks':totals(tech_calls,tech_retrievals),
        'preparation':'logs/preparation.json, generation_warmup.json, external_asset_preparation.json, per_question_shared_retrieval_warmup.jsonl; not zero or silently included twice',
        'technical_pair_subrun_log':'same technical_cost_id as nested rows; duplicate evidence storage, not additional executions'}
    return report

def comparison_markdown(report):
    text='# 同环境开发验证：真实输出\n\n|数据集|方法|EM %|F1 %|平均ms|p50 ms|p95 ms|检索/题|输入Token|输出Token|\n|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
    for ds,methods in report['methods'].items():
        for name,r in methods.items():text+=f"|{ds}|{name}|{r['em']*100:.2f}|{r['f1']*100:.2f}|{r['end_to_end_ms']:.2f}|{r['p50_ms']:.2f}|{r['p95_ms']:.2f}|{r['retrieval_calls']:.3f}|{r['total_input_tokens']:.1f}|{r['total_output_tokens']:.1f}|\n"
    return text+'\n质量均值约束、恢复/成本差区间和完整路由开销单侧上界见summary.json；本表不包含技术检查或准备成本，另有独立完整账目。\n'

def archive(root=ROOT,label='snapshot'):
    target=Path('/kaggle/working')/f'cagr-v04-{root.name}-{label}-{time.time_ns()}-results.zip'
    with zipfile.ZipFile(target,'x',zipfile.ZIP_DEFLATED) as z:
        for directory in ['results','logs','configs','scripts','data/splits','data/references']:
            for p in (root/directory).rglob('*'):
                if p.is_file() and '__pycache__' not in p.parts:z.write(p,p.relative_to(root))
        z.write(root/'package_manifest.json','package_manifest.json')
    return target

def run(partition,g,retriever,config,study,online,router,baselines,deadline):
    if partition=='calibration':require_smoke()
    a=samples(partition);out=ROOT/'results/online-v04'/partition
    if out.exists():raise RuntimeError('No automatic overwrite/resume/retry; preserve previous output')
    out.mkdir(parents=True);rows=[];technical=[];budget=online['common_budget'];methods=online['methods']
    def save_technical(fn):
        try:r=fn()
        except Exception as exc:
            r={'status':'failed','error':str(exc),'generation_calls':list(g.calls),'scope':'technical_not_policy'};append(out/'technical.jsonl',r);raise
        technical.append(r);append(out/'technical.jsonl',r)
    for i,sample in enumerate(a):
        t=time.perf_counter();warm={'status':'running'}
        try:_,warm=retriever.search(sample['question'])
        except Exception as exc:warm={'status':'failed','error':str(exc),'ms':1000*(time.perf_counter()-t)};raise
        finally:append(ROOT/'logs/per_question_shared_retrieval_warmup.jsonl',{'sample_id':sample['sample_id'],'scope':'shared_preparation_not_policy',**warm})
        for method in order(i,methods):
            if time.monotonic()>deadline:raise RuntimeError('Frozen 10800-second remote experiment budget exhausted')
            fold=sample.get('fold') if partition=='calibration' else None;p=parameters(method,router,baselines,fold);g.calls=[]
            write(ROOT/'logs/inflight_method.json',{'partition':partition,'sample_id':sample['sample_id'],'method':method,'status':'running','interruption_cost_if_no_completion':'unknown_not_zero'})
            attempt_start=time.perf_counter()
            # Reproducible sampling independent of method order and resume state.
            g.torch.manual_seed(42);g.torch.cuda.manual_seed_all(42)
            row=run_method(method,sample['question'],g,retriever,config,budget,p,study=study,fold=fold if method not in {'targ_style','sugar_l_style_budgeted'} else None)
            row.update(sample_id=sample['sample_id'],dataset=sample['dataset'],fold=sample.get('fold'),attempt=1,retry=False,attempt_wall_ms=1000*(time.perf_counter()-attempt_start),saved_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            append(out/'attempts.jsonl',row);rows.append(row);write(ROOT/'logs/inflight_method.json',{'partition':partition,'sample_id':sample['sample_id'],'method':method,'status':row['status']});validate_row(row,budget)
        if partition=='smoke':
            for tier in [0,1,2]:
                def forced(tier=tier):
                    g.calls=[];r=run_method('cagr',sample['question'],g,retriever,config,budget,router,study=study,technical_plan=[tier>0,tier>1]);r.update(sample_id=sample['sample_id'],kind='forced_tier_'+str(tier));validate_row(r,budget);return r
                save_technical(forced)
            save_technical(lambda:cache_check(sample,g,config))
        save_technical(lambda:paired_path(sample,g,retriever,config,study,budget,router,i))
        print(json.dumps({'partition':partition,'questions_completed':i+1,'total':len(a)}),flush=True)
        if partition=='calibration' and (i+1)%20==0:print('Saved checkpoint:',archive(ROOT,'calibration-checkpoint-'+str(i+1)),flush=True)
    ref=ROOT/'data/references'/(partition+'.jsonl')
    if sha(ref)!=read(ROOT/'data/splits/manifest.json')['partitions'][partition]['references_sha256']:raise RuntimeError('Reference SHA differs')
    references={r['sample_id']:r['answers'] for r in read_jsonl(ref)}
    scored=[{**r,'quality':answer_metrics(r['answer'],references[r['sample_id']],r['dataset'])} for r in rows]
    for r in scored:append(out/'scored.jsonl',r)
    report=analyze(scored,technical,methods);write(out/'summary.json',report);(out/'comparison.md').write_text(comparison_markdown(report),encoding='utf-8')
    evidence=['attempts.jsonl','scored.jsonl','summary.json','technical.jsonl']
    write(out/'acceptance.json',{'status':'passed','questions':len(a),'scope':'technical/development execution complete; scientific criteria reported separately','fingerprint':fingerprint(),
        'evidence_sha256':{p:sha(out/p) for p in evidence},'test_executed':False})

def main():
    process_started=time.monotonic()
    parser=argparse.ArgumentParser();parser.add_argument('--partition',choices=['smoke','calibration','both'],default='both');args=parser.parse_args()
    if not ROOT.resolve().is_relative_to(Path('/kaggle/working').resolve()):raise RuntimeError('Real v04 execution is permitted only in the frozen Kaggle working directory')
    config=yaml.safe_load((ROOT/'configs/experiment.yaml').read_text());online=yaml.safe_load((ROOT/'configs/online_v04.yaml').read_text());study=yaml.safe_load((ROOT/'configs/upgrade_v02.yaml').read_text());validate_online_policy(config)
    if online['runtime_entry_enabled'] is not True:raise RuntimeError('Frozen payload entry is disabled')
    fingerprint();check_versions(ROOT/'configs/requirements-runtime-v03.txt');import torch
    rt=online['runtime'];env={'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,'gpus':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        'dependencies':{n:md.version(n) for n in ['transformers','accelerate','huggingface_hub','pyarrow','pyyaml','numpy','tokenizers']},'device':'cuda:0','batch':1,'concurrency':1,'fingerprint':fingerprint()}
    write(ROOT/'logs/environment.json',env)
    if len(env['gpus'])!=2 or any('T4' not in n for n in env['gpus']) or env['torch']!=rt['torch_required'] or env['cuda']!=rt['cuda_required'] or platform.python_version().rsplit('.',1)[0]!=rt['python_major_minor']:raise RuntimeError('Frozen GPU/runtime environment differs; diagnostic saved')
    torch.manual_seed(42);torch.cuda.manual_seed_all(42);ledger=[];start=time.perf_counter();model,mounted,working=prepare_mounted_assets(ROOT,read(ROOT/'configs/mounts.json'),ledger)
    config['model']['local_model_path']=str(model);load_start=time.perf_counter();record={'kind':'model_load','status':'running'};ledger.append(record)
    try:g=LightweightGenerator(config['model'],read(ROOT/'configs/model_asset_lock.json')['revision']);record['status']='ok'
    except Exception as exc:record.update(status='failed',error_type=type(exc).__name__,error=str(exc));raise
    finally:record['ms']=1000*(time.perf_counter()-load_start);write(ROOT/'logs/preparation.json',ledger)
    retriever=Retriever(working,config['retrieval']);g.calls=[];warm=time.perf_counter();g.generate(config['generation']['answer_system'],'Question: What is 1 plus 1?',8,'warmup',collect_signals=False)
    write(ROOT/'logs/generation_warmup.json',{'generation_calls':list(g.calls),'ms':1000*(time.perf_counter()-warm),'scope':'preparation_not_policy'})
    router=read(ROOT/online['router']);baselines=read(ROOT/online['baselines']);deadline=process_started+rt['wall_budget_seconds']
    try:
        for partition in ['smoke','calibration'] if args.partition=='both' else [args.partition]:
            run(partition,g,retriever,config,study,online,router,baselines,deadline);print('Saved:',archive(ROOT,partition),flush=True)
    finally:retriever.connection.close();write(ROOT/'logs/runtime_wall.json',{'ms':1000*(time.perf_counter()-start),'test_executed':False})

if __name__=='__main__':
    process_start=time.perf_counter();state={'status':'running'}
    try:main();state['status']='ok'
    except BaseException as exc:
        state.update(status='failed',error_type=type(exc).__name__)
        write(ROOT/'logs/fatal.json',{'error_type':type(exc).__name__,'error':str(exc),'saved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()});raise
    finally:state['ms']=1000*(time.perf_counter()-process_start);write(ROOT/'logs/runtime_process.json',state)
