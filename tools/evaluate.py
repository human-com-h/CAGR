"""Public path adapter around the byte-preserved scientific implementation.

No cloud submission, fitting, credential access or reference loading occurs here.
This adapter must be GPU-validated before declaring a new run paper-equivalent.
"""
from pathlib import Path
import argparse,copy,hashlib,json,platform,sys,time
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cohort',choices=['I','II'],required=True)
    p.add_argument('--shard',type=int,choices=[1,2],required=True)
    p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--index',type=Path,required=True)
    p.add_argument('--questions',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--smoke',type=int,default=0,help='Run only N questions; never a main result')
    a=p.parse_args();sys.path.insert(0,str(ROOT/f'frozen/cohort_{a.cohort}/scripts'))
    import yaml,torch
    from public_execution import public_runtime
    public_session=public_runtime();public_receipt=public_session.__enter__()
    from run_baselines import Retriever
    from v04_runtime import run_method
    from v04_v3_native_state_checks import numeric_gate
    if a.cohort=='II':
        from kv_bridge_inference import KVBridgeGenerator as Generator
        from v2_fine_runtime import run_fine,selected_plan,science
        from v2_evaluation_plan import METHODS,method_order,pair_orders
        from heldout_v2_contract import validate_execution
    else:
        from v04_v3_native_state_inference import NativeStateGenerator as Generator
        from test1000_contract import METHODS,method_order,pair_order,parameters,path_view,validate_execution
        from v04_v3_gates import scientific_sha256
    if a.output.exists():raise RuntimeError('Output already exists; no overwrite or automatic restart')
    if not torch.cuda.is_available():raise RuntimeError('A CUDA GPU and compiler are required')
    cfgroot=ROOT/f'frozen/cohort_{a.cohort}/configs'
    cfg=yaml.safe_load((cfgroot/'experiment.yaml').read_text());study=yaml.safe_load((cfgroot/'upgrade_v02.yaml').read_text())
    protocol=yaml.safe_load((cfgroot/('test1000.yaml' if a.cohort=='I' else 'cagr_v2_evaluation.yaml')).read_text())
    d=read(cfgroot/('deployment_test1000.json' if a.cohort=='I' else 'heldout_v2_deployment.json'))
    qs=[json.loads(l) for l in a.questions.read_text(encoding='utf-8').splitlines() if l]
    expected=read(ROOT/'configs/split_ids.json')[a.cohort]
    if len(qs)!=len(expected) or any(set(r)!=(set(e)-{'question_sha256'})|{'question'} or any(r[k]!=v for k,v in e.items() if k!='question_sha256') or hashlib.sha256(r['question'].encode()).hexdigest()!=e['question_sha256'] for r,e in zip(qs,expected)):
        raise RuntimeError('Frozen question identity/order differs')
    if a.smoke<0:raise ValueError('Nonnegative --smoke required')
    qs=[s for s in qs if s['shard']==a.shard]
    if a.smoke:qs=qs[:a.smoke]
    a.output.mkdir(parents=True);budget=protocol['common_budget']
    cfg['model']['local_model_path']=str(a.model_dir.resolve())
    model_lock=read(cfgroot/'model_asset_lock.json')
    from prepare_resources import sha
    for n,e in model_lock['files'].items():
        if sha(a.model_dir/n)!=e['sha256']:raise RuntimeError('Model identity mismatch: '+n)
    index=sqlite_identity(a.index)
    expected_index=read(ROOT/'configs/corpus_expected.json')
    if index!=dict(documents=expected_index['documents'],ordered_document_ids_sha256=expected_index['ordered_document_ids_sha256']):raise RuntimeError('Index document identity differs')
    env={'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'device':'cuda:0','batch':1,'concurrency':1,'seed':42,'smoke_questions':a.smoke,'public_permission_adapter':public_receipt,
        'scope':'public_adapter_run_requires_independent_comparison','index':index}
    (a.output/'environment.json').write_text(json.dumps(env,indent=2))
    torch.cuda.set_device(0);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    started=time.perf_counter();g=Generator(cfg['model'],model_lock['revision']);g.signal_variant='B'
    if a.cohort=='II':g.bridge_options=yaml.safe_load((cfgroot/'cagr_v2_route23.yaml').read_text())['bridge']
    preparation={'model_load_and_native_build_ms':1000*(time.perf_counter()-started)}
    g.calls=[];g.generate(cfg['generation']['answer_system'],'Question: What is 1 plus 1?',8,'warmup',False)
    preparation['warmup_generation_calls']=copy.deepcopy(g.calls);preparation['numeric_gate']=numeric_gate(g)
    from run_online_v04 import cache_check
    smoke_path=a.questions.parent/'smoke_questions.jsonl'
    if not smoke_path.is_file():raise RuntimeError('Restored smoke_questions.jsonl required beside the cohort questions')
    preparation['cache_preflight']=[cache_check(s,g,cfg) for s in map(json.loads,smoke_path.read_text(encoding='utf-8').splitlines())]
    (a.output/'preparation.json').write_text(json.dumps(preparation,indent=2))
    r=Retriever(a.index,cfg['retrieval'])
    def append(file,row):
        with (a.output/file).open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    guard_seconds=protocol.get('operating_guard_seconds',protocol.get('budget',{}).get('internal_wall_guard_seconds',42000))
    shard_started={}
    def execute(s,m,scope,mode='on',plan=None):
        if time.monotonic()-shard_started[s['shard']]>guard_seconds:raise RuntimeError('Frozen per-shard wall guard reached; partial costs retained')
        g.calls=[];g.signal_collection_enabled=mode!='off';torch.manual_seed(42);torch.cuda.manual_seed_all(42);t=time.perf_counter()
        try:
            if a.cohort=='II' and m in ['cagr_v2','matched_binary_v2']:
                row=run_fine(s['question'],s['dataset'],g,r,cfg,budget,d['fine'],study,binary=m=='matched_binary_v2',plan=plan)
            else:
                name='cagr' if m=='cagr_v1' else m
                pars=parameters(m,d) if a.cohort=='I' else d['baselines'][m] if m in ['targ_style','sugar_l_style_budgeted'] else d['router']
                row=run_method(name,s['question'],g,r,cfg,budget,pars,study=study,fold=None,technical_plan=plan)
            row.update(sample_id=s['sample_id'],dataset=s['dataset'],shard=s['shard'],global_index=s['global_index'],method=m,scope=scope,fold=None,
                parameter_scope='all_200_development' if a.cohort=='I' else 'full_200_development',attempt=1,retry=False,attempt_wall_ms=1000*(time.perf_counter()-t))
            append('predictions.jsonl' if scope=='held_out_normal' else 'technical_executions.jsonl',row)
            validate_execution(row,budget);return row
        except BaseException as exc:
            append('failures.jsonl',{'sample_id':s['sample_id'],'method':m,'scope':scope,'error_type':type(exc).__name__,'wall_ms':1000*(time.perf_counter()-t),'generation_calls':g.calls,'unreturned_cost':'unknown_not_zero'})
            raise
        finally:
            g.signal_collection_enabled=True
            if hasattr(g,'drop_retained'):g.drop_retained()
    try:
        for s in qs:
            shard_started.setdefault(s['shard'],time.monotonic())
            _,trace=r.search(s['question']);append('shared_warmup.jsonl',{'sample_id':s['sample_id'],'trace':trace,'scope':'shared_preparation_not_policy'})
            for m in method_order(s['global_index']):
                row=execute(s,m,'held_out_normal')
                if m==('cagr' if a.cohort=='I' else 'cagr_v2'):mainrow=row
            if a.cohort=='II':plan=selected_plan(mainrow);orders=pair_orders(s['global_index'],2)
            else:
                plan=[False,False]
                for v in mainrow['control_trace']:
                    if v['kind']=='decision_value':plan[v['stage']]=bool(v['selected'])
                orders=[pair_order(s['global_index'],i) for i in range(2)]
            for repeat,order in enumerate(orders):
                group={mode:execute(s,'cagr' if a.cohort=='I' else 'cagr_v2','held_out_technical_not_policy',mode,plan if mode=='off' else None) for mode in order}
                exact=(science(group['on'])==science(group['off'])==science(mainrow)) if a.cohort=='II' else (scientific_sha256(group['on'])==scientific_sha256(mainrow) and path_view(group['on'])==path_view(group['off']))
                if not exact:raise RuntimeError('On/off scientific equivalence failed')
                off=group['off']['cost']['end_to_end_ms']
                if off<=0:raise RuntimeError('Nonpositive denominator')
                append('paired_observations.jsonl',{'sample_id':s['sample_id'],'dataset':s['dataset'],'shard':s['shard'],'repeat':repeat,'order':order,'ratio':(group['on']['cost']['end_to_end_ms']-off)/off})
        (a.output/'prediction_commit.json').write_text(json.dumps({'questions':len(qs),'normal_rows':len(qs)*8,'technical_rows':len(qs)*4,'pairs':len(qs)*2,
            'sha256':{n:sha(a.output/n) for n in ['predictions.jsonl','technical_executions.jsonl','paired_observations.jsonl']},'cohort':a.cohort,'shard':a.shard,'scope':'smoke' if a.smoke else 'full_shard'},indent=2))
    finally:
        r.connection.close();public_session.__exit__(None,None,None)

def sqlite_identity(p):
    import sqlite3
    conn=sqlite3.connect(f'file:{p.resolve().as_posix()}?mode=ro',uri=True)
    try:
        h=hashlib.sha256();n=0
        for docid,title,text in conn.execute('SELECT doc_id,title,text FROM passages ORDER BY rowid'):
            if hashlib.sha256(json.dumps([title,text],ensure_ascii=False,separators=(',',':')).encode()).hexdigest()!=docid:raise RuntimeError('Passage content differs from its identity')
            h.update((docid+'\n').encode());n+=1
        return {'documents':n,'ordered_document_ids_sha256':h.hexdigest()}
    finally:conn.close()
if __name__=='__main__':main()
