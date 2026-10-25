"""Held-out contract: frozen deployment, coverage, payment and score barrier."""
from pathlib import Path
import copy,hashlib,json,os,random
from v03_assets import read,write,sha
from execution_policy import load_policy
from run_online_v04 import validate_row
from v04_v3_gates import scientific_sha256
METHODS=['direct','always_rag','always_multihop','binary_gate','targ_style','sugar_l_style_budgeted','cagr','cagr_in_situ']
PAIRS=2
def jsonl(path):return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]
def append(path,row):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
def encode_lines(rows):return ''.join(json.dumps(x,ensure_ascii=False,sort_keys=True)+'\n' for x in rows).encode()
def method_order(index):
    m=list(METHODS);random.Random(42+index//8).shuffle(m);s=index%8;return m[s:]+m[:s]
def pair_order(index,repeat):
    if repeat not in [0,1]:raise ValueError('Fixed two-pair design')
    return ['on','off'] if (index+repeat)%2==0 else ['off','on']
def deployment_view(router,baselines):
    r=copy.deepcopy(router);r.pop('calibration_fold_heads',None)
    b=copy.deepcopy(baselines)
    for m in ['targ_style','sugar_l_style_budgeted']:b[m].pop('calibration_fold_parameters',None)
    return {'router':r,'baselines':b,'head_scope':'all_200_development','fold':None}
def parameters(method,deployment):
    if deployment['head_scope']!='all_200_development' or deployment['fold'] is not None:raise RuntimeError('Deployment cannot use OOF')
    p=deployment['baselines'][method] if method in ['targ_style','sugar_l_style_budgeted'] else deployment['router']
    if 'calibration_fold_heads' in p or 'calibration_fold_parameters' in p:raise RuntimeError('OOF parameters must not reach test inference')
    return p
def validate_policy(root,shard):
    p=load_policy(root);a=p.get('test_1000_authorization',{});f=read(Path(root)/'configs/method_freeze_v04_v3_native_state.json')
    if not (a.get('authorized') and shard in a.get('allowed_shards',[]) and a.get('currently_registered_shard')==shard and a.get('reference_access')=='after_shard_prediction_commit_only' and a.get('fixed_pairs_per_question')==2 and a.get('OOF_forbidden') and f['method_frozen']):raise RuntimeError('Explicit frozen-test authorization required')
    if p['assets']['allow_runtime_model_or_data_download'] or p['research']['seed']!=42 or p['experiment_submission']['agent_may_restart_cloud_run']:raise RuntimeError('Offline/seed/no-restart contract')
    return p
def samples(root,shard):
    root=Path(root);m=read(root/'data/test1000/manifest.json');e=m['shards'][str(shard)];path=root/e['questions_file']
    if sha(path)!=e['questions_sha256']:raise RuntimeError('Test question SHA differs')
    q=jsonl(path)
    if len(q)!=500 or len({s['sample_id'] for s in q})!=500:raise RuntimeError('Shard question coverage')
    allowed={'sample_id','dataset','source_split','question','selection_sha256','source_index','global_index','shard'}
    if any(set(s)-allowed or s['shard']!=shard for s in q):raise RuntimeError('Labels/folds/unexpected fields in inference questions')
    if any(sum(s['dataset']==ds for s in q)!=250 for ds in ['hotpotqa','triviaqa']):raise RuntimeError('Shard dataset quotas')
    if [s['global_index'] for s in q]!=list(range((shard-1)*500,shard*500)):raise RuntimeError('Frozen interleaved test order changed')
    return q
def fingerprint(root):
    root=Path(root);p=read(root/'package_manifest.json');files=p['files']
    for n,e in files.items():
        if sha(root/n)!=e['sha256']:raise RuntimeError('Frozen inference payload changed: '+n)
    return hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
def path_view(row):
    return [row['answer'],row['tier'],row['context_ids'],[[c['kind'],c['prompt_sha256'],c['input_tokens'],c['output_token_ids']] for c in row['generation_calls']],
        [{k:v for k,v in x.items() if k!='ms' and not k.endswith('_ms') and k!='saved_utc'} for x in row['retrieval_trace']]]
def validate_execution(row,budget):
    validate_row(row,budget)
    if row.get('fold') is not None or row.get('parameter_scope')!='all_200_development':raise RuntimeError('Test OOF leakage')
    if 'quality' in row or 'answers' in row:raise RuntimeError('Labels cannot enter prediction ledger')
    for c in row['generation_calls']:
        if c.get('signal_added_explicit_device_syncs')!=0 or c.get('signal_cuda_event_records')!=0:raise RuntimeError('Frozen event/sync violated')
        if c.get('signal_buffer_bytes')!=1519360 or c.get('signal_fixed_buffer_tensor_args_per_position')!=0:raise RuntimeError('Frozen Native State contract')
        if len(c.get('signal_work_intervals',[]))!=len(c['token_logit_gaps']) or len(c['token_logit_gaps'])>5:raise RuntimeError('K5 capture boundary')
        if not c.get('signal_collection','').endswith('_B'):raise RuntimeError('Frozen B implementation required')
def totals(rows):
    fields=['generation_calls','total_input_tokens','total_output_tokens','generation_ms','retrieval_calls','retrieval_ms','failed_generation_calls','failed_retrieval_calls','automatic_retries']
    return {k:sum(r['cost'][k] for r in rows) for k in fields}
def verify_prediction_matrix(root,shard,questions,normal,technical,pairs,budget):
    ids={s['sample_id'] for s in questions};idx={(r['sample_id'],r['method']):r for r in normal}
    if len(normal)!=4000 or len(idx)!=4000 or set(idx)!={(sid,m) for sid in ids for m in METHODS}:raise RuntimeError('Complete 500x8 predictions required')
    expected=[(s['sample_id'],m) for s in questions for m in method_order(s['global_index'])]
    if [(r['sample_id'],r['method']) for r in normal]!=expected:raise RuntimeError('Same-question balanced method order changed')
    allrows=normal+technical;seen=set()
    for row in allrows:
        validate_execution(row,budget);eid=row['execution_id']
        if eid in seen or row['shard']!=shard or row['sample_id'] not in ids:raise RuntimeError('Execution payment duplicate/wrong shard')
        seen.add(eid)
    for r in normal:
        if r['scope']!='held_out_normal' or r['execution_id']!=f'{shard}:{r["sample_id"]}:normal:{r["method"]}':raise RuntimeError('Normal execution ownership')
    if any(r['method']!='cagr' or r['scope']!='held_out_technical_not_policy' for r in technical):raise RuntimeError('Technical execution ownership')
    if len(technical)!=2000 or len(pairs)!=1000:raise RuntimeError('Fixed two-pair technical coverage')
    tidx={r['execution_id']:r for r in technical};pseen=set()
    for p in pairs:
        key=(p['sample_id'],p['repeat'])
        if key in pseen or key[0] not in ids:raise RuntimeError('Duplicate pair')
        pseen.add(key);q=next(x for x in questions if x['sample_id']==key[0])
        if p['order']!=pair_order(q['global_index'],p['repeat']):raise RuntimeError('Pair order differs')
        on,off=[tidx[p['execution_ids'][m]] for m in ['on','off']];ref=idx[key[0],'cagr']
        if any(p['execution_ids'][m]!=f'{shard}:{key[0]}:technical:{key[1]}:{m}' for m in ['on','off']):raise RuntimeError('Pair execution cost ID differs')
        if scientific_sha256(on)!=scientific_sha256(ref) or path_view(on)!=path_view(off):raise RuntimeError('Technical frozen route/Token/retrieval changed')
        if any(c['token_logit_gaps'] or c.get('signal_work_intervals') or c.get('signal_ms',0.)!=0 for c in off['generation_calls']):raise RuntimeError('Features-off executed signal work')
        if off['cost']['end_to_end_ms']<=0:raise RuntimeError('Invalid independent denominator')
        ratio=(on['cost']['end_to_end_ms']-off['cost']['end_to_end_ms'])/off['cost']['end_to_end_ms']
        if p['ratio']!=ratio:raise RuntimeError('Pair statistic mismatch')
    if pseen!={(sid,j) for sid in ids for j in range(2)}:raise RuntimeError('Missing pair')
    if len({r['session_id'] for r in allrows})!=1:raise RuntimeError('Same-shard GPU session differs')
    return {'normal':totals(normal),'technical':totals(technical),'normal_rows':4000,'technical_executions':2000,'paired_observations':1000,'questions':500}
def commit_predictions(root,shard,questions,budget):
    root=Path(root);out=root/'results/test1000';normal=jsonl(out/'predictions.jsonl');tech=jsonl(out/'technical_executions.jsonl');pairs=jsonl(out/'paired_observations.jsonl')
    coverage=verify_prediction_matrix(root,shard,questions,normal,tech,pairs,budget)
    receipt={'status':'all_predictions_committed_before_references','shard':shard,'fingerprint':fingerprint(root),**coverage,
        'evidence_sha256':{n:sha(out/n) for n in ['predictions.jsonl','technical_executions.jsonl','paired_observations.jsonl']},'reference_semantics_opened':False}
    write(out/'prediction_commit.json',receipt);return receipt
def require_prediction_commit(root,shard):
    root=Path(root);out=root/'results/test1000';p=out/'prediction_commit.json'
    if not p.is_file():raise RuntimeError('Reference barrier: complete predictions required before scoring')
    e=read(p)
    if e['status']!='all_predictions_committed_before_references' or e['shard']!=shard or e['normal_rows']!=4000 or e['technical_executions']!=2000 or e['fingerprint']!=fingerprint(root):raise RuntimeError('Prediction receipt incomplete or wrong fingerprint')
    for n,h in e['evidence_sha256'].items():
        if sha(out/n)!=h:raise RuntimeError('Committed predictions changed')
    import yaml
    budget=yaml.safe_load((root/'configs/test1000.yaml').read_text())['common_budget']
    verify_prediction_matrix(root,shard,samples(root,shard),jsonl(out/'predictions.jsonl'),jsonl(out/'technical_executions.jsonl'),jsonl(out/'paired_observations.jsonl'),budget)
    return e
