"""Full-deployment, prediction-only contract for the two fresh heldout-v2 shards."""
from pathlib import Path
import hashlib,json,math,os
import yaml
from v03_assets import read,write,sha
from execution_policy import load_policy
from v2_evaluation_plan import METHODS,method_order,pair_orders
from v2_fine_runtime import science,selected_plan
from route23_runtime import validate_row as validate_cache_budget


def rows(path):
    with Path(path).open(encoding='utf-8') as f:return [json.loads(s) for s in f if s.strip()]


def append(path,row):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())


def canonical_sha(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def deployment(root):
    root=Path(root);cfg=yaml.safe_load((root/'configs/cagr_v2_evaluation.yaml').read_text(encoding='utf-8'))
    freeze=read(root/cfg['method_freeze']);d=read(root/cfg['deployment'])
    if not freeze['method_frozen'] or freeze['candidate_id']!='fine_382' or sha(root/cfg['method_freeze'])!=cfg['method_freeze_sha256']:raise RuntimeError('Frozen fine_382 method required')
    if sha(root/cfg['deployment'])!=cfg['deployment_sha256'] or d['freeze_sha256']!=cfg['method_freeze_sha256']:raise RuntimeError('Deployment SHA differs')
    if d['head_scope']!='full_200_development' or d['fold'] is not None or cfg['methods']!=list(METHODS) or cfg['seed']!=42:raise RuntimeError('Full-only eight-method/seed42 deployment required')
    def forbid(value):
        if isinstance(value,dict):
            if any(k in value for k in ['fold_heads','calibration_fold_heads','calibration_fold_parameters']):raise RuntimeError('OOF head cannot enter held-out inference')
            for v in value.values():forbid(v)
        elif isinstance(value,list):
            for v in value:forbid(v)
    forbid(d)
    if canonical_sha(d['fine']['deployment'])!=freeze['deployment_head_canonical_sha256'] or d['candidate_source_sha256']!=freeze['candidate_sha256']:raise RuntimeError('Frozen Full head changed')
    return cfg,d


def validate_policy(root,shard):
    root=Path(root);p=load_policy(root);a=p.get('heldout_v2_authorization',{})
    if not (a.get('authorized') and shard in a.get('allowed_shards',[]) and a.get('currently_registered_shard')==shard and a.get('fixed_pairs_per_question')==2 and a.get('OOF_forbidden') and a.get('no_intermediate_quality_gate')):raise RuntimeError('Explicit fresh two-shard authorization required')
    if p['assets']['allow_runtime_model_or_data_download'] or p['research']['seed']!=42 or p['experiment_submission']['agent_may_restart_cloud_run']:raise RuntimeError('Offline/seed/no-restart contract')
    cfg,_=deployment(root)
    if a.get('method_freeze_sha256')!=cfg['method_freeze_sha256'] or cfg['budget']['remote_timeout_seconds']!=43200 or cfg['budget']['internal_wall_guard_seconds']!=42000:raise RuntimeError('Frozen method or 12h budget changed')
    return p


def fingerprint(root):
    root=Path(root);files=read(root/'package_manifest.json')['files']
    for n,e in files.items():
        if sha(root/n)!=e['sha256']:raise RuntimeError('Frozen payload changed: '+n)
    return canonical_sha(files)


def samples(root,shard):
    root=Path(root);m=read(root/'data/heldout-v2-run/manifest.json');entry=m['shards'][str(shard)];f=root/entry['questions_file']
    if sha(f)!=entry['questions_sha256']:raise RuntimeError('Question SHA differs')
    q=rows(f);allowed={'sample_id','dataset','question','selection_index','global_index','shard'}
    if len(q)!=500 or len({s['sample_id'] for s in q})!=500 or any(set(s)!=allowed or s['shard']!=shard for s in q):raise RuntimeError('Question schema/coverage/labels differ')
    if [s['global_index'] for s in q]!=list(range((shard-1)*500,shard*500)):raise RuntimeError('Fixed question order changed')
    if [s['dataset'] for s in q]!=['hotpotqa','triviaqa']*250:raise RuntimeError('Balanced shard differs')
    return q


def validate_execution(row,budget):
    if row['status'] not in ['ok','budget_limited_answer']:raise RuntimeError('Failed physical execution: '+str(row.get('error')))
    validate_cache_budget({**row,'status':'ok'},budget)
    if row.get('fold') is not None or row.get('parameter_scope')!='full_200_development':raise RuntimeError('OOF leakage')
    if 'quality' in row or 'answers' in row or row['cost']['untracked_backend_cost'] or row.get('retry_trace') or row.get('retry'):raise RuntimeError('References/retry/unknown cost in inference')
    c=row['cost'];calls=row['generation_calls'];ret=row['retrieval_trace']
    if c['generation_calls']!=len(calls) or c['retrieval_calls']!=len(ret):raise RuntimeError('Physical call ownership changed')
    if any(not math.isfinite(c[k]) or c[k]<0 for k in ['end_to_end_ms','generation_ms','retrieval_ms','routing_total_ms']):raise RuntimeError('Invalid costs')
    for field,key in [('total_input_tokens','input_tokens'),('total_output_tokens','output_tokens')]:
        if c[field]!=sum(x[key] for x in calls):raise RuntimeError('Unbilled tokens')
    for field,expected in [('generation_ms',sum(x['ms'] for x in calls)),('retrieval_ms',sum(x['ms'] for x in ret))]:
        if not math.isclose(c[field],expected,rel_tol=0,abs_tol=1e-8):raise RuntimeError('Unbilled call time')
    for call in calls:
        if call['status']!='ok' or call.get('attempt',0)!=0 or call['output_tokens']>budget['max_answer_tokens']:raise RuntimeError('Generation failure/retry/answer cap')
        if call['kind']=='query_rewrite' and call['output_tokens']>budget['max_rewrite_tokens']:raise RuntimeError('Rewrite budget changed')
        if call['kind']=='kv_bridge':
            parent=next(x for x in calls if x['kind']=='rag1')
            if call['reused_context_tokens']!=parent['cache_final_sequence_length'] or call['parent_output_token_ids']!=parent['output_token_ids'] or call['parent_prompt_sha256']!=parent['prompt_sha256']:raise RuntimeError('Bridge parent cache changed')
        gaps=call.get('token_logit_gaps',[])
        if len(gaps)>5 or any(not math.isfinite(v) or v<0 for v in gaps):raise RuntimeError('K5/numeric signal changed')
        if gaps:
            if call.get('signal_collection')!='v3_fused_argmax_top2_B' or call.get('signal_buffer_bytes')!=1519360 or call.get('signal_fixed_buffer_tensor_args_per_position')!=0:raise RuntimeError('Frozen Native B changed')
        if call.get('signal_added_explicit_device_syncs',0)!=0 or call.get('signal_cuda_event_records',0)!=0:raise RuntimeError('Extra sync/event forbidden')
        if not row['features_enabled'] and (gaps or call.get('signal_work_intervals') or call.get('signal_ms',0.)!=0):raise RuntimeError('Features-off performed signal work')


def cost_totals(rr):
    fields=['generation_calls','total_input_tokens','total_output_tokens','generation_ms','retrieval_calls','retrieval_ms','routing_total_ms','end_to_end_ms','failures','automatic_retries']
    return {k:sum(r['cost'].get(k,0) for r in rr) for k in fields}


def verify_matrix(root,shard,normal,technical,pairs,budget):
    qs=samples(root,shard);ids={q['sample_id'] for q in qs};allrows=normal+technical
    index={(r['sample_id'],r['method']):r for r in normal}
    if len(normal)!=4000 or set(index)!={(s,m) for s in ids for m in METHODS}:raise RuntimeError('Need all 500x8 normal predictions')
    if [(r['sample_id'],r['method']) for r in normal]!=[(q['sample_id'],m) for q in qs for m in method_order(q['global_index'])]:raise RuntimeError('Balanced same-question method order changed')
    if len(technical)!=2000 or len(pairs)!=1000 or len({r['execution_id'] for r in allrows})!=6000:raise RuntimeError('Physical counts/duplicate cost ID')
    expected_technical=[f'{shard}:{q["sample_id"]}:technical:{j}:{mode}' for q in qs for j,order in enumerate(pair_orders(q['global_index'],2)) for mode in order]
    if [r['execution_id'] for r in technical]!=expected_technical:raise RuntimeError('Actual on/off execution order changed')
    if len({r['session_id'] for r in allrows})!=1:raise RuntimeError('Same-shard GPU session changed')
    qindex={q['sample_id']:q for q in qs};tindex={r['execution_id']:r for r in technical};seen=set();used=set()
    for r in allrows:
        if r['sample_id'] not in ids or r['shard']!=shard or r['global_index']!=qindex[r['sample_id']]['global_index'] or r['dataset']!=qindex[r['sample_id']]['dataset']:raise RuntimeError('Wrong shard/question ownership')
        validate_execution(r,budget)
    for r in normal:
        if r['scope']!='held_out_normal' or r['execution_id']!=f'{shard}:{r["sample_id"]}:normal:{r["method"]}':raise RuntimeError('Normal ownership')
    for p in pairs:
        key=p['sample_id'],p['repeat'];q=qindex.get(key[0])
        if q is None or key in seen or key[1] not in [0,1] or p['dataset']!=q['dataset']:raise RuntimeError('Wrong or duplicate pair')
        seen.add(key);order=list(pair_orders(q['global_index'],2)[key[1]])
        if p['order']!=order:raise RuntimeError('Fixed pair balance changed')
        group={}
        for mode in ['on','off']:
            eid=f'{shard}:{key[0]}:technical:{key[1]}:{mode}'
            if p['execution_ids'][mode]!=eid:raise RuntimeError('Pair cost identity changed')
            row=tindex[eid];used.add(eid);group[mode]=row
            if row['method']!='cagr_v2' or row['scope']!='held_out_technical_not_policy' or row['features_enabled']!=(mode=='on'):raise RuntimeError('Technical ownership')
        reference=index[key[0],'cagr_v2'];on,off=group['on'],group['off']
        if science(on)!=science(off) or science(on)!=science(reference) or selected_plan(on)!=selected_plan(off) or selected_plan(on)!=selected_plan(reference):raise RuntimeError('On/off changed science or frozen decisions')
        denominator=off['cost']['end_to_end_ms']
        if denominator<=0 or p['ratio']!=(on['cost']['end_to_end_ms']-denominator)/denominator:raise RuntimeError('Invalid pair ratio')
    if seen!={(sid,j) for sid in ids for j in [0,1]} or used!=set(tindex):raise RuntimeError('Missing pair or unpaid technical row')
    return dict(questions=500,normal_rows=4000,technical_executions=2000,paired_observations=1000,normal_cost=cost_totals(normal),technical_cost=cost_totals(technical))


def commit_predictions(root,shard,budget):
    root=Path(root);out=root/'results/heldout_v2'
    coverage=verify_matrix(root,shard,rows(out/'predictions.jsonl'),rows(out/'technical_executions.jsonl'),rows(out/'paired_observations.jsonl'),budget)
    receipt=dict(status='all_predictions_committed_before_references',shard=shard,fingerprint=fingerprint(root),**coverage,
        evidence_sha256={n:sha(out/n) for n in ['predictions.jsonl','technical_executions.jsonl','paired_observations.jsonl']},reference_values_opened=False)
    if (out/'prediction_commit.json').exists():raise RuntimeError('Never overwrite a prediction commit')
    write(out/'prediction_commit.json',receipt);return receipt


def require_commit(root,shard):
    root=Path(root);out=root/'results/heldout_v2';path=out/'prediction_commit.json'
    if not path.is_file():raise RuntimeError('Reference barrier: full shard commit required')
    receipt=read(path)
    if receipt['status']!='all_predictions_committed_before_references' or receipt['shard']!=shard or receipt['fingerprint']!=fingerprint(root) or receipt['reference_values_opened']:raise RuntimeError('Wrong prediction commit')
    if set(receipt['evidence_sha256'])!={'predictions.jsonl','technical_executions.jsonl','paired_observations.jsonl'}:raise RuntimeError('Incomplete SHA evidence')
    for n,h in receipt['evidence_sha256'].items():
        if sha(out/n)!=h:raise RuntimeError('Committed predictions changed')
    cfg,_=deployment(root)
    actual=verify_matrix(root,shard,rows(out/'predictions.jsonl'),rows(out/'technical_executions.jsonl'),rows(out/'paired_observations.jsonl'),cfg['common_budget'])
    if any(receipt[k]!=v for k,v in actual.items()):raise RuntimeError('Commit coverage/cost changed')
    return receipt


def verify_merge(roots):
    if len(roots)!=2:raise RuntimeError('Both fixed shards required')
    identifiers=set();contracts=[]
    for shard,root in enumerate(roots,1):
        require_commit(root,shard);qq=samples(root,shard);new={q['sample_id'] for q in qq}
        if identifiers&new:raise RuntimeError('Overlapping shards')
        identifiers.update(new);cfg,d=deployment(root);contracts.append((cfg,d))
    if len(identifiers)!=1000 or contracts[0]!=contracts[1]:raise RuntimeError('Scientific contract differs between shards')
    return dict(questions=1000,normal=8000,technical=4000,pairs=2000,scientific_contracts_identical=True)
