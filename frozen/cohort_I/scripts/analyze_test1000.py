"""Fixed per-question test statistics and two-shard coverage/payment merge."""
from pathlib import Path
import argparse,json
import numpy as np
from test1000_contract import METHODS,read,write,sha,jsonl,samples,verify_prediction_matrix
from v04_v3_gates import bootstrap as boot
def stratified_draw(rows,n=20000):
    rng=np.random.default_rng(42);parts=[]
    for ds in ['hotpotqa','triviaqa']:
        for shard in sorted({r['shard'] for r in rows}):
            ii=[i for i,r in enumerate(rows) if r['dataset']==ds and r['shard']==shard]
            if ii:parts.append(rng.choice(ii,(n,len(ii)),replace=True))
    return np.concatenate(parts,axis=1)
def summarize(rows,pairs):
    ids=sorted({r['sample_id'] for r in rows});idx={(r['sample_id'],r['method']):r for r in rows}
    if len(rows)!=len(ids)*8 or len(idx)!=len(rows):raise RuntimeError('Scored matrix duplicate/missing')
    for sid in ids:
        if {m for s,m in idx if s==sid}!=set(METHODS):raise RuntimeError('Missing scored method')
    ratios={sid:[] for sid in ids}
    for p in pairs:ratios[p['sample_id']].append(p['ratio'])
    if any(len(v)!=2 for v in ratios.values()):raise RuntimeError('Fixed two-pair statistics required')
    report={'scope':'held_out_generalization_no_tuning','methods':{},'paired':{},'overhead':{},'quality_mean_floor':{},'questions':len(ids),'test_result_changes_frozen_method':False}
    for ds in ['hotpotqa','triviaqa','overall']:
        ss=[sid for sid in ids if ds=='overall' or idx[sid,'cagr']['dataset']==ds];qr=[idx[sid,'cagr'] for sid in ss];draw=stratified_draw(qr)
        report['methods'][ds]={};report['paired'][ds]={}
        for method in METHODS:
            rr=[idx[s,method] for s in ss];v={k:float(np.mean([r['quality'][k] for r in rr])) for k in ['em','f1']}
            v.update({k:float(np.mean([r['cost'][k] for r in rr])) for k in ['end_to_end_ms','retrieval_calls','total_input_tokens','total_output_tokens','routing_total_ms']})
            v.update(tier_counts=[sum(r['tier']==j for r in rr) for j in range(3)],p50_ms=float(np.median([r['cost']['end_to_end_ms'] for r in rr])),p95_ms=float(np.quantile([r['cost']['end_to_end_ms'] for r in rr],.95)))
            report['methods'][ds][method]=v
        # Dataset estimates use the SAME registered mean-of-paired-ratios and
        # question bootstrap as development, with 500 questions after merging.
        full=boot([r['cost']['routing_total_ms']/r['cost']['end_to_end_ms'] for r in qr]);ind=boot([float(np.mean(ratios[s])) for s in ss])
        report['overhead'][ds]={'full_components':full,'independent_on_off':ind,'full_upper_below_0_5_percent':full['upper_one_sided95']<.005,'independent_upper_below_0_5_percent':ind['upper_one_sided95']<.005}
        report['quality_mean_floor'][ds]={k:report['methods'][ds]['cagr'][k]>=report['methods'][ds]['always_rag'][k]-.01-1e-12 for k in ['em','f1']}
        for other in ['binary_gate','targ_style','always_rag','always_multihop','sugar_l_style_budgeted']:
            d={}
            for metric in ['em','f1','end_to_end_ms','retrieval_calls','total_input_tokens','total_output_tokens']:
                key='quality' if metric in ['em','f1'] else 'cost';v=np.array([idx[s,'cagr'][key][metric]-idx[s,other][key][metric] for s in ss]);b=v[draw].mean(1)
                d[metric]={'mean':float(v.mean()),'ci95':np.quantile(b,[.025,.975]).tolist(),'upper_one_sided95':float(np.quantile(b,.95))}
            a=np.array([idx[s,'cagr']['cost']['end_to_end_ms'] for s in ss]);b=np.array([idx[s,other]['cost']['end_to_end_ms'] for s in ss]);speed=b[draw].mean(1)/a[draw].mean(1)
            d['speedup']={'ratio_of_means':float(b.mean()/a.mean()),'ci95':np.quantile(speed,[.025,.975]).tolist()};report['paired'][ds][other]=d
    return report
def preparation_ledger(root):
    root=Path(root);extra=[];details={}
    for name in ['test_model_warmup.json','test_cache_preflight.jsonl']:
        path=root/'logs'/name
        if path.exists():
            data=jsonl(path) if path.suffix=='.jsonl' else [read(path)]
            for e in data:extra.extend(e['generation_calls'])
            details[name]={'generation_calls':sum(len(e['generation_calls']) for e in data),'file_sha256':sha(path)}
        else:details[name]={'status':'not_recorded_unknown_not_zero'}
    for name in ['preparation.json','dependency_preparation.json','test_native_numeric.json','test_prediction_status.json','notebook_test_status.json','bootstrap_cost.json']:
        path=root/'logs'/name;details[name]=read(path) if path.exists() else {'status':'not_recorded_unknown_not_zero'}
    warm=root/'logs/test_shared_retrieval_warmup.jsonl';rows=jsonl(warm) if warm.exists() else []
    return {'extra_model_calls':len(extra),'extra_input_tokens':sum(c['input_tokens'] for c in extra),'extra_output_tokens':sum(c['output_tokens'] for c in extra),'extra_generation_ms':sum(c['ms'] for c in extra),
        'shared_retrieval_warmup_calls':len(rows),'shared_retrieval_warmup_ms':sum(r['ms'] for r in rows),'details':details,
        'accounting':'Normal/technical/extras disjoint. Preparation/process/numeric/scoring/archives are inclusive scopes, never summed with their children; absent costs UNKNOWN, not zero.'}
def merge(root1,root2,manifest,budget,output,compute_statistics=True):
    root1,root2,output=map(Path,[root1,root2,output]);manifest=read(manifest);allq=[];normal=[];tech=[];pairs=[];scored=[];receipts=[];contracts=[]
    for k,root in enumerate([root1,root2],1):
        q=samples(root,k);out=root/'results/test1000';nr=jsonl(out/'predictions.jsonl');tr=jsonl(out/'technical_executions.jsonl');pr=jsonl(out/'paired_observations.jsonl')
        coverage=verify_prediction_matrix(root,k,q,nr,tr,pr,budget)
        from test1000_contract import require_prediction_commit
        require_prediction_commit(root,k);receipt=read(out/'scoring_receipt.json')
        if receipt['status']!='complete_offline_scoring' or sha(out/'scored.jsonl')!=receipt['scored_sha256'] or sha(out/'summary.json')!=receipt['summary_sha256']:raise RuntimeError('Scored receipt SHA differs')
        sr=jsonl(out/'scored.jsonl')
        if len(sr)!=4000 or any({z:v for z,v in r.items() if z!='quality'}!=n for r,n in zip(sr,nr)):raise RuntimeError('Scoring modified predictions')
        allq+=q;normal+=nr;tech+=tr;pairs+=pr;scored+=sr;receipts.append(coverage);contracts.append(read(out/'summary.json')['common_contract'])
    if contracts[0]!=contracts[1]:raise RuntimeError('Shards used different frozen method/deployment')
    expected={}
    for k in [1,2]:
        # Manifest binds exact deterministic membership and source file SHA.
        expected[k]=manifest['shards'][str(k)]['questions_sha256']
    if len(allq)!=1000 or len({q['sample_id'] for q in allq})!=1000:raise RuntimeError('Shard IDs duplicate')
    if len(allq)!=1000 or {q['global_index'] for q in allq}!=set(range(1000)) or len(normal)!=8000 or len(tech)!=4000 or len(pairs)!=2000:raise RuntimeError('Merged coverage differs')
    if len({r['execution_id'] for r in normal+tech})!=12000:raise RuntimeError('Merged cost IDs duplicated')
    if any(sum(q['dataset']==d for q in allq)!=500 for d in ['hotpotqa','triviaqa']):raise RuntimeError('Merged dataset quotas')
    for k,root in enumerate([root1,root2],1):
        if sha(root/f'data/test1000/shard_{k}.jsonl')!=expected[k]:raise RuntimeError('Merged manifest membership SHA')
    if output.exists():raise RuntimeError('Never overwrite merged results')
    output.mkdir(parents=True);report=summarize(scored,pairs) if compute_statistics else {'scope':'local_synthetic_control_only_not_GPU'}
    report.update(shards=[1,2],normal_rows=8000,technical_executions=4000,paired_observations=2000,coverage=receipts,common_contract=contracts[0],no_duplicate_costs=True,
        preparation_and_extra_costs={str(k):preparation_ledger(root) for k,root in enumerate([root1,root2],1)},
        source_shards={str(k):{'root':str(root.resolve()),'scoring_receipt_sha256':sha(root/'results/test1000/scoring_receipt.json'),'prediction_commit_sha256':sha(root/'results/test1000/prediction_commit.json')} for k,root in enumerate([root1,root2],1)})
    write(output/'merged_summary.json',report);return report
def main():
    import yaml
    p=argparse.ArgumentParser();p.add_argument('--shard-1',type=Path,required=True);p.add_argument('--shard-2',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    b=yaml.safe_load((a.shard_1/'configs/test1000.yaml').read_text())['common_budget'];r=merge(a.shard_1,a.shard_2,a.manifest,b,a.output);print(json.dumps({'normal_rows':r['normal_rows'],'questions':r['questions']}))
if __name__=='__main__':main()
