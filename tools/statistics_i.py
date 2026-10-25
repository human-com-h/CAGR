"""Unchanged estimator functions extracted from the original analysis source."""
import numpy as np
import collections
METHODS=['direct', 'always_rag', 'always_multihop', 'binary_gate', 'targ_style', 'sugar_l_style_budgeted', 'cagr', 'cagr_in_situ']
SETS=['hotpotqa','triviaqa','overall']
def bootstrap(values):
    v=np.asarray(values,dtype=float)
    if v.ndim!=1 or not len(v) or not np.isfinite(v).all():raise ValueError('Finite nonempty paired values required')
    rng=np.random.default_rng(42);draw=v[rng.integers(0,len(v),(20000,len(v)))].mean(1)
    return {'n':len(v),'mean':float(v.mean()),'ci95':np.quantile(draw,[.025,.975]).tolist(),'upper_one_sided95':float(np.quantile(draw,.95))}
boot=bootstrap
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
