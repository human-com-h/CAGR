"""Unchanged estimator functions extracted from the original analysis source."""
import numpy as np
import collections
METHODS=['direct', 'always_rag', 'always_multihop', 'matched_binary_v2', 'targ_style', 'sugar_l_style_budgeted', 'cagr_v1', 'cagr_v2']
SETS=['hotpotqa','triviaqa','overall']
def stratified_draw(question_rows, repeats=20000):
    rng = np.random.default_rng(42)
    parts = []
    for ds in ['hotpotqa', 'triviaqa']:
        for shard in [1, 2]:
            indices = np.array([i for i, r in enumerate(question_rows)
                                if r['dataset'] == ds and r['shard'] == shard])
            if len(indices):
                parts.append(rng.choice(indices, (repeats, len(indices)), replace=True))
    if not parts:
        raise ValueError('No registered strata')
    return np.concatenate(parts, axis=1)

def bootstrap(values, draw):
    a = np.asarray(values, dtype=float)
    if a.ndim != 1 or not len(a) or not np.isfinite(a).all():
        raise ValueError('Finite per-question observations required')
    means = a[draw].mean(1)
    return dict(n=len(a), mean=float(a.mean()), sd=float(a.std(ddof=1)),
        median=float(np.median(a)), min=float(a.min()), max=float(a.max()),
        ci95=np.quantile(means, [.025, .975]).tolist(),
        upper_one_sided95=float(np.quantile(means, .95)))

def method_summary(rr):
    if not rr:
        raise ValueError('Empty method')
    result = {k: float(np.mean([r['quality'][k] for r in rr])) for k in ['em', 'f1']}
    result.update({k: float(np.mean([r['cost'][k] for r in rr])) for k in
        ['end_to_end_ms', 'retrieval_calls', 'generation_calls', 'total_input_tokens',
         'total_output_tokens', 'routing_total_ms', 'generation_ms', 'retrieval_ms']})
    result.update(n=len(rr), tier_counts=[sum(r['tier'] == j for r in rr) for j in range(3)],
        p50_ms=float(np.median([r['cost']['end_to_end_ms'] for r in rr])),
        p95_ms=float(np.quantile([r['cost']['end_to_end_ms'] for r in rr], .95)))
    return result

def summarize(scored, pairs, comparisons=True):
    index = {(r['sample_id'], r['method']): r for r in scored}
    ids = sorted({r['sample_id'] for r in scored})
    if len(index) != len(scored) or set(index) != {(s, m) for s in ids for m in METHODS}:
        raise ValueError('Missing/duplicate normal observations')
    ratios = collections.defaultdict(list)
    for p in pairs:
        ratios[p['sample_id']].append(p['ratio'])
    if set(ratios) != set(ids) or any(len(v) != 2 for v in ratios.values()):
        raise ValueError('Exactly two pairs per question required')
    report = dict(methods={}, overhead={}, paired={}, quality_mean_floor={})
    for ds in SETS:
        ss = [s for s in ids if ds == 'overall' or index[s, 'cagr_v2']['dataset'] == ds]
        qq = [index[s, 'cagr_v2'] for s in ss]
        draw = stratified_draw(qq)
        report['methods'][ds] = {m: method_summary([index[s, m] for s in ss]) for m in METHODS}
        full = bootstrap([r['cost']['routing_total_ms']/r['cost']['end_to_end_ms'] for r in qq], draw)
        independent = bootstrap([float(np.mean(ratios[s])) for s in ss], draw)
        report['overhead'][ds] = dict(full_components=full, independent_on_off=independent,
            full_U95_below_0_5_percent=full['upper_one_sided95'] < .005,
            independent_U95_below_0_5_percent=independent['upper_one_sided95'] < .005,
            ratio_of_system_totals=sum(r['cost']['routing_total_ms'] for r in qq)/sum(r['cost']['end_to_end_ms'] for r in qq))
        report['quality_mean_floor'][ds] = {k: report['methods'][ds]['cagr_v2'][k] >=
            report['methods'][ds]['always_rag'][k] - .01 - 1e-12 for k in ['em', 'f1']}
        report['paired'][ds] = {}
        if comparisons:
            for other in [m for m in METHODS if m != 'cagr_v2']:
                d = {}
                for metric in ['em', 'f1', 'end_to_end_ms', 'retrieval_calls', 'total_input_tokens', 'total_output_tokens']:
                    key = 'quality' if metric in ['em', 'f1'] else 'cost'
                    d[metric] = bootstrap([index[s, 'cagr_v2'][key][metric]-index[s, other][key][metric] for s in ss], draw)
                a = np.array([index[s, 'cagr_v2']['cost']['end_to_end_ms'] for s in ss])
                b = np.array([index[s, other]['cost']['end_to_end_ms'] for s in ss])
                speeds = b[draw].mean(1)/a[draw].mean(1)
                d['speedup'] = dict(ratio_of_means=float(b.mean()/a.mean()), ci95=np.quantile(speeds, [.025, .975]).tolist())
                report['paired'][ds][other] = d
    return report
