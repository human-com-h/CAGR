"""Preregistered v3 gates; CPU statistics only, never performs inference."""
import math,hashlib,json
import numpy as np
V2_RAG_MS=595.2593172450088
ROUTED={'cagr','cagr_in_situ','binary_gate','targ_style'}

def bootstrap(values):
    v=np.asarray(values,dtype=float)
    if v.ndim!=1 or not len(v) or not np.isfinite(v).all():raise ValueError('Finite nonempty paired values required')
    rng=np.random.default_rng(42);draw=v[rng.integers(0,len(v),(20000,len(v)))].mean(1)
    return {'n':len(v),'mean':float(v.mean()),'ci95':np.quantile(draw,[.025,.975]).tolist(),'upper_one_sided95':float(np.quantile(draw,.95))}

def scientific_view(row):
    """Only deterministic scientific fields; exclude all timing/implementation metadata."""
    def retrieval(v):return {k:x for k,x in v.items() if k!='ms' and not k.endswith('_ms') and k!='saved_utc'}
    return {'answer':row['answer'],'tier':row['tier'],'context_ids':row['context_ids'],
        'generation_calls':[{k:c.get(k) for k in ['kind','prompt_sha256','input_tokens','output_token_ids','token_logit_gaps','token_gap_exp_uncertainty']} for c in row['generation_calls']],
        'decisions':[d for d in row['control_trace'] if d['kind']=='decision_value'],
        'retrieval_trace':[retrieval(v) for v in row['retrieval_trace']]}

def scientific_sha256(row):
    return hashlib.sha256(json.dumps(scientific_view(row),sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()

def double_drop(pairs):
    """Signal significance plus observed mean wall non-rebound; CI always reported.

    Do not demand every noisy wall-time observation improve. No trimming,
    hardware correction, answer-based selection or timing subtraction here.
    """
    if not pairs:raise ValueError('Missing measured A/B versus C pairs')
    out={}
    for key in ['signal_ms','generation_ms','end_to_end_ms']:
        out[key]=bootstrap([p['candidate'][key]-p['v2'][key] for p in pairs])
    out['signal_significantly_lower']=out['signal_ms']['upper_one_sided95']<0
    out['generation_wall_no_rebound']=out['generation_ms']['mean']<=0
    out['end_to_end_no_rebound']=out['end_to_end_ms']['mean']<=0
    out['measured_overhead']=bootstrap([p['candidate']['signal_ms']/p['candidate']['generation_ms'] for p in pairs])
    out['passed']=all([out['signal_significantly_lower'],out['generation_wall_no_rebound'],out['end_to_end_no_rebound'],out['measured_overhead']['upper_one_sided95']<.005])
    return out

def freeze_gate(summary,mean_cagr,mean_rag,exact,technical_passed,partition):
    if not math.isfinite(mean_cagr+mean_rag) or min(mean_cagr,mean_rag)<=0:raise ValueError('Positive measured latencies required')
    normalized=mean_cagr/mean_rag*V2_RAG_MS
    floor=all(all(summary['quality_mean_floor'][ds]['cagr'].values()) for ds in summary['methods'])
    overhead=all(summary['overhead'][ds]['cagr']['full_components']['upper_one_sided95']<.005 and
        summary['overhead'][ds]['paired_e2e']['upper_one_sided95']<.005 for ds in summary['methods'])
    latency=mean_cagr<580 or (normalized<580 and mean_cagr<mean_rag)
    return {'unique_main':'cagr','exact_880_pairs_1058_decisions':exact,'technical_double_drop_passed':technical_passed,
        'per_dataset_EM_F1_mean_floor':floor,'per_dataset_full_and_independent_on_off_overhead_upper_below_0_5_percent':overhead,
        'actual_cagr_ms':mean_cagr,'same_batch_rag_ms':mean_rag,'v2_rag_anchor_ms':V2_RAG_MS,
        'normalized_to_v2_ms':normalized,'absolute_below_580':mean_cagr<580,
        'normalized_below_580_and_measured_below_same_batch_rag':normalized<580 and mean_cagr<mean_rag,
        'latency_passed':latency,'method_frozen':partition=='calibration' and exact and technical_passed and floor and overhead and latency,
        'held_out_test_unsealed':False}
