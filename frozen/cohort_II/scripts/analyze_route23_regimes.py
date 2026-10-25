"""Paid-component cost replacement and label-free budget rank diagnostics."""
import math
import numpy as np
from route23_runtime import stage0,stage1
from v03_assets import write

def break_even(own,other):
    # L_other - L_own = fixed difference + call difference*r.
    dn=other['retrievals']-own['retrievals'];dc=other['nonretrieval_ms']-own['nonretrieval_ms']
    return {'retrieval_ms':None if abs(dn)<1e-12 else -dc/dn,'comparator_minus_main_intercept_ms':dc,'comparator_minus_main_retrieval_slope':dn,
        'interpretation':'constant-cost replacement only; not an executed reranker'}

def analyze_regimes(rr,candidate,cfg,out):
    rows=[];features=[]
    for r in rr:
        fold=r['fold'];first,tr=stage0(candidate,r['stage0_features'],fold);act=stage1(candidate,r['stage1_features'],fold)[0] if first else 'direct'
        rows.append((r,act));features.append(tr['score'])
    costs={}
    for method in ['cagr_v2','matched_binary','TARG','always_rag']:
        q=[]
        for r,a in rows:
            act=a if method=='cagr_v2' else ('stop' if a!='direct' else 'direct') if method=='matched_binary' else r['TARG_action'] if method=='TARG' else 'stop'
            value=r['actions'][act]
            if method=='TARG':q.append((r['TARG_actual_ms']-r['TARG_retrieval_ms'],int(act!='direct')))
            elif method=='always_rag':q.append((r['Always_RAG_actual_ms']-r['Always_RAG_retrieval_ms'],1))
            elif method=='matched_binary':q.append((r['TARG_arm_proxy_ms'][act]-(r['Always_RAG_retrieval_ms'] if act=='stop' else 0.),int(act!='direct')))
            else:q.append((value['latency_proxy_ms']-value['retrieval_ms'],value['retrievals']))
        costs[method]={'nonretrieval_ms':float(np.mean([x[0] for x in q])),'retrievals':float(np.mean([x[1] for x in q]))}
    replacement=[]
    for retms in cfg['route3']['retrieval_ms']:
        values={m:v['nonretrieval_ms']+v['retrievals']*retms for m,v in costs.items()};replacement.append({'retrieval_ms':retms,'latency_ms':values,'TARG_over_CAGR_speedup':values['TARG']/values['cagr_v2']})
    # Exact quota matching here is an explicitly OFFLINE batch budget diagnostic.
    # Rank uses OOF observable predictions only; no gold-based oracle ordering.
    order=sorted(range(len(rr)),key=lambda i:(-features[i],rr[i]['sample_id']));targ=sorted(range(len(rr)),key=lambda i:(-rr[i]['stage0_features']['targ_mean_exp_gap'],rr[i]['sample_id']));curve=[]
    for b in cfg['route3']['budget_anchors']:
      budget=round(b*len(rr))
      for scope in ['first_retrieval','total_retrieval']:
        first_count=budget if scope=='first_retrieval' else round(budget/1.2);selected=set(order[:first_count]);acts=['direct']*len(rr);ups=[]
        for i in selected:
            action,trace=stage1(candidate,rr[i]['stage1_features'],rr[i]['fold']);acts[i]='stop'
            if action!='stop':ups.append((trace['utilities'][action],i,action))
        for _,i,a in sorted(ups,key=lambda x:(-x[0],rr[x[1]]['sample_id']))[:len(ups) if scope=='first_retrieval' else max(0,budget-first_count)]:acts[i]=a
        main_calls=sum(0 if a=='direct' else 1 if a=='stop' else 2 for a in acts);target=first_count if scope=='first_retrieval' else main_calls;tg=set(targ[:target]);binaryset=set(order[:target])
        row={'B':b,'scope':scope,'CAGR_first_calls':first_count,'CAGR_total_calls':main_calls,'TARG_calls':target,'Binary_calls':target,'quality':{},
            'analysis_scope':'offline OOF-prediction budget rank; batch lookahead, not deployed online threshold performance'}
        for ds in ['hotpotqa','triviaqa']:
            ids=[i for i,r in enumerate(rr) if r['dataset']==ds];row['quality'][ds]={}
            for name,aa in [('cagr',acts),('matched_binary_first',['stop' if i in selected else 'direct' for i in range(len(rr))]),('binary_matched_scope',['stop' if i in binaryset else 'direct' for i in range(len(rr))]),('TARG_matched_scope',['stop' if i in tg else 'direct' for i in range(len(rr))])]:
                row['quality'][ds][name]={k:float(np.mean([rr[i]['actions'][aa[i]]['quality'][k] for i in ids])) for k in ['em','f1']}
        curve.append(row)
    write(out/'retrieval_regimes.json',{'kind':'counterfactual deterministic per-call retrieval cost replacement','actual_BM25_recorded_separately':True,'assumes_identical_documents_not_valid_for_real_new_reranker':True,'component_means':costs,'costs':replacement,'break_even_TARG':break_even(costs['cagr_v2'],costs['TARG'])})
    write(out/'budget_spectrum.json',{'seed':42,'finite_anchors_and_stepwise_extension':True,'unbiased_independent_test':False,'gold_ordering':False,'curves':curve})
