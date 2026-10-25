"""Automatic calibration-only feature fallback and pilot policy replay.

Replay latency estimates are development diagnostics, not measured CAGR latency.
"""
from pathlib import Path
import argparse,datetime,json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
ALPHAS=[0.1,1.0,10.0]
LAMBDAS=[0.0,1e-7,3e-7,1e-6,3e-6,1e-5,3e-5,1e-4,3e-4]

def fit(x,y,alpha):
    mean=x.mean(0)
    scale=x.std(0)
    scale=np.where(scale>1e-12,scale,1.0)
    design=np.column_stack([np.ones(len(x)),(x-mean)/scale])
    penalty=np.diag([0.0]+[alpha]*x.shape[1])
    coefficients=np.linalg.solve(design.T@design+penalty,design.T@y)
    return {"mean":mean.tolist(),"scale":scale.tolist(),"coefficients":coefficients.tolist(),"alpha":alpha}

def predict(model,x):
    z=(x-np.array(model["mean"]))/np.array(model["scale"])
    return np.column_stack([np.ones(len(x)),z])@np.array(model["coefficients"])

def choose_alpha(x,y,folds):
    errors=[]
    for alpha in ALPHAS:
        error=[]
        for fold in sorted(set(folds)):
            valid=folds==fold
            training=~valid
            model=fit(x[training],y[training],alpha)
            error.extend((predict(model,x[valid])-y[valid])**2)
        errors.append(float(np.mean(error)))
    return ALPHAS[int(np.argmin(errors))]

def oof(x,y,folds):
    prediction=np.empty(len(x))
    selected=[]
    for fold in sorted(set(folds)):
        valid=folds==fold
        training=~valid
        alpha=choose_alpha(x[training],y[training],folds[training])
        prediction[valid]=predict(fit(x[training],y[training],alpha),x[valid])
        selected.append(alpha)
    final_alpha=choose_alpha(x,y,folds)
    return prediction,{**fit(x,y,final_alpha),"outer_fold_alphas":selected}

def auc(labels,scores):
    positive=scores[labels]
    negative=scores[~labels]
    if not len(positive) or not len(negative):
        return None
    comparison=positive[:,None]-negative[None,:]
    return float(np.mean((comparison>0)+0.5*(comparison==0)))

def auc_interval(labels,scores,datasets,seed=42):
    point=auc(labels,scores)
    if point is None:
        return {"auc":None,"ci95":None,"reason":"missing_benefit_or_nonbenefit_class"}
    groups=[np.flatnonzero((datasets==dataset)&(labels==label)) for dataset in sorted(set(datasets)) for label in [False,True]]
    groups=[g for g in groups if len(g)]
    rng=np.random.default_rng(seed)
    values=[]
    for _ in range(2000):
        indices=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups])
        values.append(auc(labels[indices],scores[indices]))
    return {"auc":point,"ci95":np.quantile(values,[0.025,0.975]).tolist(),"bootstrap":"stratified by dataset and benefit label","replicates":2000,"seed":seed}

def best_policy(q,c,gain01,cost01,gain12,cost12):
    target=q[:,1,:].mean(0)-0.01
    best=None
    for lam in LAMBDAS:
        score0=gain01-lam*cost01
        score1=gain12-lam*cost12 if gain12 is not None else None
        thresholds0=[-float("inf"),*np.quantile(score0,[0,.1,.25,.5,.75,.9,1]),float("inf")]
        thresholds1=[-float("inf"),*np.quantile(score1,[0,.1,.25,.5,.75,.9,1]),float("inf")] if score1 is not None else [float("inf")]
        for t0 in thresholds0:
            retrieve=score0>t0
            for t1 in thresholds1:
                escalate=(score1>t1)&retrieve if score1 is not None else np.zeros(len(q),dtype=bool)
                tier=retrieve.astype(int)+escalate.astype(int)
                quality=q[np.arange(len(q)),tier].mean(0)
                latency=float(c[np.arange(len(c)),tier].mean())
                if np.all(quality>=target) and (best is None or latency<best["estimated_mean_ms"]):
                    def finite_threshold(t):
                        return float(t) if np.isfinite(t) else "always" if t<0 else "never"
                    best={"lambda":lam,"threshold01":finite_threshold(t0),"threshold12":finite_threshold(t1),"em":float(quality[0]),"f1":float(quality[1]),"estimated_mean_ms":latency,"tier_counts":np.bincount(tier,minlength=3).tolist(),"scope":"calibration OOF replay; requires actual online validation"}
    return best

def best_policy_exact(q,c,gain01,cost01,gain12,cost12):
    """Search all distinct score partitions; do not split equal-score samples."""
    n=len(q)
    target=q[:,1,:].mean(0)-0.01
    total_q=q[:,0,:].sum(0)
    total_c=c[:,0].sum()
    delta_q=q[:,1,:]-q[:,0,:]
    delta_c=c[:,1]-c[:,0]
    best=None
    def cuts(values):
        return np.r_[0,np.flatnonzero(values[:-1]!=values[1:])+1,len(values)].astype(int) if len(values) else np.array([0])
    def threshold(values,k):
        if k==0:
            return "never"
        if k==len(values):
            return "always"
        return float(values[k]+(values[k-1]-values[k])/2)
    for lam in LAMBDAS:
        s0=gain01-lam*cost01
        order0=np.argsort(-s0,kind="stable")
        sorted0=s0[order0]
        cumulative_q=np.vstack([np.zeros(2),np.cumsum(delta_q[order0],axis=0)])
        cumulative_c=np.r_[0,np.cumsum(delta_c[order0])]
        s1=gain12-lam*cost12 if gain12 is not None else None
        for k in cuts(sorted0):
            retrieved=order0[:k]
            base_q=total_q+cumulative_q[k]
            base_c=total_c+cumulative_c[k]
            if s1 is None or k==0:
                sorted1=np.array([])
                eligible=np.array([],dtype=int)
                qpath=base_q[None,:]
                cpath=np.array([base_c])
                ecounts=np.array([0])
            else:
                eligible=retrieved[np.argsort(-s1[retrieved],kind="stable")]
                sorted1=s1[eligible]
                qpath=base_q+np.vstack([np.zeros(2),np.cumsum(q[eligible,2,:]-q[eligible,1,:],axis=0)])
                cpath=base_c+np.r_[0,np.cumsum(c[eligible,2]-c[eligible,1])]
                ecounts=cuts(sorted1)
            feasible=np.all(qpath[ecounts]/n>=target,axis=1)
            if not feasible.any():
                continue
            allowed=ecounts[feasible]
            e=int(allowed[np.argmin(cpath[allowed])])
            latency=float(cpath[e]/n)
            if best is None or latency<best["estimated_mean_ms"]:
                quality=qpath[e]/n
                best={"lambda":lam,"threshold01":threshold(sorted0,int(k)),"threshold12":threshold(sorted1,e),"em":float(quality[0]),"f1":float(quality[1]),"estimated_mean_ms":latency,"tier_counts":[n-int(k),int(k)-e,e],"scope":"calibration OOF replay; requires actual online validation","threshold_search":"all_distinct_score_partitions"}
    return best

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--results-root",default="results")
    parser.add_argument("--threshold-search",choices=["quantile","exact"],default="quantile")
    parser.add_argument("--upgrade-arm",choices=["titles_control","evidence_always","evidence_novelty_stop"])
    args=parser.parse_args()
    results_root=(ROOT/args.results_root).resolve()
    assert results_root.is_relative_to(ROOT/"results")
    rows=[json.loads(s) for s in (results_root/"calibration/scored.jsonl").read_text(encoding="utf-8").splitlines()]
    expected=[json.loads(s)["sample_id"] for s in (ROOT/"data/splits/calibration.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows)==len(expected)==200
    assert {r["sample_id"] for r in rows}==set(expected)
    assert all(r["status"]=="ok" for r in rows),"Failures must be resolved or protocol revised before fitting"
    folds=np.array([r["fold"] for r in rows])
    datasets=np.array([r["dataset"] for r in rows])
    x0=np.array([[r["baselines"]["direct"]["features"][k] for k in ["prefix_entropy","probability_margin"]] for r in rows])
    x1=np.array([[r["baselines"]["rag1"]["features"][k] for k in ["prefix_entropy","retrieval_score_variance","answer_length_tokens"]] for r in rows])
    y01=np.array([r["delta01_f1"] for r in rows])
    upgrades=[r["variants"][args.upgrade_arm] if args.upgrade_arm else r["baselines"]["multihop2"] for r in rows]
    paths=[[r["baselines"]["direct"],r["baselines"]["rag1"],u] for r,u in zip(rows,upgrades)]
    y12=np.array([u["quality"]["f1"]-r["baselines"]["rag1"]["quality"]["f1"] for r,u in zip(rows,upgrades)])
    q=np.array([[[b["quality"][k] for k in ["em","f1"]] for b in path] for path in paths])
    c=np.array([[b["cost"]["end_to_end_ms"] for b in path] for path in paths])
    # Route-to-retrieval paths pay the initial prefix, unlike Always-RAG.
    prefix=np.array([r["baselines"]["direct"]["generation_calls"][0]["pilot_prefix_ms"] for r in rows])
    c[:,1:]+=prefix[:,None]
    assert np.isfinite(x0).all() and np.isfinite(x1).all() and np.isfinite(c).all()
    g0,head0=oof(x0,y01,folds)
    k0,cost0=oof(x0,c[:,1]-c[:,0],folds)
    gh,headh=oof(x1[:,:1],y12,folds)
    kh,costh=oof(x1[:,:1],c[:,2]-c[:,1],folds)
    gc,headc=oof(x1,y12,folds)
    kc,costc=oof(x1,c[:,2]-c[:,1],folds)
    entropy_auc=auc_interval(y12>1e-12,gh,datasets)
    combo_auc=auc_interval(y12>1e-12,gc,datasets)
    search=best_policy_exact if args.threshold_search=="exact" else best_policy
    binary=search(q,c,g0,k0,None,None)
    entropy=search(q,c,g0,k0,gh,kh)
    combo=search(q,c,g0,k0,gc,kc)
    reasons=[]
    if entropy_auc["ci95"] is None or entropy_auc["ci95"][0]<=0.5:
        reasons.append("entropy_upgrade_predictiveness_not_significant")
    if entropy is None or binary is None or entropy["estimated_mean_ms"]>=binary["estimated_mean_ms"]:
        reasons.append("entropy_no_feasible_cost_improvement_over_fixed_binary_replay")
    use_combo=bool(reasons)
    date=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).date().isoformat()
    result={"date":date,"results_root":str(results_root.relative_to(ROOT)),"method":"CAGR","scope":"pilot calibration only","auto_fallback_selected":use_combo,"trigger_reasons":reasons,"entropy_oof_auc":entropy_auc,"combination_oof_auc":combo_auc,"fixed_binary_internal_ablation":binary,"entropy_policy_replay":entropy,"combination_policy_replay":combo,"selected_stage1_features":["prefix_entropy","retrieval_score_variance","answer_length_tokens"] if use_combo else ["prefix_entropy"],"stage0_gain_head":head0,"stage0_cost_head":cost0,"stage1_gain_head":headc if use_combo else headh,"stage1_cost_head":costc if use_combo else costh,"selected_replay_policy":combo if use_combo else entropy,"ready_for_held_out_test":False,"measured_online_cagr_advantage":False,"next_validation":"actual online routing, strong TARG/SUGAR-L comparisons and independent full-corpus protocol"}
    result["threshold_search"]=args.threshold_search
    result["upgrade_arm"]=args.upgrade_arm or "configured_development_candidate"
    filename="router_development_exact.json" if args.threshold_search=="exact" else "router_development.json"
    if args.upgrade_arm:
        filename=filename[:-5]+"_"+args.upgrade_arm+".json"
    (results_root/"calibration"/filename).write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"auto_fallback_selected":use_combo,"trigger_reasons":reasons,"ready_for_held_out_test":False}),flush=True)

if __name__=="__main__":
    main()
