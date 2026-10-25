"""Reconstruct selected ridge heads on development data only; no test tuning."""
from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'frozen/cohort_II/scripts'))
from fit_stage0 import fit
from v04_features import predict_linear
from v2_fine_runtime import first_decision,second_decision
def read(p):return json.loads(Path(p).read_text())
def lines(n):return [json.loads(l) for l in (ROOT/'evidence/calibration'/n).read_text().splitlines() if l]
def compare(actual,expected):
    for k in ['mean','scale','coefficients']:
        np.testing.assert_allclose(actual[k],expected[k],rtol=1e-11,atol=1e-11)
def main():
    r=read(ROOT/'frozen/cohort_I/configs/deployment_test1000.json')['router'];rr=lines('cagr_r_targets.jsonl');count=0
    for stage in [0,1]:
        for target in ['gain','cost']:
            h=r[f'stage{stage}'][target];names=h['feature_names'];x=np.array([[row[f'stage{stage}_features'][k] for k in names] for row in rr]);y=np.array([row['targets'][f'stage{stage}_{target}'] for row in rr]);compare(fit(x,y,h['alpha']),h);count+=1
    fine=read(ROOT/'frozen/cohort_II/configs/heldout_v2_deployment.json')['fine'];fine['fold_heads']=read(ROOT/'evidence/calibration/fine_fold_heads.json');rr=lines('cagr_f_targets.jsonl');folds=np.array([r['fold'] for r in rr]);rebuilt={}
    for scope in ['full',*map(str,range(5))]:
        mask=np.ones(200,dtype=bool) if scope=='full' else folds!=int(scope)
        locked=fine['deployment'] if scope=='full' else fine['fold_heads'][scope];heads={}
        for action in ['in_situ','kv_bridge','rewrite']:
            h=locked['action_heads'][action];x=np.array([[r['stage1_features'][k] for k in h['feature_names']] for r in rr]);y=np.array([r['gain_targets'][action] for r in rr]);head={**fit(x[mask],y[mask],h['alpha']),'feature_names':h['feature_names']};compare(head,h);count+=1;heads[action]=head
        t0=float(np.quantile([r['stage0_features']['targ_mean_exp_gap'] for r,keep in zip(rr,mask) if keep],.5))
        confidence=float(np.quantile([r['stage1_features']['mean_logit_gap'] for r,keep in zip(rr,mask) if keep and r['dataset']=='triviaqa'],.75))
        values=[]
        for row,keep in zip(rr,mask):
            if not keep or row['stage0_features']['targ_mean_exp_gap']<=t0:continue
            if row['dataset']=='triviaqa' and row['stage1_features']['mean_logit_gap']>=confidence:continue
            values.append(max(predict_linear(h,row['stage1_features']) for h in heads.values()))
        t1=max(0.,float(np.quantile(values,.95)))
        np.testing.assert_allclose([t0,confidence,t1],[locked['threshold0'],locked['trivia_confidence_threshold'],locked['threshold1']],rtol=1e-11,atol=1e-11)
        rebuilt[scope]={'threshold0':t0,'threshold1':t1,'trivia_confidence_threshold':confidence,'action_heads':heads}
    # Replay 400 OOF/Full decisions against the historical development trace.
    candidate={**fine,'deployment':rebuilt['full'],'fold_heads':{k:v for k,v in rebuilt.items() if k!='full'}}
    trace=read(ROOT/'evidence/calibration/fine_decision_reference.json')['traces']
    for scope in ['OOF','full']:
        for i,row in enumerate(rr):
            fold=row['fold'] if scope=='OOF' else None;first=first_decision(candidate,row['stage0_features'],fold)
            action=second_decision(candidate,row['stage1_features'],row['dataset'],fold) if first else 'direct'
            assert first==trace[scope]['first'][i] and action==trace[scope]['attempted'][i]
    result={'status':'passed','selected_ridge_heads_rebuilt':count,'CAGR-F_threshold_sets_rebuilt':6,'development_decisions_replayed':400,'test_rows_used':0,'new_parameter_search':False,'numeric_tolerance':1e-11}
    target=ROOT/'reproduced';target.mkdir(exist_ok=True);(target/'parameter_rebuild.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
