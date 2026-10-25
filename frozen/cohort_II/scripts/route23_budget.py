"""Causal budget-conditioned online rules, calibrated on training folds only."""
import copy
import numpy as np
from route23_runtime import stage0,stage1

def train_budget_conditions(rr,candidate,inner_predictions,folds,anchors):
    conditions={};c=candidate['configuration']
    for b in anchors:
      for kind in ['first_retrieval','total_retrieval']:
        key=kind+':'+str(b);rule={'B':b,'budget_kind':kind,'lambda1':c['lambda1']/b,'action_threshold':c['action_threshold']/b,'thresholds':{}}
        adjusted=copy.deepcopy(candidate);adjusted['configuration'].update(lambda1=rule['lambda1'],action_threshold=rule['action_threshold'])
        for scope,p in inner_predictions.items():
            selected=rr if scope=='full' else [r for r in rr if r['fold']!=int(scope)];scores=p['gain']+c['positive_weight']*np.maximum(p['positive_gain'],0)-c['lambda0']*np.maximum(p['cost'],0)
            if kind=='first_retrieval':threshold=float(np.quantile(scores,1-b)) if b<1 else -1e30
            else:
                order=np.argsort(-scores,kind='stable');used=0;n=0
                for i in order:
                    act=stage1(adjusted,selected[int(i)]['stage1_features'],None if scope=='full' else int(scope))[0];increment=1+int(act!='stop')
                    if used+increment>b*len(selected)+1e-12:break
                    used+=increment;n+=1
                threshold=1e30 if not n else -1e30 if n==len(order) else float((scores[order[n-1]]+scores[order[n]])/2.)
            rule['thresholds'][scope]=threshold
        conditions[key]=rule
    return conditions

def condition(candidate,B,kind='total_retrieval'):
    key=kind+':'+str(B)
    if key not in candidate['budget_conditions']:raise ValueError('Unregistered B anchor')
    rule=candidate['budget_conditions'][key];out=copy.deepcopy(candidate);out['configuration'].update(lambda1=rule['lambda1'],action_threshold=rule['action_threshold'])
    out['deployment']['threshold0']=rule['thresholds']['full']
    for fold,h in out['OOF_heads'].items():h['threshold0']=rule['thresholds'][fold]
    return out

def replay_online_conditions(rr,candidate):
    result=[]
    for key,p in candidate['budget_conditions'].items():
        adjusted=condition(candidate,p['B'],p['budget_kind']);acts=[]
        for r in rr:acts.append(stage1(adjusted,r['stage1_features'],r['fold'])[0] if stage0(adjusted,r['stage0_features'],r['fold'])[0] else 'direct')
        result.append({'B':p['B'],'budget_kind':p['budget_kind'],'first_retrievals':sum(a!='direct' for a in acts),'total_retrievals':sum(0 if a=='direct' else 1 if a=='stop' else 2 for a in acts),
            'per_dataset':{ds:{k:float(np.mean([r['actions'][a]['quality'][k] for r,a in zip(rr,acts) if r['dataset']==ds])) for k in ['em','f1']} for ds in ['hotpotqa','triviaqa']},
            'scope':'OOF matrix replay of causal online B-conditioned rules; realized test budget may deviate; no test quota fitting'})
    return result
