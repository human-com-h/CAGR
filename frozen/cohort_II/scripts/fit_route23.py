"""Cal200-only finite four-action crossfit, strict missing values and no old gate."""
from pathlib import Path
import copy,json,time
import numpy as np
from fit_stage0 import fit,predict
from route23_runtime import stage0,stage1
from v03_assets import write
from route23_budget import train_budget_conditions,replay_online_conditions
S0=['mean_logit_gap','gap_uncertainty_of_mean','query_term_count','query_capitalized_count','targ_mean_exp_gap','exp_gap_variance']

def train(x,y,folds,names,alphas):
    errors=[]
    for alpha in alphas:
        loss=[]
        for f in sorted(set(folds)):
            tr=folds!=f;loss.extend((predict(fit(x[tr],y[tr],alpha),x[~tr])-y[~tr])**2)
        errors.append(float(np.mean(loss)))
    return {**fit(x,y,alphas[int(np.argmin(errors))]),'feature_names':names}

def evaluate(rr,acts,baselines):
    result={}
    for ds in ['hotpotqa','triviaqa','overall']:
        indices=[i for i,r in enumerate(rr) if ds=='overall' or r['dataset']==ds];quality={k:float(np.mean([rr[i]['actions'][acts[i]]['quality'][k] for i in indices])) for k in ['em','f1']}
        result[ds]={**quality,'n':len(indices),'counts':{a:sum(acts[i]==a for i in indices) for a in ['direct','stop','in_situ','kv_bridge','rewrite']},
            'latency_proxy_ms':float(np.mean([rr[i]['actions'][acts[i]]['latency_proxy_ms'] for i in indices])),
            **{k+'_minus_RAG':quality[k]-float(np.mean([rr[i]['actions']['stop']['quality'][k] for i in indices])) for k in quality},
            **{k+'_minus_TARG':quality[k]-baselines[ds][k] for k in quality}}
        result[ds]['latency_minus_TARG_ms']=result[ds]['latency_proxy_ms']-baselines[ds]['latency_proxy_ms']
    return result

def fit_candidates(rr,cfg,baseline_config,out):
    start=time.perf_counter();out=Path(out);out.mkdir(parents=True,exist_ok=False);assert len(rr)==200 and len({r['sample_id'] for r in rr})==200
    c=cfg['selection'];folds=np.array([r['fold'] for r in rr]);x=np.array([[r['stage0_features'][k] for k in S0] for r in rr]);names=c['stage1_features'];ax=np.array([[r['stage1_features'][k] for k in names] for r in rr]);alpha=c['ridge_alpha_grid']
    assert all(r['actions'][a]['quality'] is not None and r['actions'][a]['latency_proxy_ms'] is not None for r in rr for a in ['direct','stop','in_situ','kv_bridge','rewrite'])
    y=np.array([r['actions']['stop']['quality']['f1']-r['actions']['direct']['quality']['f1'] for r in rr]);cost=np.array([r['actions']['stop']['latency_proxy_ms']-r['actions']['direct']['latency_proxy_ms'] for r in rr]);targets={'gain':y,'positive_gain':np.maximum(y,0),'cost':cost};heads={};scores={};inner={}
    for scope in ['full',*map(str,range(5))]:
        mask=np.ones(200,dtype=bool) if scope=='full' else folds!=int(scope);h={k:train(x[mask],v[mask],folds[mask],S0,alpha) for k,v in targets.items()};h['action_heads']={}
        for a in ['in_situ','kv_bridge','rewrite']:
            for kind in ['gain','cost']:
                t=np.array([r['actions'][a]['quality']['f1']-r['actions']['stop']['quality']['f1'] if kind=='gain' else r['actions'][a]['latency_proxy_ms']-r['actions']['stop']['latency_proxy_ms'] for r in rr]);h['action_heads'][a+'_'+kind]=train(ax[mask],t[mask],folds[mask],names,alpha)
        heads[scope]=h
        train_predictions={k:np.empty(int(mask.sum())) for k in targets};tf=folds[mask]
        for f in sorted(set(tf)):
            it=tf!=f;va=~it
            for k,t in targets.items():train_predictions[k][va]=predict(train(x[mask][it],t[mask][it],tf[it],S0,alpha),x[mask][va])
        inner[scope]=train_predictions
    # Root threshold uses ordinary outer-OOF predictions, matching prior deployment convention.
    z={k:np.empty(200) for k in targets}
    for f in range(5):
        for k in targets:z[k][folds==f]=predict(heads[str(f)][k],x[folds==f])
    inner['full']=z
    baselines={}
    for scope in ['OOF','full']:
        picked=[];times=[]
        for r in rr:
            t=baseline_config['targ_style'];threshold=t['threshold'] if scope=='full' else t['calibration_fold_parameters'][str(r['fold'])]['threshold'];a='stop' if r['stage0_features']['targ_mean_exp_gap']>threshold else 'direct';picked.append(r['actions'][a]['quality']);times.append(r['TARG_actual_ms'] if scope=='OOF' else r['TARG_arm_proxy_ms'][a])
        baselines[scope]={ds:{**{k:float(np.mean([v[k] for r,v in zip(rr,picked) if ds=='overall' or r['dataset']==ds])) for k in ['em','f1']},'latency_proxy_ms':float(np.mean([v for r,v in zip(rr,times) if ds=='overall' or r['dataset']==ds]))} for ds in ['hotpotqa','triviaqa','overall']}
    grid=[];candidates={};i=0
    for weight in c['stage0_positive_weights']:
      for lam0 in c['stage0_latency_penalties']:
       for rate in c['stage0_retrieval_rates']:
        bundle=copy.deepcopy(heads)
        for scope,h in bundle.items():
            p=inner[scope];v=p['gain']+weight*np.maximum(p['positive_gain'],0)-lam0*np.maximum(p['cost'],0);h['threshold0']=float(np.quantile(v,1-rate)) if rate<1 else -1e30
        for lam1 in c['stage1_latency_penalties']:
         for offset in c['action_thresholds']:
            i+=1;candidate={'candidate_id':'route23_'+str(i),'configuration':{'positive_weight':weight,'lambda0':lam0,'training_retrieval_rate':rate,'lambda1':lam1,'action_threshold':offset},
                'deployment':bundle['full'],'OOF_heads':{str(k):bundle[str(k)] for k in range(5)},'old_reference_gate':False,'method_frozen':False};reports={};actions={}
            for scope in ['OOF','full']:
                acts=[]
                for r in rr:
                    fold=r['fold'] if scope=='OOF' else None;first=stage0(candidate,r['stage0_features'],fold)[0];acts.append(stage1(candidate,r['stage1_features'],fold)[0] if first else 'direct')
                reports[scope]=evaluate(rr,acts,baselines[scope]);actions[scope]=acts
            floors=all(reports[s][ds][k+'_minus_RAG']>=-.01-1e-12 for s in reports for ds in ['hotpotqa','triviaqa'] for k in ['em','f1'])
            quality=floors and all(reports[s][ds][k+'_minus_TARG']>=-1e-12 for s in reports for ds in ['hotpotqa','triviaqa'] for k in ['em','f1'])
            faster=all(reports[s]['overall']['latency_minus_TARG_ms']<0 for s in reports)
            upgraded=[sum(reports[s]['overall']['counts'][a] for a in ['in_situ','kv_bridge','rewrite']) for s in reports];natural=all(c['natural_upgrades_min']<=n<=c['natural_upgrades_max'] for n in upgraded)
            rec={'candidate_id':candidate['candidate_id'],**candidate['configuration'],'RAG_floors':floors,'TARG_all_quality':quality,'faster_proxy':faster,'natural_upgrade_range':natural,'reports':reports};grid.append(rec);candidates[candidate['candidate_id']]=(copy.deepcopy(candidate),actions)
    joint=[r for r in grid if r['TARG_all_quality'] and r['faster_proxy'] and r['natural_upgrade_range']]
    pool=joint or [r for r in grid if r['RAG_floors'] and r['natural_upgrade_range']] or grid;chosen=min(pool,key=lambda r:r['reports']['OOF']['overall']['latency_proxy_ms']);candidate,acts=candidates[chosen['candidate_id']]
    candidate.update(reports=chosen['reports'],development_joint_proxy_goal_passed=bool(joint),selected_only_on_Cal200=True,quality_uses_frozen_same_prompt_real_answers=True,
        OOF_is_configuration_selected_exploration=True,final_GPU_policy_validation_required=True)
    candidate['budget_conditions']=train_budget_conditions(rr,candidate,inner,folds,cfg['route3']['budget_anchors'])
    write(out/'online_budget_conditions.json',replay_online_conditions(rr,candidate));write(out/'candidate.json',candidate);write(out/'grid.json',grid)
    report={'selected':chosen,'joint_proxy_candidates':len(joint),'total':len(grid),'baselines':baselines,'old_test_or_fresh_validation_used':False,'method_frozen':False,'CPU_wall_ms':1000*(time.perf_counter()-start)};write(out/'selection_report.json',report)
    write(out/'selected_actions.json',{'sample_ids':[r['sample_id'] for r in rr],'actions':acts});return candidate,report
