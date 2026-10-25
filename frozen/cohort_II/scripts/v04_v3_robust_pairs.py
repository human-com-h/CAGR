"""Balanced three-way smoke timing with one authoritative full-trace log."""
import time
import numpy as np
import run_online_v04_v3 as v3
from v04_v3_benchmark import cost
from v04_v3_gates import scientific_view,scientific_sha256
from v04_v3_clockfix_pairs import check_same_path
from v04_v3_robust_benchmark import ORDERS

SMOKE_REPEATS=120;CALIBRATION_REPEATS=12
FROZEN_ANALYZE=v3.analyze


def totals(row):
    calls=row['generation_calls'];retrieval=row['retrieval_trace']
    return {'generation_calls':len(calls),'total_input_tokens':sum(c['input_tokens'] for c in calls),'total_output_tokens':sum(c['output_tokens'] for c in calls),
        'generation_ms':sum(c['ms'] for c in calls),'retrieval_calls':len(retrieval),'retrieval_ms':sum(t['ms'] for t in retrieval),
        'failed_generation_calls':sum(c['status']=='failed' for c in calls)}


def observation(row):return {'execution_id':row['technical_cost_id'],'cost':cost(row),'totals':totals(row),'scientific_sha256':scientific_sha256(row)}


def summarize(records,count):
    if len(records)!=count or [r['repeat'] for r in records]!=list(range(count)):raise RuntimeError('Fixed balanced timing coverage incomplete')
    ratios=[]
    for r in records:
        a,b=r['observations']['on']['cost']['end_to_end_ms'],r['observations']['off']['cost']['end_to_end_ms']
        if b<=0:raise RuntimeError('Invalid independent on/off denominator')
        ratios.append((a-b)/b)
    return float(np.mean(ratios))


def paired_path(sample,g,retriever,config,study,budget,router,index):
    base=v3.base;previous=v3.SCOPE;v3.SCOPE='technical_pair';records=[];first={}
    count=SMOKE_REPEATS if v3.PARTITION=='smoke' else CALIBRATION_REPEATS
    def execute(mode,repeat,plan=None):
        if time.monotonic()>g.robust_process_deadline:raise RuntimeError('Frozen remote budget exhausted in fixed timing blocks')
        generator=v3.LEGACY if mode=='C' else g;enabled=mode!='off';generator.calls=[];generator.signal_collection_enabled=enabled
        execution_id=f'{v3.PARTITION}:{sample["sample_id"]}:{mode}:{repeat}';row=None;start=time.perf_counter()
        try:
            row=v3.execute('cagr',sample['question'],generator,retriever,config,budget,router,study=study,technical_plan=plan if not enabled else None)
            row['technical_cost_id']=execution_id;base.validate_row(row,budget);return row
        finally:
            generator.signal_collection_enabled=True
            # No deepcopy and no duplicate call array: one complete row owns
            # one execution. Compact repeat records reference this ID.
            base.append(base.ROOT/'logs/v3_balanced_executions.jsonl',{'execution_id':execution_id,'partition':v3.PARTITION,'sample_id':sample['sample_id'],
                'mode':mode,'repeat':repeat,'scope':'one paid execution, authoritative full trace','row':row,
                'generation_calls_if_no_row':generator.calls if row is None else None,'attempt_wall_ms':1000*(time.perf_counter()-start),
                'incomplete_cost':'unknown_not_zero' if row is None else None})
    try:
        normal=execute('selection',None);normal_sha=scientific_sha256(normal);plan=[False,False]
        for t in normal['control_trace']:
            if t['kind']=='decision_value' and t['stage'] in [0,1]:plan[t['stage']]=t['selected']
        for repeat in range(count):
            if v3.PARTITION=='smoke':order=[{'A':'on','B':'off','C':'C'}[m] for m in ORDERS[(repeat+index)%6]]
            else:order=['on','off'] if (repeat+index)%2==0 else ['off','on']
            group={}
            for mode in order:
                group[mode]=execute(mode,repeat,plan)
                if mode!='off' and scientific_sha256(group[mode])!=normal_sha:raise RuntimeError('Balanced on/C full-head prediction or decision changed')
            check_same_path(group['on'],group['off'])
            record={'repeat':repeat,'order':order,'observations':{m:observation(r) for m,r in group.items()}};records.append(record)
            if repeat==0:first=group
        ratio=summarize(records,count)
        if v3.PARTITION=='smoke':
            def average(mode):return {k:float(np.mean([r['observations'][mode]['cost'][k] for r in records])) for k in ['signal_ms','generation_ms','end_to_end_ms']}
            comparison={'sample_id':sample['sample_id'],'dataset':sample['dataset'],'candidate':average('on'),'v2':average('C'),
                'repetitions':count,'all_balanced_orders':True,'scope':'question means across ALL 120 paired C/on/off blocks, bootstrap questions only'}
            v3.COMPARISONS.append(comparison);base.append(base.ROOT/'logs/v3_technical_double_drop.jsonl',comparison)
        return {'sample_id':sample['sample_id'],'dataset':sample['dataset'],'kind':'full_path_feature_pair','scope':'technical_not_policy','status':'ok',
            'selected_plan':plan,'order':records[0]['order'],'pairs':{'True':first['on'],'False':first['off']},'selection_reference':normal,
            'paired_ratio':ratio,'repetitions':count,'sampling_unit':'question mean of all fixed paired ratios','balanced_observations':records,
            'authoritative_execution_log':'logs/v3_balanced_executions.jsonl'}
    finally:v3.SCOPE=previous


def analyze(scored,technical,methods):
    report=FROZEN_ANALYZE(scored,technical,methods);extra={};comparators={}
    def add(target,t):
        for k,v in t.items():target[k]=target.get(k,0)+v
    for row in technical:
        if row.get('kind')=='full_path_feature_pair':
            expected=SMOKE_REPEATS if v3.PARTITION=='smoke' else CALIBRATION_REPEATS
            if row['repetitions']!=expected or row['paired_ratio']!=summarize(row['balanced_observations'],expected):raise RuntimeError('Balanced question timing evidence differs')
            for repeat in row['balanced_observations']:
                for mode,r in repeat['observations'].items():
                    if mode=='C':add(comparators,r['totals'])
                    elif repeat['repeat']>0:add(extra,r['totals'])
    report['full_cost_totals']['additional_balanced_on_off']=extra
    report['full_cost_totals']['all_balanced_C_comparators']=comparators
    report['full_cost_totals']['balanced_ownership']='normal/forced/cache/selection/repeat0 on/off in original totals; other on/off only additional_balanced; ALL C only all_balanced_C_comparators. Compact records duplicate authoritative execution IDs, never additional calls.'
    report['independent_on_off_design']={'fixed_smoke_triplets_per_question':SMOKE_REPEATS,'fixed_calibration_pairs_per_question':CALIBRATION_REPEATS,
        'sampling_unit':'question','bootstrap_replicates':20000,'seed':42,'gate_upper_ratio':.005,'observations_removed':0,'optional_stopping':False}
    return report
