"""Fixed balanced full-path timing repeats, clustered at the question level.

Twelve on/off pairs per question, never repeated until a desired result.
All executions and failures have unique cost IDs and remain in raw logs.
"""
import copy,time
import numpy as np
import run_online_v04_v3 as v3
from v03_assets import write
from v04_v3_gates import scientific_view
from v04_v3_benchmark import cost

REPEATS=12
original_analyze=v3.analyze


def pair_order(index,repeat):
    return [True,False] if (index+repeat)%2==0 else [False,True]


def summarize_repeats(records):
    if len(records)!=REPEATS or [r['repeat'] for r in records]!=list(range(REPEATS)):
        raise RuntimeError('Fixed independent timing repeats incomplete')
    ratios=[]
    for r in records:
        a,b=r['pairs']['True'],r['pairs']['False'];off=b['cost']['end_to_end_ms']
        if off<=0:raise RuntimeError('Invalid independent on/off denominator')
        ratios.append((a['cost']['end_to_end_ms']-off)/off)
    return float(np.mean(ratios))


def check_same_path(a,b):
    if a['answer']!=b['answer'] or a['context_ids']!=b['context_ids'] or \
       [c['output_token_ids'] for c in a['generation_calls']]!=[c['output_token_ids'] for c in b['generation_calls']] or \
       scientific_view(a)['retrieval_trace']!=scientific_view(b)['retrieval_trace']:
        raise RuntimeError('Repeated independent on/off changed frozen path/tokens/retrieval')


def paired_path(sample,g,retriever,config,study,budget,router,index):
    base=v3.base;previous=v3.SCOPE;v3.SCOPE='technical_pair';deadline=time.monotonic()+10800
    # The global process deadline is also checked inside every added run.
    if hasattr(g,'clockfix_process_deadline'):deadline=g.clockfix_process_deadline
    baseline=None;records=[]
    def execute(generator,enabled,repeat,plan=None,legacy=False):
        if time.monotonic()>deadline:raise RuntimeError('Frozen remote wall budget exhausted during paid timing repeats')
        label='C' if legacy else 'selection' if repeat is None else 'features-'+str(enabled)
        execution_id=f'{v3.PARTITION}:{sample["sample_id"]}:{label}:{repeat}'
        generator.calls=[];generator.signal_collection_enabled=enabled;start=time.perf_counter();row=None
        try:
            row=v3.execute('cagr',sample['question'],generator,retriever,config,budget,router,study=study,
                technical_plan=plan if not enabled else None)
            row['technical_cost_id']=execution_id;base.validate_row(row,budget);return row
        finally:
            generator.signal_collection_enabled=True
            base.append(base.ROOT/'logs/v3_repeated_pair_executions.jsonl',{'execution_id':execution_id,'partition':v3.PARTITION,
                'sample_id':sample['sample_id'],'repeat':repeat,'label':label,'scope':'paid technical only; one line one execution',
                'status':row['status'] if row else 'failed','generation_calls':copy.deepcopy(generator.calls),
                'row':row,'attempt_wall_ms':1000*(time.perf_counter()-start),
                'unrecorded_cost_on_failure':'unknown_not_zero' if row is None else None})
    try:
        if v3.PARTITION=='smoke' and index%2:baseline=execute(v3.LEGACY,True,None,legacy=True)
        normal=execute(g,True,None);plan=[False,False]
        for t in normal['control_trace']:
            if t['kind']=='decision_value' and t['stage'] in [0,1]:plan[t['stage']]=t['selected']
        for repeat in range(REPEATS):
            pairs={}
            for enabled in pair_order(index,repeat):
                row=execute(g,enabled,repeat,plan)
                if enabled and scientific_view(row)!=scientific_view(normal):raise RuntimeError('Repeated enabled prediction/decision differs from fixed full-head selection')
                pairs[str(enabled)]=row
            check_same_path(pairs['True'],pairs['False'])
            r={'repeat':repeat,'order':pair_order(index,repeat),'pairs':pairs};records.append(r)
            base.append(base.ROOT/'logs/v3_repeated_pairs.jsonl',{'partition':v3.PARTITION,'sample_id':sample['sample_id'],**r,
                'scope':'duplicate evidence of repeated_pair_executions; do not bill twice'})
            if repeat==0 and v3.PARTITION=='smoke' and baseline is None:baseline=execute(v3.LEGACY,True,None,legacy=True)
        first=records[0]['pairs']
        if v3.PARTITION=='smoke':
            if baseline is None:baseline=execute(v3.LEGACY,True,None,legacy=True)
            if scientific_view(first['True'])!=scientific_view(baseline):raise RuntimeError('Same-T4 C comparator differs')
            comparison={'sample_id':sample['sample_id'],'dataset':sample['dataset'],'candidate':cost(first['True']),'v2':cost(baseline),
                'order':'C-before' if index%2 else 'C-after','baseline_row':baseline,'scope':'one C execution in repeated_pair_executions; no double billing'}
            v3.COMPARISONS.append(comparison);base.append(base.ROOT/'logs/v3_technical_double_drop.jsonl',comparison)
        ratio=summarize_repeats(records)
        return {'sample_id':sample['sample_id'],'dataset':sample['dataset'],'kind':'full_path_feature_pair','scope':'technical_not_policy','status':'ok',
            'selected_plan':plan,'order':records[0]['order'],'pairs':first,'selection_reference':normal,
            'paired_e2e_increment_ms':float(np.mean([r['pairs']['True']['cost']['end_to_end_ms']-r['pairs']['False']['cost']['end_to_end_ms'] for r in records])),
            'paired_ratio':ratio,'repetitions':REPEATS,'sampling_unit':'question; arithmetic mean of all 12 paired ratios, bootstrap questions only',
            'raw_repeats':records,'raw_execution_log':'logs/v3_repeated_pair_executions.jsonl'}
    finally:v3.SCOPE=previous


def analyze(scored,technical,methods):
    report=original_analyze(scored,technical,methods)
    extra=[];retrievals=[]
    for row in technical:
        if row.get('kind')=='full_path_feature_pair':
            if row.get('repetitions')!=REPEATS or row['paired_ratio']!=summarize_repeats(row['raw_repeats']):raise RuntimeError('Question-level repeated timing receipt differs')
            for r in row['raw_repeats'][1:]:
                for sub in r['pairs'].values():extra.extend(sub['generation_calls']);retrievals.extend(sub['retrieval_trace'])
    report['full_cost_totals']['additional_fixed_timing_repeats']={
        'generation_calls':len(extra),'total_input_tokens':sum(c['input_tokens'] for c in extra),'total_output_tokens':sum(c['output_tokens'] for c in extra),
        'generation_ms':sum(c['ms'] for c in extra),'retrieval_calls':len(retrievals),'retrieval_ms':sum(t['ms'] for t in retrievals),
        'failed_generation_calls':sum(c['status']=='failed' for c in extra),
        'ownership':'repeats 1..11 only; selection/repeat0 in separate_technical_checks, C in v3_extra_technical; raw logs are duplicate evidence'}
    report['independent_on_off_design']={'fixed_pairs_per_question':REPEATS,'balanced_orders':[6,6],'bootstrap_unit':'question',
        'replicates':20000,'seed':42,'excluded_observations':0,'no_optional_stopping':True,'gate_upper_ratio':.005}
    return report
