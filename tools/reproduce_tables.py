"""Assemble publication tables and plots from existing results; no inference."""
from pathlib import Path
import csv
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reproduced'
QA = OUT / 'audit'
for p in [OUT / 'tables', OUT / 'figures', OUT / 'source_data', QA]:
    p.mkdir(parents=True, exist_ok=True)

def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))

SOURCES = {
    'I': 'results/test1000-final/frozen-v1/merged_summary.json',
    'II': 'results/heldout-v2-analysis/v1-independent/summary.json',
    'stage0': 'results/cagr-v3-local/feasibility-v1/calibration_report.json',
    'stage1': 'results/v2-feasibility/local-v3/feature_ablation_replay.json',
    'stage1_quality': 'results/v2-feasibility/local-v3/feature_ablation_quality.json',
    'prefix': 'results/v2-feasibility/local-v3/prefix_K_sensitivity.json',
    'random': 'results/v2-feasibility/local-v3/random_gate_quality_replay.json',
    'actions': 'results/cagr-v2-route23-audit/v1-independent/diagnostics-final/action_quality.json',
    'bridge': 'results/cagr-v2-route23-audit/v1-independent/diagnostics-final/bridge_diagnosis.json',
    'stop': 'results/v2-feasibility/local-v3/novelty_stop_replay.json',
}
D = {k: read(v) for k, v in SOURCES.items()}
names = {'direct': 'Direct', 'always_rag': 'Always-RAG', 'always_multihop': 'Always-Multihop',
         'binary_gate': 'Matched Binary', 'matched_binary_v2': 'Matched Binary',
         'targ_style': 'TARG-style', 'sugar_l_style_budgeted': 'SUGAR-L-style',
         'cagr': 'CAGR-R', 'cagr_in_situ': 'CAGR-R (answer expansion)',
         'cagr_v1': 'CAGR-R', 'cagr_v2': 'CAGR-F'}
order = {'I': ['direct', 'always_rag', 'always_multihop', 'binary_gate', 'targ_style',
               'sugar_l_style_budgeted', 'cagr', 'cagr_in_situ'],
         'II': ['direct', 'always_rag', 'always_multihop', 'matched_binary_v2', 'targ_style',
                'sugar_l_style_budgeted', 'cagr_v1', 'cagr_v2']}
tables = {}

def esc(s):
    return str(s).replace('&', r'\&').replace('%', r'\%').replace('_', r'\_')

def table(key, headers, rows, caption):
    tables[key] = '\n'.join(['Table: ' + caption, '', '| ' + ' | '.join(headers) + ' |',
                            '| ' + ' | '.join(['---'] + ['---:'] * (len(headers)-1)) + ' |'] +
                           ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])
    with (OUT / 'source_data' / (key + '.csv')).open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(headers); w.writerows(rows)
    # Standalone standard LaTeX table, also usable outside the full manuscript.
    tex = ['\\begin{table}[!htbp]', '\\centering', '\\small', '\\caption{' + esc(caption) + '}',
           '\\label{tab:' + key.replace('_','-') + '}',
           '\\ifdefined\\cagrtablebox\\else\\newsavebox{\\cagrtablebox}\\fi',
           '\\sbox{\\cagrtablebox}{%', '\\begin{tabular}{l' + 'r'*(len(headers)-1) + '}',
           '\\toprule', ' & '.join(esc(x.replace('Direct/Single/Upgrade','D/S/E')) for x in headers) + r' \\', '\\midrule']
    tex += [' & '.join(map(esc, row)) + r' \\' for row in rows]
    tex += [r'\bottomrule', r'\end{tabular}}',
            r'\ifdim\wd\cagrtablebox>\linewidth\resizebox{\linewidth}{!}{\usebox{\cagrtablebox}}\else\usebox{\cagrtablebox}\fi',
            r'\end{table}', '']
    (OUT / 'tables' / (key + '.tex')).write_text('\n'.join(tex), encoding='utf-8')

for cohort in ['I', 'II']:
    rows=[]
    for k in order[cohort]:
        h,t,o=[D[cohort]['methods'][s][k] for s in ['hotpotqa','triviaqa','overall']]
        rows.append([names[k],f"{100*h['em']:.2f}",f"{100*h['f1']:.2f}",
                     f"{100*t['em']:.2f}",f"{100*t['f1']:.2f}",
                     f"{100*o['em']:.2f}",f"{100*o['f1']:.2f}",
                     f"{o['end_to_end_ms']:.2f}",f"{o['retrieval_calls']:.3f}"])
    table('cohort_'+cohort.lower(), ['Method','H EM','H F1','T EM','T F1','All EM','All F1','Latency (ms)','Retr./q'], rows,
          f'Test Cohort {cohort}: eight methods on 500 HotpotQA and 500 TriviaQA questions. EM and F1 are percentages.')
    rows=[]
    for ds,label in [('hotpotqa','HotpotQA'),('triviaqa','TriviaQA'),('overall','Overall')]:
        for k in order[cohort]:
            r=D[cohort]['methods'][ds][k]
            rows.append([label,names[k],f"{r['end_to_end_ms']:.2f}",f"{r['p50_ms']:.2f}",f"{r['p95_ms']:.2f}",
                         f"{r['total_input_tokens']:.3f}",f"{r['total_output_tokens']:.3f}",
                         f"{r['retrieval_calls']:.3f}",'/'.join(map(str,r['tier_counts']))])
    table('cost_'+cohort.lower(),['Dataset','Method','Mean ms','P50 ms','P95 ms','Input tok.','Output tok.','Retr./q','Direct/Single/Upgrade'], rows,
          f'Test Cohort {cohort}: complete method costs and branch counts. Tokens include all generation calls within a method.')

rows=[]
for cohort in ['I','II']:
    for ds,label in [('hotpotqa','HotpotQA'),('triviaqa','TriviaQA')]:
        r=D[cohort]['overhead'][ds]; f=r['full_components']; i=r['independent_on_off']
        rows.append([cohort,label]+[f'{100*x:.5f}' for x in [f['mean'],f['upper_one_sided95'],i['mean'],i['upper_one_sided95']]])
table('overhead',['Cohort','Dataset','Component mean (%)','Component U95 (%)','On/off mean (%)','On/off U95 (%)'],rows,
      'Routing overhead for CAGR-R in Test Cohort I and CAGR-F in Test Cohort II; N=500 questions per row. U95 is a one-sided 95% upper bound.')

for cohort,target in [('I','CAGR-R'),('II','CAGR-F')]:
    rows=[]
    for ds,label in [('hotpotqa','HotpotQA'),('triviaqa','TriviaQA'),('overall','Overall')]:
        for comp in ['always_rag', 'binary_gate' if cohort=='I' else 'matched_binary_v2','targ_style','always_multihop','sugar_l_style_budgeted']:
            r=D[cohort]['paired'][ds][comp]
            vals=[]
            for metric,mult in [('f1',100),('end_to_end_ms',1)]:
                a=r[metric]; vals.append(f"{mult*a['mean']:+.4f} [{mult*a['ci95'][0]:+.4f}, {mult*a['ci95'][1]:+.4f}]")
            rows.append([label,names[comp]]+vals)
    table('contrasts_'+cohort.lower(),['Dataset','Comparator','F1 difference, pp [95% CI]','Latency difference, ms [95% CI]'],rows,
          f'Test Cohort {cohort}: {target} minus each comparator. Paired percentile intervals use 20,000 resamples.')

rows=[]
for mode,label,dim in [('pilot','Pilot K=5',1),('structure','Question structure',30),('fused','Fusion',61)]:
    r=D['stage0']['ablations'][mode]
    rows.append([label,dim,f"{r['OOF_benefit_auc']:.4f}",f"{r['OOF_mse']:.4f}",
                 f"{100*r['reports']['OOF']['overall']['f1']:.2f}",f"{100*r['reports']['Full']['overall']['f1']:.2f}"])
table('stage0',['Features','Dimensions','OOF AUC','OOF MSE','OOF F1 (%)','Training F1 (%)'],rows,
      'Stage 0 exploratory feature comparison on the Development Calibration Split (N=200, five folds). Each family was selected within a finite development search.')

rows=[]
for key in ['gap_base','gap_text']:
    r=D['stage1'][key]; q=D['stage1_quality'][key]
    rows.append([key,3 if key=='gap_base' else 8,f"{r['gain_AUC']:.4f}",f"{r['gain_RMSE']:.5f}",
                 f"{100*q['hotpotqa']['f1']:.2f}",f"{100*q['triviaqa']['f1']:.2f}",'/'.join(map(str,r['common_lambda_3e_5_threshold_0_tiers']))])
table('stage1',['Features','Dimensions','OOF AUC','OOF RMSE','H F1 (%)','T F1 (%)','Direct/Single/Upgrade'],rows,
      'Stage 1 development ablation at common cost weight 0.00003 and utility threshold zero. This diagnostic uses different thresholds from the frozen operating points.')

rows=[]
for key,label in [('stop','Stop after first retrieval'),('in_situ','Answer expansion'),('kv_bridge','KV-Bridge'),('rewrite','Query rewriting')]:
    h,t=[D['actions'][s][key] for s in ['hotpotqa','triviaqa']]
    signs=h['F1_signs_vs_H1']
    rows.append([label,f"{100*h['em']:.2f}",f"{100*h['f1']:.2f}",f"{100*t['em']:.2f}",f"{100*t['f1']:.2f}",
                 '/'.join(str(signs[s]) for s in ['positive','tie','negative'])])
table('actions',['Action','H EM (%)','H F1 (%)','T EM (%)','T F1 (%)','H gain/tie/harm'],rows,
      'Complete four-action development matrix, forced on all 200 questions. Gain/tie/harm is relative to the first retrieved answer; these are potential action outcomes, not a routed method.')

b=D['bridge']; rows=[]
for label,k1,k2 in [('Mean call time (ms)','rewrite_ms','bridge_ms'),('Mean new input tokens','rewrite_input_tokens','bridge_input_tokens'),('Mean output tokens','rewrite_output_tokens','bridge_output_tokens')]:
    rows.append([label,f"{b[k1]['mean']:.3f}",f"{b[k2]['mean']:.3f}"])
rows.insert(1,['P95 call time (ms)',f"{b['rewrite_ms']['p95']:.2f}",f"{b['bridge_ms']['p95']:.2f}"])
table('bridge',['Measure','Query rewriting','KV-Bridge'],rows,
      'KV-Bridge development calls on 200 paired questions; end-to-end call times include the appended suffix, decoding, and host work.')

rows=[]
for k in range(1,6):
    r=D['prefix'][str(k)]
    for method,label in [('CAGR_fixed_head_stage0','CAGR-R first gate'),('TARG_fixed_threshold','TARG-style')]:
        h,t,o=[r[method][s] for s in ['hotpotqa','triviaqa','overall']]
        rows.append([k,label,h['first_retrievals'],t['first_retrievals'],f"{100*h['f1']:.2f}",f"{100*t['f1']:.2f}",f"{100*o['f1']:.2f}"])
table('prefix',['K','Fixed rule','H retrievals','T retrievals','H F1 (%)','T F1 (%)','All F1 (%)'],rows,
      'Test Cohort I prefix truncation diagnostic: first-stage quality only. The K=5 heads and thresholds remain fixed; no K-specific retraining or new timed generation was performed.')

rows=[]
for key,label in [('global_quota','Global quotas'),('per_dataset_quota','Within-dataset quotas')]:
    r=D['random'][key]
    rows.append([label]+[f"{100*r[s]['mean']:.2f} ({100*r[s]['sd']:.2f})" for s in ['hotpot_f1','trivia_f1']])
rows.append(['CAGR-R observed']+[f"{100*D['I']['methods'][ds]['cagr']['f1']:.2f}" for ds in ['hotpotqa','triviaqa']])
table('random',['Assignment','H F1 %, mean (SD)','T F1 %, mean (SD)'],rows,
      'Test Cohort I matched-quota Random Gate: 20,000 assignments from seed 42. SD describes the random-policy distribution, not uncertainty from new test sampling.')

rows=[]
for cost in [131,300,500,2500]:
    row=[cost]
    for key in ['always_rag','targ_style','cagr_v1','cagr_v2']:
        r=D['II']['methods']['overall'][key]
        row.append(f"{r['end_to_end_ms']-r['retrieval_ms']+cost*r['retrieval_calls']:.2f}")
    rows.append(row)
table('cost_spectrum',['Cost per retrieval (ms)','Always-RAG','TARG-style','CAGR-R','CAGR-F'],rows,
      'Fixed-path cost substitution on Test Cohort II, in milliseconds per question. These are algebraic scenarios with observed answers and branch counts held fixed, not new hardware measurements.')

(QA/'table_blocks.json').write_text(json.dumps(tables,ensure_ascii=False,indent=2),encoding='utf-8')
(QA/'source_map.json').write_text(json.dumps({k:{'path':v,'sha256':hashlib.sha256((ROOT/v).read_bytes()).hexdigest()} for k,v in SOURCES.items()},indent=2),encoding='utf-8')


print('Generated 14 tables from archived evaluation outputs.')
