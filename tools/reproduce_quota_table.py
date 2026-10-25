"""Export the existing first-stage quota diagnostic, without policy replay."""
from pathlib import Path
import csv
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reproduced'
LOG=OUT/'audit'
saved=json.loads((ROOT/'results/v2-feasibility/local-v3/stage0_quota_diagnostics.json').read_text(encoding='utf-8'))
headers=['First-stage rule','H/T searches','H F1 (%)','H delta (pp)','T F1 (%)','All F1 (%)','All delta (pp)']
rows=[]
base=saved['metrics']['original_410']
for key,label in [('original_410','Original gate'),('CAGR_score_top_555','Score: 555 total'),
                  ('CAGR_score_match_TARG_per_dataset','Score: dataset quotas'),('exact_TARG_selection','TARG-style selections')]:
    h,t,o=[saved['metrics'][key][ds] for ds in ['hotpotqa','triviaqa','overall']]
    rows.append([label,f"{h['first_retrievals']}/{t['first_retrievals']}",f"{100*h['f1']:.4f}",
                 f"{100*(h['f1']-base['hotpotqa']['f1']):+.4f}",f"{100*t['f1']:.4f}",
                 f"{100*o['f1']:.4f}",f"{100*(o['f1']-base['overall']['f1']):+.4f}"])
caption=('Post hoc first-stage quota decomposition in Test Cohort I, using saved Direct and single-retrieval answers only. '
         'H and T each contain 500 questions. Deltas use the original binary gate as reference. '
         'Score ordering uses the frozen initiation utility; dataset quotas match 314 HotpotQA and 241 TriviaQA searches. '
         'No second-stage action or new latency measurement is included.')
block='\n'.join(['Table: '+caption,'','| '+' | '.join(headers)+' |',
                '| '+' | '.join(['---']+['---:']*(len(headers)-1))+' |']+
                ['| '+' | '.join(row)+' |' for row in rows])
with (OUT/'source_data/quota_alignment.csv').open('w',newline='',encoding='utf-8') as f:
    writer=csv.writer(f);writer.writerow(headers);writer.writerows(rows)
def esc(s):return s.replace('&',r'\&').replace('%',r'\%').replace('_',r'\_')
tex=[r'\begin{table}[!htbp]',r'\centering',r'\small',r'\caption{'+esc(caption)+'}',
     r'\label{tab:quota-alignment}',r'\ifdefined\cagrtablebox\else\newsavebox{\cagrtablebox}\fi',
     r'\sbox{\cagrtablebox}{%',r'\begin{tabular}{lrrrrrr}',r'\toprule',
     ' & '.join(map(esc,headers))+r' \\',r'\midrule']
tex+=[' & '.join(map(esc,row))+r' \\' for row in rows]
tex+=[r'\bottomrule',r'\end{tabular}}',
      r'\ifdim\wd\cagrtablebox>\linewidth\resizebox{\linewidth}{!}{\usebox{\cagrtablebox}}\else\usebox{\cagrtablebox}\fi',
      r'\end{table}','']
(OUT/'tables/quota_alignment.tex').write_text('\n'.join(tex),encoding='utf-8')
path=LOG/'table_blocks.json'
blocks=json.loads(path.read_text(encoding='utf-8'));blocks['quota_alignment']=block
path.write_text(json.dumps(blocks,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
source_path=ROOT/'results/v2-feasibility/local-v3/stage0_quota_diagnostics.json'
mapping_path=LOG/'source_map.json'
mapping=json.loads(mapping_path.read_text(encoding='utf-8'))
mapping['quota_alignment']={'path':source_path.relative_to(ROOT).as_posix(),
                           'sha256':hashlib.sha256(source_path.read_bytes()).hexdigest(),
                           'scope':'existing post hoc first-stage saved-answer diagnostic'}
mapping_path.write_text(json.dumps(mapping,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Exported one existing quota diagnostic; no inference or policy fitting.')
