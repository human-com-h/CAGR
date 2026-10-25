"""Score committed predictions separately; never used by routing."""
from pathlib import Path
import argparse,csv,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'frozen/cohort_II/scripts'))
from run_baselines import answer_metrics
def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--references',type=Path,required=True);a=p.parse_args()
    receipt=json.loads((a.run/'prediction_commit.json').read_text())
    for n,digest in receipt['sha256'].items():
        if hashlib.sha256((a.run/n).read_bytes()).hexdigest()!=digest:raise RuntimeError('Prediction commit changed')
    refs={r['sample_id']:r['answers'] for r in map(json.loads,a.references.read_text(encoding='utf-8').splitlines())}
    result=[]
    for r in map(json.loads,(a.run/'predictions.jsonl').read_text(encoding='utf-8').splitlines()):
        q=answer_metrics(r['answer'],refs[r['sample_id']],r['dataset']);out={k:r[k] for k in ['sample_id','dataset','shard','method','tier','status']}
        out.update({'quality_'+k:v for k,v in q.items()});out.update({'cost_'+k:v for k,v in r['cost'].items() if isinstance(v,(int,float,bool))});result.append(out)
    with (a.run/'question_metrics.csv').open('x',encoding='utf-8',newline='') as f:
        names=list(dict.fromkeys(k for r in result for k in r));w=csv.DictWriter(f,names);w.writeheader();w.writerows(result)
    print(json.dumps({'scored_rows':len(result),'receipt_scope':receipt['scope']}))
if __name__=='__main__':main()
