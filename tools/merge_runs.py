"""Merge separately committed, scored shards after full prediction completion."""
from pathlib import Path
import argparse,csv,hashlib,json
def main():
    p=argparse.ArgumentParser();p.add_argument('--shard-1',type=Path,required=True);p.add_argument('--shard-2',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise RuntimeError('Refusing to overwrite merged outputs')
    metrics=[];pairs=[];cohorts=[]
    for i,root in enumerate([a.shard_1,a.shard_2],1):
        receipt=json.loads((root/'prediction_commit.json').read_text())
        if receipt['scope']!='full_shard' or receipt['shard']!=i or receipt['normal_rows']!=4000 or receipt['technical_rows']!=2000 or receipt['pairs']!=1000:raise RuntimeError('Incomplete/smoke shard cannot be merged')
        for n,digest in receipt['sha256'].items():
            if hashlib.sha256((root/n).read_bytes()).hexdigest()!=digest:raise RuntimeError('Shard commit changed')
        cohorts.append(receipt['cohort'])
        with (root/'question_metrics.csv').open(encoding='utf-8') as f:metrics+=list(csv.DictReader(f))
        pairs+=[json.loads(l) for l in (root/'paired_observations.jsonl').read_text().splitlines() if l]
    if len(set(cohorts))!=1 or len(metrics)!=8000 or len({(r['sample_id'],r['method']) for r in metrics})!=8000 or len({r['sample_id'] for r in metrics})!=1000 or len(pairs)!=2000 or len({(r['sample_id'],r['repeat']) for r in pairs})!=2000:raise RuntimeError('Cross-cohort or duplicate/missing observations')
    a.output.mkdir(parents=True)
    with (a.output/'question_metrics.csv').open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,list(dict.fromkeys(k for r in metrics for k in r)));w.writeheader();w.writerows(metrics)
    with (a.output/'paired_observations.jsonl').open('x',encoding='utf-8') as f:
        for row in pairs:f.write(json.dumps(row)+'\n')
    print(json.dumps({'cohort':cohorts[0],'question_method_rows':len(metrics),'pairs':len(pairs)}))
if __name__=='__main__':main()
