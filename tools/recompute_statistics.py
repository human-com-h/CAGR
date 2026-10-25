"""Recompute original question-level estimators from released measurement CSVs."""
from pathlib import Path
import argparse,csv,json,math
from statistics_i import summarize as summarize_i
from statistics_ii import summarize as summarize_ii
ROOT=Path(__file__).resolve().parents[1]
def load_metrics(p):
    with p.open(encoding='utf-8') as f:rr=list(csv.DictReader(f))
    result=[]
    for r in rr:
        result.append({'sample_id':r['sample_id'],'dataset':r['dataset'],'shard':int(r['shard']),'method':r['method'],'tier':int(r['tier']),
            'quality':{k[8:]:float(v) for k,v in r.items() if k.startswith('quality_') and v},
            'cost':{k[5:]:float(v) for k,v in r.items() if k.startswith('cost_') and v and v not in ['True','False']}})
    return result
def load_pairs(p):
    if p.suffix=='.jsonl':return [json.loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l]
    with p.open(encoding='utf-8') as f:return [{**r,'shard':int(r['shard']),'repeat':int(r['repeat']),'ratio':float(r['ratio'])} for r in csv.DictReader(f)]
def matching(actual,expected):
    n=0
    if isinstance(actual,dict):
        for k in actual.keys() & expected.keys():n+=matching(actual[k],expected[k])
    elif isinstance(actual,list):
        assert len(actual)==len(expected)
        for a,b in zip(actual,expected):n+=matching(a,b)
    elif isinstance(actual,(int,float)) and not isinstance(actual,bool):
        if not math.isclose(actual,expected,rel_tol=1e-10,abs_tol=1e-10):raise AssertionError((actual,expected))
        n=1
    return n
def main():
    p=argparse.ArgumentParser();p.add_argument('--cohort',choices=['I','II','both'],default='both');p.add_argument('--metrics',type=Path);p.add_argument('--pairs',type=Path);p.add_argument('--output',type=Path,default=ROOT/'reproduced/statistics');a=p.parse_args()
    if (a.metrics is None)!=(a.pairs is None) or (a.metrics and a.cohort=='both'):p.error('Custom --metrics and --pairs require one cohort')
    a.output.mkdir(parents=True,exist_ok=True);checks=0
    for c in ['I','II'] if a.cohort=='both' else [a.cohort]:
        rows=load_metrics(a.metrics or ROOT/f'evidence/test_cohort_{c}_question_metrics.csv');pairs=load_pairs(a.pairs or ROOT/f'evidence/test_cohort_{c}_paired_ratios.csv')
        r=(summarize_i if c=='I' else summarize_ii)(rows,pairs)
        if not a.metrics:
            original=json.loads((ROOT/('results/test1000-final/frozen-v1/merged_summary.json' if c=='I' else 'results/heldout-v2-analysis/v1-independent/summary.json')).read_text())
            checks+=matching(r,original)
        (a.output/f'cohort_{c.lower()}.json').write_text(json.dumps(r,indent=2))
    print(json.dumps({'status':'passed','numeric_statistics_compared':checks,'bootstrap_seed':42,'bootstrap_replicates':20000,'new_inference':0}))
if __name__=='__main__':main()
