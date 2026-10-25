"""Independently recompute main quality/cost tables from scalar question records."""
from pathlib import Path
import collections,csv,hashlib,json,statistics
ROOT=Path(__file__).resolve().parents[1]
def rows(p):
    with Path(p).open(encoding='utf-8') as f:return list(csv.DictReader(f))
def verify():
    aliases=json.loads((ROOT/'evidence/method_labels.json').read_text());checks=0
    for cohort in ['I','II']:
        rr=rows(ROOT/f'evidence/test_cohort_{cohort}_question_metrics.csv');pairs=rows(ROOT/f'evidence/test_cohort_{cohort}_paired_ratios.csv')
        if len(rr)!=8000 or len({(r['sample_id'],r['method']) for r in rr})!=8000:raise ValueError('Invalid 1000 x 8 coverage')
        labels={**aliases['common'],**aliases[cohort]};label_to_id={v:k for k,v in labels.items()}
        for display in rows(ROOT/f'evidence/aggregate_tables/cohort_{cohort.lower()}.csv'):
            part=[r for r in rr if r['method']==label_to_id[display['Method']]]
            assert len(part)==1000
            for prefix,ds in [('H','hotpotqa'),('T','triviaqa'),('All',None)]:
                ss=[r for r in part if ds is None or r['dataset']==ds]
                for score in ['EM','F1']:
                    value=100*statistics.mean(float(r['quality_'+score.lower()]) for r in ss)
                    if f'{value:.2f}'!=display[prefix+' '+score]:raise ValueError((cohort,display['Method'],score,value))
                    checks+=1
            for col,key,decimals in [('Latency (ms)','cost_end_to_end_ms',2),('Retr./q','cost_retrieval_calls',3)]:
                if f'{statistics.mean(float(r[key]) for r in part):.{decimals}f}'!=display[col]:raise ValueError((cohort,col))
                checks+=1
        assert len(pairs)==2000
        for ds in ['hotpotqa','triviaqa']:
            grouped=collections.defaultdict(list)
            for r in pairs:
                if r['dataset']==ds:grouped[r['sample_id']].append(float(r['ratio']))
            assert len(grouped)==500 and all(len(v)==2 for v in grouped.values())
    manifest=json.loads((ROOT/'provenance/frozen_sources.json').read_text());files=0
    for cohort,entry in manifest.items():
        for n,digest in entry['files'].items():
            if hashlib.sha256((ROOT/'frozen'/cohort/n).read_bytes()).hexdigest()!=digest:raise ValueError('Frozen source changed: '+n)
            files+=1
    result={'status':'passed','main_scalar_checks':checks,'question_method_rows':16000,'paired_observations':4000,'frozen_source_files':files}
    print(json.dumps(result,indent=2));return result
if __name__=='__main__':verify()
