"""Offline scoring ONLY after a shard's entire prediction commit is verified."""
from pathlib import Path
import argparse,gzip,json,time
from test1000_contract import read,write,sha,jsonl,samples,require_prediction_commit
from run_baselines import answer_metrics
from analyze_test1000 import summarize
ROOT=Path(__file__).resolve().parents[1]
def score(root,shard,vault):
    root=Path(root);start=time.perf_counter();commit=require_prediction_commit(root,shard) # BEFORE vault access
    manifest=read(root/'data/test1000/manifest.json');entry=manifest['shards'][str(shard)]
    if sha(vault)!=entry['scoring_vault_sha256']:raise RuntimeError('Scoring vault SHA differs')
    references=[json.loads(x) for x in gzip.decompress(Path(vault).read_bytes()).decode('utf-8').splitlines()]
    refs={r['sample_id']:r['answers'] for r in references};questions=samples(root,shard)
    if len(references)!=500 or len(refs)!=500 or set(refs)!={q['sample_id'] for q in questions}:raise RuntimeError('Reference shard coverage')
    out=root/'results/test1000';target=out/'scored.jsonl'
    if target.exists():raise RuntimeError('Never overwrite held-out scoring')
    rows=jsonl(out/'predictions.jsonl')
    with target.open('x',encoding='utf-8') as f:
        for r in rows:f.write(json.dumps({**r,'quality':answer_metrics(r['answer'],refs[r['sample_id']],r['dataset'])},ensure_ascii=False)+'\n')
    report=summarize(jsonl(target),jsonl(out/'paired_observations.jsonl'))
    report.update(shard=shard,reference_access='only after complete prediction SHA receipt',prediction_commit_sha256=sha(out/'prediction_commit.json'),
        common_contract={n:sha(root/n) for n in ['configs/method_freeze_v04_v3_native_state.json','configs/deployment_test1000.json','configs/test1000.yaml']})
    write(out/'summary.json',report)
    receipt={'status':'complete_offline_scoring','shard':shard,'normal_rows':4000,'scored_sha256':sha(target),'summary_sha256':sha(out/'summary.json'),
        'prediction_commit_sha256':sha(out/'prediction_commit.json'),'reference_vault_sha256':sha(vault),'ms':1000*(time.perf_counter()-start),'new_model_calls':0}
    write(out/'scoring_receipt.json',receipt);return receipt
def main():
    p=argparse.ArgumentParser();p.add_argument('--shard',type=int,choices=[1,2],required=True);p.add_argument('--vault',type=Path,required=True);a=p.parse_args();print(json.dumps(score(ROOT,a.shard,a.vault)))
if __name__=='__main__':main()
