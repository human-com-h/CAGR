"""Restore frozen benchmark IDs from official sources; build TRAIN-only BM25."""
from pathlib import Path
import argparse, hashlib, io, json, sqlite3, struct, urllib.request

ROOT=Path(__file__).resolve().parents[1]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def acquire(dataset,entry,rawdir):
    p=rawdir/dataset/entry['file'];p.parent.mkdir(parents=True,exist_ok=True)
    if not p.exists():
        if p.with_suffix('.partial').exists():raise RuntimeError('Partial download exists; inspect it before retrying')
        source=read(ROOT/'configs/source_lock.json')['sources'][dataset]
        url=f"https://huggingface.co/datasets/{source['repo_id']}/resolve/{source['revision']}/{entry['file']}"
        with urllib.request.urlopen(url,timeout=60) as r,p.with_suffix('.partial').open('xb') as f:
            while chunk:=r.read(1024*1024):f.write(chunk)
        if sha(p.with_suffix('.partial'))!=entry['sha256']:raise RuntimeError('Downloaded source SHA mismatch')
        p.with_suffix('.partial').rename(p)
    if sha(p)!=entry['sha256']:raise RuntimeError('Source SHA mismatch: '+str(p))
    return p
def parquet(p):
    import pyarrow as pa
    import pyarrow.parquet as pq
    # A metadata-only compatibility transform from the recorded local reader.
    # All compressed data pages remain identical; original Parquet is untouched.
    from parquet_compat import Compact,field
    raw=p.read_bytes();n=struct.unpack('<I',raw[-8:-4])[0]
    parser=Compact(raw[-8-n:-8]);metadata=parser.value(12)
    if parser.i!=n:raise RuntimeError('Invalid Parquet metadata')
    for group in field(metadata,4)[1]:
        for column in field(group,1)[1]:
            cm=field(column,3);cm.pop(16,None);cm.pop(17,None)
    writer=Compact();writer.put(12,metadata)
    return pq.ParquetFile(pa.BufferReader(raw[:-8-n]+writer.out+struct.pack('<I',len(writer.out))+b'PAR1'))
def restore(rawdir,out):
    specs=read(ROOT/'configs/generation_settings.json')['data_sources'];available={}
    for e in specs:
        ds=e['dataset'];p=acquire(ds,e,rawdir);idcol='id' if ds=='hotpotqa' else 'question_id'
        available[ds]={}
        for batch in parquet(p).iter_batches(batch_size=512,columns=[idcol,'question','answer']):
            for r in batch.to_pylist():
                sid=ds+':'+str(r[idcol]);answer=r['answer']
                answers=[answer] if ds=='hotpotqa' else list(dict.fromkeys([answer['value']]+answer['aliases']+answer['normalized_aliases']))
                available[ds][sid]=(r['question'].strip(),answers)
    counts={}
    for partition,ids in read(ROOT/'configs/split_ids.json').items():
        qs=[];refs=[]
        for entry in ids:
            q,answers=available[entry['dataset']][entry['sample_id']]
            if hashlib.sha256(q.encode()).hexdigest()!=entry['question_sha256']:raise RuntimeError('Frozen question mismatch')
            qs.append({**{k:v for k,v in entry.items() if k!='question_sha256'},'question':q})
            refs.append({'sample_id':entry['sample_id'],'answers':answers})
        for label,rows in [('questions',qs),('references',refs)]:
            p=out/f'{partition}_{label}.jsonl'
            with p.open('x',encoding='utf-8',newline='\n') as f:
                for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
        counts[partition]=len(qs)
    return counts
def corpus(rawdir,out):
    expected=read(ROOT/'configs/corpus_expected.json');target=out/'corpus.sqlite'
    if target.exists():raise RuntimeError('Refusing to replace an index')
    conn=sqlite3.connect(target)
    try:
        conn.execute("CREATE VIRTUAL TABLE passages USING fts5(doc_id UNINDEXED,title,text, tokenize='porter unicode61 remove_diacritics 2')")
        seen=set();digest=hashlib.sha256();rows=0
        for entry in expected['source_files']:
            p=acquire('hotpotqa',entry,rawdir)
            for batch in parquet(p).iter_batches(batch_size=512,columns=['context']):
                for row in batch.to_pylist():
                    rows+=1;c=row['context']
                    for title,sentences in zip(c['title'],c['sentences']):
                        text=''.join(sentences).strip()
                        if not text:continue
                        docid=hashlib.sha256(json.dumps([title,text],ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
                        if docid in seen:continue
                        seen.add(docid);digest.update((docid+'\n').encode());conn.execute('INSERT INTO passages VALUES (?,?,?)',(docid,title,text))
                conn.commit()
        conn.execute("INSERT INTO passages(passages) VALUES ('optimize')");conn.commit()
        if len(seen)!=expected['documents'] or digest.hexdigest()!=expected['ordered_document_ids_sha256'] or rows!=expected['source_rows']:raise RuntimeError('Corpus identity mismatch')
    finally:conn.close()
    result={'documents':len(seen),'ordered_document_ids_sha256':digest.hexdigest(),'sqlite_sha256':sha(target),'sqlite_version':sqlite3.sqlite_version,
        'original_sqlite_byte_match':sha(target)==expected['sqlite_sha256'],'train_context_only':True}
    (out/'corpus_receipt.json').write_text(json.dumps(result,indent=2))
    return result
def model(out):
    from huggingface_hub import snapshot_download
    source=read(ROOT/'configs/source_lock.json')['sources']['generator'];lock=read(ROOT/'frozen/cohort_II/configs/model_asset_lock.json')
    path=Path(snapshot_download(source['repo_id'],revision=source['revision'],local_dir=out/'model'))
    for n,e in lock['files'].items():
        if sha(path/n)!=e['sha256'] or (path/n).stat().st_size!=e['bytes']:raise RuntimeError('Model SHA mismatch: '+n)
    return {'revision':source['revision'],'files_verified':len(lock['files'])}
def main():
    p=argparse.ArgumentParser();p.add_argument('--datasets',action='store_true');p.add_argument('--corpus',action='store_true');p.add_argument('--model',action='store_true');p.add_argument('--raw-dir',type=Path,default=ROOT/'resources/raw');p.add_argument('--output',type=Path,default=ROOT/'resources');a=p.parse_args()
    if not any([a.datasets,a.corpus,a.model]):p.error('Choose --datasets, --corpus and/or --model')
    a.output.mkdir(parents=True,exist_ok=True);result={}
    if a.datasets:result['splits']=restore(a.raw_dir,a.output)
    if a.corpus:result['corpus']=corpus(a.raw_dir,a.output)
    if a.model:result['model']=model(a.output)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
