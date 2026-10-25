"""Build a pilot BM25 corpus using only HotpotQA TRAIN context fields."""
from pathlib import Path
import hashlib,json,os,shutil,sqlite3,subprocess,sys,time
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            h.update(block)
    return h.hexdigest()

def main():
    out=ROOT/"data/corpus"
    out.mkdir(parents=True,exist_ok=True)
    if (out/"manifest.json").exists():
        print("Existing frozen pilot corpus retained.")
        return
    source=json.loads((ROOT/"configs/source_lock.json").read_text())["sources"]["hotpotqa"]
    files=sorted(p for p in source["files"] if p.startswith("distractor/train-") and p.endswith(".parquet"))
    assert len(files)==2
    hf=shutil.which("hf") or str(Path(sys.executable).parent/("hf.exe" if os.name=="nt" else "hf"))
    dest=ROOT/"data/raw/hotpotqa"
    help_text=subprocess.run([hf,"download","--help"],capture_output=True,text=True,check=True).stdout
    repo_type_flag="--type" if "--type " in help_text else "--repo-type"
    subprocess.run([hf,"download",source["repo_id"],*files,repo_type_flag,"dataset","--revision",source["revision"],"--local-dir",str(dest)],env={**os.environ,"HF_HUB_DISABLE_XET":"1"},check=True)
    temp=out/"pilot-building.sqlite"
    if temp.exists():
        raise RuntimeError("Incomplete build exists; inspect it before restarting")
    start=time.perf_counter()
    conn=sqlite3.connect(temp)
    conn.execute("CREATE VIRTUAL TABLE passages USING fts5(doc_id UNINDEXED,title,text, tokenize='porter unicode61 remove_diacritics 2')")
    seen=set()
    digest=hashlib.sha256()
    sources=[]
    raw_rows=0
    for filename in files:
        path=dest/filename
        for batch in pq.ParquetFile(path).iter_batches(batch_size=512,columns=["context"]):
            # No question, answer, or supporting-fact field is read for indexing.
            for row in batch.to_pylist():
                raw_rows+=1
                ctx=row["context"]
                for title,sentences in zip(ctx["title"],ctx["sentences"]):
                    text="".join(sentences).strip()
                    if not text:
                        continue
                    record=json.dumps([title,text],ensure_ascii=False,separators=(",",":"))
                    docid=hashlib.sha256(record.encode()).hexdigest()
                    if docid in seen:
                        continue
                    seen.add(docid)
                    digest.update((docid+"\n").encode())
                    conn.execute("INSERT INTO passages(doc_id,title,text) VALUES (?,?,?)",(docid,title,text))
            conn.commit()
        sources.append({"file":filename,"sha256":sha(path)})
    conn.execute("INSERT INTO passages(passages) VALUES ('optimize')")
    conn.commit()
    count=conn.execute("SELECT count(*) FROM passages").fetchone()[0]
    conn.close()
    target=out/"pilot.sqlite"
    if target.exists():
        raise RuntimeError("Existing corpus file without manifest; inspect before replacing")
    temp.rename(target)
    info={"scope":"pilot_train_context_corpus_not_fullwiki","repo_id":source["repo_id"],"revision":source["revision"],"source_split":"train","source_fields_read":["context"],"source_files":sources,"source_rows":raw_rows,"documents":count,"ordered_document_ids_sha256":digest.hexdigest(),"sqlite_sha256":sha(target),"sqlite_version":sqlite3.sqlite_version,"build_seconds":time.perf_counter()-start,"test_or_calibration_used_for_index":False,"corpus_license":"CC-BY-SA-4.0"}
    (out/"manifest.json").write_text(json.dumps(info,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(info,ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
