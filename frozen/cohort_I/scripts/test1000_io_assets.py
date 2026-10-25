"""Sequential mount copy with the SAME full verifier on its byte-identical local copy.

No SQLite connection is opened against Kaggle's high-latency mounted filesystem.
The source manifest and every source byte are checked before the local index is
accepted. Model verification and working-copy SQLite/ordered-ID checks are the
frozen v03 implementation. No download, reuse of partial files, or retry.
"""
from pathlib import Path
import hashlib,json,os,time
from v03_assets import read,write,verify_model,verify_corpus,mounted_root

def measured(root,ledger,kind,operation):
    record={'kind':kind,'status':'running'};ledger.append(record)
    write(Path(root)/'logs/preparation.json',ledger);start=time.perf_counter()
    try:
        record['result']=operation();record['status']='ok';return record['result']
    except BaseException as exc:
        record.update(status='failed',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        record['ms']=1000*(time.perf_counter()-start)
        write(Path(root)/'logs/preparation.json',ledger)

def copy_verified_corpus(source,target,expected):
    source,target=Path(source),Path(target)
    if not (source/'pilot.sqlite').is_file() or not (source/'manifest.json').is_file():
        raise RuntimeError('Corpus mount incomplete; runtime download disabled')
    if read(source/'manifest.json')!=expected:raise RuntimeError('Corpus manifest differs')
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() or (target.parent/'manifest.json').exists():
        raise RuntimeError('Never overwrite/reuse an unverified or interrupted test copy')
    h=hashlib.sha256();count=0;source_size=(source/'pilot.sqlite').stat().st_size
    with (source/'pilot.sqlite').open('rb') as src,target.open('xb') as dst:
        for block in iter(lambda:src.read(4*1024*1024),b''):
            count+=len(block);h.update(block);dst.write(block)
        dst.flush();os.fsync(dst.fileno())
    if count!=source_size or target.stat().st_size!=count or h.hexdigest()!=expected['sqlite_sha256']:
        raise RuntimeError('Copied corpus source SHA256/size differs; partial copy retained, no inference')
    # Preserve the exact mounted manifest bytes; the frozen verifier rechecks it.
    with (target.parent/'manifest.json').open('xb') as dst:dst.write((source/'manifest.json').read_bytes())
    return {'action':'sequential_copy_and_full_source_sha256','bytes_copied':count,'source_bytes':source_size,
            'source_sha256':h.hexdigest(),'mounted_SQLite_queries':0,'working_verification_required':True}

def prepare_mounted_assets(root,mounts,ledger):
    root=Path(root);model=mounted_root(mounts['model'],'model');corpus=mounted_root(mounts['corpus'],'corpus')
    measured(root,ledger,'verify_model',lambda:verify_model(model,read(root/'configs/model_asset_lock.json'),
        require_manifest=mounts['model'].get('source_type','dataset')!='kaggle_model'))
    expected=read(root/'data/corpus/expected_manifest.json');target=root/'data/corpus/pilot.sqlite'
    measured(root,ledger,'copy_and_verify_mounted_corpus_bytes',lambda:copy_verified_corpus(corpus,target,expected))
    measured(root,ledger,'verify_working_corpus',lambda:verify_corpus(target.parent,expected))
    external={'provisioning':read(root/'configs/asset_preparation_v03.json')} if (root/'configs/asset_preparation_v03.json').is_file() else {}
    for name,folder in [('model',model),('corpus',corpus)]:
        if (folder/'preparation_cost.json').is_file():
            external[name]={'scope':'user asset preparation report, separate from measured Notebook setup',
                            'mounted_root':str(folder),'report':read(folder/'preparation_cost.json')}
        if (folder/'preparation_attempts.jsonl').is_file():
            external.setdefault(name,{})['all_reported_attempts']=[json.loads(s) for s in (folder/'preparation_attempts.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
    write(root/'logs/external_asset_preparation.json',external)
    return model,corpus/'pilot.sqlite',target
