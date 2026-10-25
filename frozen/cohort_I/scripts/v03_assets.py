"""Offline asset verification; no Hub or Kaggle client imports."""
from pathlib import Path
import hashlib, json, shutil, sqlite3, time

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')

def verify_model(folder, lock, require_manifest=True):
    folder = Path(folder)
    if not folder.is_dir():
        raise RuntimeError(f'Model mount missing: {folder}. Mount the complete locked snapshot and set model.root.')
    if (folder/'adapter_config.json').exists():
        raise RuntimeError('Unexpected adapter_config.json in the model asset; use the exact locked base snapshot.')
    records = {}
    for name, expected in lock['files'].items():
        p = folder / name
        if not p.is_file() or p.stat().st_size != expected['bytes'] or sha(p) != expected['sha256']:
            raise RuntimeError(f'Model missing or SHA256 mismatch: {name}. Re-prepare the locked asset; runtime download is disabled.')
        records[name] = {'bytes': expected['bytes'], 'sha256': expected['sha256']}
    index = read(folder / 'model.safetensors.index.json')
    if set(index['weight_map'].values()) != {n for n in records if n.endswith('.safetensors')}:
        raise RuntimeError('Model shard index does not match the locked two shards.')
    if require_manifest:
        p = folder / 'asset_manifest.json'
        if not p.is_file():
            raise RuntimeError('Missing model asset_manifest.json. Run prepare_v03_assets.py model once before uploading.')
        manifest = read(p)
        if manifest.get('repo_id') != lock['repo_id'] or manifest.get('revision') != lock['revision'] or manifest.get('files') != records:
            raise RuntimeError('Model asset manifest differs from the trusted lock.')
    return records

def verify_corpus(folder, expected):
    folder = Path(folder)
    p, m = folder / 'pilot.sqlite', folder / 'manifest.json'
    if not p.is_file() or not m.is_file():
        raise RuntimeError(f'Corpus mount incomplete: {folder}. Upload existing pilot.sqlite and manifest.json together.')
    actual = read(m)
    if actual != expected or sha(p) != expected['sqlite_sha256']:
        raise RuntimeError('Corpus manifest or SHA256 mismatch; the frozen corpus must be reused.')
    db = sqlite3.connect(p.resolve().as_uri() + '?mode=ro', uri=True)
    try:
        if db.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
            raise RuntimeError('SQLite quick_check failed.')
        h, count = hashlib.sha256(), 0
        for (docid,) in db.execute('SELECT doc_id FROM passages ORDER BY rowid'):
            count += 1
            h.update((docid + '\n').encode())
        if count != expected['documents'] or h.hexdigest() != expected['ordered_document_ids_sha256']:
            raise RuntimeError('Corpus ordered document IDs differ from the frozen manifest.')
    finally:
        db.close()
    return {'bytes': p.stat().st_size, 'sha256': expected['sqlite_sha256'], 'documents': count,
            'ordered_document_ids_sha256': h.hexdigest()}

def mounted_root(item, name):
    source_type=item.get('source_type','dataset')
    if source_type=='dataset':
        identity=item.get('dataset_id')
    elif source_type=='kaggle_model' and name=='model':
        import re
        identity=item.get('model_handle')
        if not identity or not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/[1-9][0-9]*',identity):
            raise RuntimeError('Provide the actual versioned Kaggle model handle (owner/model/framework/variation/version).')
        if item.get('dataset_id'):
            raise RuntimeError('A Kaggle Model source must not be recorded as a Dataset ID.')
    else:
        raise RuntimeError('Unsupported mounted source type: '+source_type)
    if not identity or not item.get('root'):
        raise RuntimeError(f'Fill the actual {name} source identity and mounted root. No source ID is inferred.')
    p = Path(item['root']).resolve()
    if not p.is_relative_to(Path('/kaggle/input').resolve()):
        raise RuntimeError(f'{name}.root must be an explicit /kaggle/input mount.')
    return p

def prepare_mounted_assets(root, mounts, ledger):
    root = Path(root)
    model = mounted_root(mounts['model'], 'model')
    corpus = mounted_root(mounts['corpus'], 'corpus')
    for name, fn in [('verify_model', lambda: verify_model(model, read(root/'configs/model_asset_lock.json'),require_manifest=mounts['model'].get('source_type','dataset')!='kaggle_model')),
                     ('verify_mounted_corpus', lambda: verify_corpus(corpus, read(root/'data/corpus/expected_manifest.json')))]:
        start = time.perf_counter()
        record = {'kind': name, 'status': 'running'}
        ledger.append(record)
        try:
            record['result'] = fn()
            record['status'] = 'ok'
        except Exception as exc:
            record.update(status='failed',error_type=type(exc).__name__,error=str(exc))
            raise
        finally:
            record['ms'] = 1000 * (time.perf_counter() - start)
            write(root/'logs/preparation.json', ledger)
    target = root/'data/corpus/pilot.sqlite'
    start = time.perf_counter()
    record = {'kind': 'copy_index_to_working', 'bytes': (corpus/'pilot.sqlite').stat().st_size, 'status': 'running'}
    ledger.append(record)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            verify_corpus(target.parent, read(root/'data/corpus/expected_manifest.json'))
            record.update(action='reuse_verified_copy_after_interrupted_preparation',bytes_copied=0)
        else:
            shutil.copyfile(corpus/'pilot.sqlite', target)
            shutil.copyfile(corpus/'manifest.json', target.parent/'manifest.json')
            record.update(action='copy',bytes_copied=target.stat().st_size)
        record['status'] = 'ok'
    except Exception as exc:
        record.update(status='failed',error_type=type(exc).__name__,error=str(exc),bytes_copied=target.stat().st_size if target.exists() else 0)
        raise
    finally:
        record['ms'] = 1000*(time.perf_counter()-start)
        write(root/'logs/preparation.json', ledger)
    start = time.perf_counter()
    record = {'kind': 'verify_working_corpus', 'status': 'running'}
    ledger.append(record)
    try:
        record['result'] = verify_corpus(target.parent, read(root/'data/corpus/expected_manifest.json'))
        record['status'] = 'ok'
    except Exception as exc:
        record.update(status='failed',error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        record['ms'] = 1000*(time.perf_counter()-start)
        write(root/'logs/preparation.json', ledger)
    external={'provisioning':read(root/'configs/asset_preparation_v03.json')} if (root/'configs/asset_preparation_v03.json').is_file() else {}
    for name,folder in [('model',model),('corpus',corpus)]:
        if (folder/'preparation_cost.json').is_file():
            external[name]={'scope':'user asset preparation report, separate from measured Notebook setup',
                            'mounted_root':str(folder),'report':read(folder/'preparation_cost.json')}
        if (folder/'preparation_attempts.jsonl').is_file():
            external.setdefault(name,{})['all_reported_attempts']=[json.loads(s) for s in (folder/'preparation_attempts.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
    write(root/'logs/external_asset_preparation.json',external)
    return model, corpus/'pilot.sqlite', target

def find_kaggle_model_root(handle, input_root=Path('/kaggle/input')):
    """Resolve a mounted version by its exact framework/variation/version suffix."""
    owner,model,framework,variation,version=handle.split('/')
    suffix=(model,framework.lower(),variation,version)
    matches=[]
    for candidate in Path(input_root).rglob('config.json'):
        p=candidate.parent
        if tuple(p.parts[-4:])==suffix:matches.append(p)
    if len(matches)!=1:
        raise RuntimeError('Expected exactly one mounted version for '+handle+'; found '+str(matches))
    return matches[0]
