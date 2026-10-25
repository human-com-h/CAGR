"""Discover existing locked assets; never infer identities or download files.

Historical Kaggle paths are hints, not a guarantee of the next VM's layout.
Only existing sentinels under /kaggle/input with the exact configured identity
and locked content may resolve. Frozen configs remain byte-for-byte intact.
"""
from pathlib import Path,PurePosixPath
import copy,re,time
from v03_assets import read,write,sha
from v03_dependencies import install_wheels

INPUT=Path('/kaggle/input')

def _identity(item,kind,validate_hint=True):
    if kind=='model':
        value=item.get('model_handle','')
        if not isinstance(value,str) or item.get('dataset_id') or item.get('source_type')!='kaggle_model' or not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/[1-9][0-9]*',value):raise RuntimeError('Actual versioned model identity required')
    else:
        value=item.get('dataset_id','')
        if not isinstance(value,str) or item.get('source_type','dataset')!='dataset' or not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',value):raise RuntimeError('Actual Dataset ID required for '+kind)
    if not isinstance(item.get('root'),str):raise RuntimeError('Explicit configured mount hint required')
    hint=PurePosixPath(item['root'])
    if validate_hint and ('..' in hint.parts or not hint.is_relative_to(PurePosixPath('/kaggle/input'))):raise RuntimeError('Configured mount hint must stay under /kaggle/input')
    return value

def _sentinel_matches(path,kind,item,root):
    if kind=='model':
        owner,model,framework,variation,version=item['model_handle'].split('/')
        parts=path.parent.parts
        if len(parts)<4 or (parts[-4],parts[-3].lower(),parts[-2],parts[-1])!=(model,framework.lower(),variation,version):return False
        lock=read(root/'configs/model_asset_lock.json')['files']['config.json']
        if path.stat().st_size!=lock['bytes'] or sha(path)!=lock['sha256']:raise RuntimeError('Mounted model config differs from locked version')
    else:
        if item['dataset_id'].split('/')[1] not in path.parent.parts:return False
        if kind=='wheels':
            lock=root/'configs/wheel_manifest_lock_v03.json'
            if path.stat().st_size!=lock.stat().st_size or sha(path)!=sha(lock):raise RuntimeError('Wheel manifest SHA differs from the previously uploaded locked asset')
        elif read(path)!=read(root/'data/corpus/expected_manifest.json'):raise RuntimeError('Mounted corpus manifest differs from frozen corpus')
    return True

def resolve_mounts(root,input_root=INPUT):
    root=Path(root);input_root=Path(input_root).resolve();start=time.perf_counter()
    record={'status':'running','input_root':str(input_root),'configured_mounts_sha256':sha(root/'configs/mounts.json'),
        'scope':'separate offline asset preparation, no inference or downloads','discovery':{}}
    try:
        if not input_root.is_dir():raise RuntimeError('Kaggle input mount is missing; attach the registered assets')
        record['top_level_entries']=sorted(p.name for p in input_root.iterdir())
        mounts=read(root/'configs/mounts.json');resolved=copy.deepcopy(mounts)
        for kind,sentinel in [('wheels','wheel_manifest.json'),('corpus','manifest.json'),('model','config.json')]:
            item=mounts[kind];identity=_identity(item,kind);found=[];observed=[]
            for path in sorted(input_root.rglob(sentinel)):
                actual=path.resolve()
                if not actual.is_relative_to(input_root):raise RuntimeError('Asset sentinel symlink escapes /kaggle/input')
                if not _sentinel_matches(actual,kind,item,root):continue
                observed.append(str(actual));found.append(actual.parent)
            found=sorted(set(found));record['discovery'][kind]={'identity':identity,'configured_hint':item['root'],'matched_sentinels':observed}
            if len(found)!=1:raise RuntimeError(f'Expected exactly one existing hash-locked mount for {identity}; found {len(found)}. Attach that registered source; automatic download is disabled.')
            resolved[kind]['root']=str(found[0])
        record.update(status='ok',resolved=resolved,wheel_manifest_lock_sha256=sha(root/'configs/wheel_manifest_lock_v03.json'))
        return resolved
    except BaseException as exc:
        record.update(status='failed',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        record['ms']=1000*(time.perf_counter()-start);write(root/'logs/resolved_mounts.json',record)

def prepare_offline_dependencies(root,input_root=INPUT):
    root=Path(root);start=time.perf_counter();record={'kind':'asset_bootstrap','status':'running','download_bytes':0,'model_calls':0}
    try:
        mounts=resolve_mounts(root,input_root)
        install_wheels(root,mounts['wheels'])
        record['status']='ok';return mounts
    except BaseException as exc:
        record.update(status='failed',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        record['ms']=1000*(time.perf_counter()-start);write(root/'logs/asset_bootstrap.json',record)

def checked_resolved_mounts(root,input_root=INPUT):
    root=Path(root);input_root=Path(input_root).resolve();record=read(root/'logs/resolved_mounts.json');configured=read(root/'configs/mounts.json')
    if record.get('status')!='ok' or record['configured_mounts_sha256']!=sha(root/'configs/mounts.json'):raise RuntimeError('Missing or changed asset resolution receipt')
    result=record['resolved']
    for kind,sentinel in [('wheels','wheel_manifest.json'),('corpus','manifest.json'),('model','config.json')]:
        if _identity(configured[kind],kind)!=_identity(result[kind],kind,validate_hint=False):raise RuntimeError('Resolved source identity changed')
        path=Path(result[kind]['root']).resolve()
        if not path.is_relative_to(input_root) or not _sentinel_matches(path/sentinel,kind,configured[kind],root):raise RuntimeError('Resolved mount no longer matches locked sentinel')
    return result
