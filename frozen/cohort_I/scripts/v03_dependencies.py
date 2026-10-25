"""Explicit optional installation ONLY from hash-verified local wheels."""
from pathlib import Path
import importlib.metadata as md, platform, subprocess, sys, time
from v03_assets import read, write, sha, mounted_root

def check_versions(requirements):
    errors=[]
    for text in Path(requirements).read_text().splitlines():
        if not text.strip(): continue
        name,version=text.strip().split('==')
        try: actual=md.version(name)
        except md.PackageNotFoundError: actual='missing'
        if actual!=version: errors.append(f'{name}: expected {version}, observed {actual}')
    if errors:
        raise RuntimeError('Locked runtime dependencies unavailable. Prepare and mount offline wheels; no runtime download is allowed. '+'; '.join(errors))

def install_wheels(root, item):
    root=Path(root)
    folder=mounted_root(item,'wheels')
    manifest=read(folder/'wheel_manifest.json')
    start=time.perf_counter()
    record={'kind':'offline_dependency_install','status':'running','source':str(folder),'dataset_id':item['dataset_id']}
    path=root/'logs/dependency_preparation.json'
    try:
        if platform.system()!=manifest['platform'] or platform.python_version().rsplit('.',1)[0]!=manifest['python_major_minor']:
            raise RuntimeError('Wheel environment does not match the recorded preparation environment.')
        for name,version in manifest['base_dependencies'].items():
            if md.version(name)!=version:
                raise RuntimeError(f'Base dependency differs from wheel preparation: {name}. Prepare a matching asset; do not silently upgrade.')
        wheels=[]
        for name,e in manifest['files'].items():
            if Path(name).name!=name or not name.endswith('.whl'):
                raise RuntimeError('Invalid wheel inventory path.')
            p=folder/name
            if not p.is_file() or p.stat().st_size!=e['bytes'] or sha(p)!=e['sha256']:
                raise RuntimeError('Wheel hash mismatch: '+name)
            wheels.append(str(p))
        # Resolution was frozen during preparation; never resolve a new network dependency here.
        process=subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps',*wheels],capture_output=True,text=True)
        (root/'logs/dependency_install.log').write_text(process.stdout+process.stderr)
        if process.returncode: raise RuntimeError('Offline wheel installation failed; inspect logs/dependency_install.log.')
        for name,version in manifest['requirements'].items():
            if md.version(name)!=version: raise RuntimeError('Installed wheel version differs: '+name)
        check_versions(root/'configs/requirements-runtime-v03.txt')
        record.update(status='ok',wheel_manifest_sha256=sha(folder/'wheel_manifest.json'),wheel_count=len(wheels))
    except Exception as exc:
        record.update(status='failed',error_type=type(exc).__name__,error=str(exc))
        raise
    finally:
        record['ms']=1000*(time.perf_counter()-start)
        write(path,record)
