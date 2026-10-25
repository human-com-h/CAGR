"""Independent mount-only repair; frozen v3 scientific runner unchanged."""
import time,zipfile
import run_online_v04_v3 as entry
from v03_assets import write
from v04_v3_mounts import checked_resolved_mounts

ORIGINAL_PREPARE=entry.base.prepare_mounted_assets

def archive(root=entry.base.ROOT,label='snapshot'):
    path=entry.archive(root,label)
    with zipfile.ZipFile(path,'a',zipfile.ZIP_DEFLATED) as z:
        name='data/corpus/expected_manifest.json'
        if name not in z.namelist():z.write(root/name,name)
    return path

def prepare_assets(root,mounts,ledger):
    # Frozen mount hints and their fingerprint are unchanged; the receipt
    # records actual existing paths separately. Full model/index SHA checks,
    # ID order verification and working-copy preparation remain mandatory.
    resolved=checked_resolved_mounts(root)
    return ORIGINAL_PREPARE(root,resolved,ledger)

def main():
    entry.base.prepare_mounted_assets=prepare_assets
    entry.main()

if __name__=='__main__':
    start=time.perf_counter();state={'status':'running'}
    try:main();state['status']='ok'
    except BaseException as exc:
        state.update(status='failed',error_type=type(exc).__name__)
        write(entry.base.ROOT/'logs/fatal.json',{'error_type':type(exc).__name__,'error':str(exc)});raise
    finally:state['ms']=1000*(time.perf_counter()-start);write(entry.base.ROOT/'logs/runtime_process.json',state)
