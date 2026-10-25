"""Build a code-only Git archive only after index and complete-history approval."""
from pathlib import Path
import argparse,hashlib,json,subprocess
from check_publication_scope import ROOT,check_index,check_history,inspect_payload
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise RuntimeError('Refusing to overwrite an existing release archive')
    for label,result in [('index',check_index()),('history',check_history())]:
        if result['issues']:raise RuntimeError('Publication blocked by '+label+' manuscript check')
    status=subprocess.run(['git','-C',str(ROOT),'status','--porcelain'],capture_output=True,text=True,check=True).stdout
    if status.strip():raise RuntimeError('Commit and review the publication tree before packaging')
    head=subprocess.run(['git','-C',str(ROOT),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
    archive=subprocess.run(['git','-C',str(ROOT),'archive','--format=zip','--prefix=CAGR/','HEAD'],capture_output=True,check=True).stdout
    if inspect_payload(a.output.name,archive):raise RuntimeError('Archive manuscript check failed')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_bytes(archive)
    checksum=hashlib.sha256(archive).hexdigest();a.output.with_suffix(a.output.suffix+'.sha256').write_text(checksum+'  '+a.output.name+'\n')
    print(json.dumps({'archive':a.output.name,'commit':head,'sha256':checksum,'publication_scope':'passed'}))
if __name__=='__main__':main()
