"""Scan only publication files; never read credentials or generated raw resources."""
from pathlib import Path
import hashlib,json,re
ROOT=Path(__file__).resolve().parents[1]
EXCLUDED={'.git','resources','reproduced','runs','__pycache__','.pytest_cache','.venv'}
PATTERNS=[rb'gh[pousr]_[A-Za-z0-9]{30,}',rb'github_pat_[A-Za-z0-9_]{30,}',rb'hf_[A-Za-z0-9]{30,}',rb'sk-[A-Za-z0-9]{32,}',rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',rb'AKIA[A-Z0-9]{16}',rb'[CE]:[\\/]+Users[\\/]+',rb'E:[\\/]+CAGR']
def files():
    return sorted(p for p in ROOT.rglob('*') if p.is_file() and not any(v in EXCLUDED or v.startswith('.venv') for v in p.relative_to(ROOT).parts))
def main():
    issues=[];checked=0
    for p in files():
        raw=p.read_bytes();checked+=1
        if p.stat().st_size>20*1024*1024:issues.append([p.relative_to(ROOT).as_posix(),'oversize publication file'])
        if p.name in ['hosts.yml','.env','questions.key'] or p.suffix in ['.sqlite','.safetensors','.pem','.key']:issues.append([p.relative_to(ROOT).as_posix(),'restricted resource filename'])
        if any(re.search(pattern,raw) for pattern in PATTERNS):issues.append([p.relative_to(ROOT).as_posix(),'credential/private-path pattern; content withheld'])
    if issues:raise RuntimeError(json.dumps(issues))
    print(json.dumps({'security_scan':'passed','publication_files_checked':checked,'excluded_generated_resources':True}))
if __name__=='__main__':main()
