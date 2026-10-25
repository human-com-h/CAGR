"""Check small pinned provider metadata files; no model/data payload downloads."""
from pathlib import Path
import json,urllib.request
ROOT=Path(__file__).resolve().parents[1]
def main():
    sources=json.loads((ROOT/'configs/source_lock.json').read_text())['sources'];checked=[]
    for name in ['generator','hotpotqa','triviaqa']:
        s=sources[name];prefix='' if name=='generator' else 'datasets/'
        url=f"https://huggingface.co/{prefix}{s['repo_id']}/resolve/{s['revision']}/README.md"
        with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'CAGR-resource-check'}),timeout=30) as r:
            if r.status!=200 or not r.read(128):raise RuntimeError('Pinned metadata unavailable: '+name)
        checked.append(url)
    print(json.dumps({'provider_links_checked':checked,'status':'passed'}))
if __name__=='__main__':main()
