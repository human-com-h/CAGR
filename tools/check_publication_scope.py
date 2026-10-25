"""Fail closed on manuscript files in the index, reachable Git history or archives.

This policy has no automatic exemption. Paper publication requires a separate,
explicit instruction from the authors; a code-release request is not sufficient.
"""
from pathlib import Path,PurePosixPath
import argparse,gzip,io,json,re,subprocess,tarfile,zipfile

ROOT=Path(__file__).resolve().parents[1]
DENIED_SUFFIXES={'.pdf','.tex','.bib','.doc','.docx','.odt','.rtf','.dvi','.ps'}
DENIED_DIRS={'paper','papers','manuscript','manuscripts','preprint','preprints','submission','submissions','rebuttal','rebuttals','review_response','revision_draft','论文','手稿','投稿','返修','审稿'}
CODE_SUFFIXES={'.py','.cpp','.cu','.h','.yml','.yaml','.json','.toml'}
MAX_BYTES=64*1024*1024

def path_issue(name):
    path=PurePosixPath(name.replace('!','/').replace('\\','/'));parts={p.casefold() for p in path.parts}
    if parts&DENIED_DIRS:return 'internal-paper-directory'
    if path.suffix.casefold() in DENIED_SUFFIXES:return 'document-file-type'
    if re.search(r'(?:^|[_\-])(manuscript|preprint|rebuttal|submission|review[_\-]?response|revision[_\-]?draft)(?:[_.\-]|$)',path.name,re.I):return 'internal-document-name'
    return None

def inspect_payload(name,payload,depth=0):
    issues=[];reason=path_issue(name)
    if reason:issues.append({'path':name,'reason':reason})
    if len(payload)>MAX_BYTES:raise ValueError('Publication payload exceeds inspection budget')
    if payload.lstrip()[:5]==b'%PDF-':issues.append({'path':name,'reason':'pdf-content'})
    if re.search(rb'(?m)^\s*\\documentclass(?:\[|\{)',payload) and re.search(rb'(?m)^\s*\\begin\{document\}',payload):issues.append({'path':name,'reason':'complete-latex-document'})
    suffix=PurePosixPath(name).suffix.casefold()
    if suffix not in CODE_SUFFIXES:
        headings=[bool(re.search(pattern,payload)) for pattern in [
            rb'(?mi)^#{1,3}\s+(?:\d+[.)]?\s+)?Introduction\s*$',
            rb'(?mi)^#{1,3}\s+(?:\d+[.)]?\s+)?Conclusions?\s*$',
            rb'(?mi)^#{1,3}\s+(?:\d+[.)]?\s+)?References\s*$',
            rb'(?mi)^#{1,3}\s+Appendix\b']]
        if sum(headings)>=3:issues.append({'path':name,'reason':'complete-paper-structure'})
    if depth>4:raise ValueError('Nested archive limit exceeded')
    stream=io.BytesIO(payload)
    if zipfile.is_zipfile(stream):
        with zipfile.ZipFile(stream) as z:
            members=z.infolist()
            if len(members)>5000 or sum(m.file_size for m in members)>MAX_BYTES:raise ValueError('Archive exceeds inspection budget')
            if any(m.filename=='word/document.xml' for m in members):issues.append({'path':name,'reason':'word-document-content'})
            for member in members:
                if not member.is_dir():issues.extend(inspect_payload(name+'!'+member.filename,z.read(member),depth+1))
    elif name.lower().endswith(('.tar','.tar.gz','.tgz','.tar.bz2','.tar.xz')):
        with tarfile.open(fileobj=stream,mode='r:*') as t:
            members=t.getmembers()
            if len(members)>5000 or sum(m.size for m in members)>MAX_BYTES:raise ValueError('Archive exceeds inspection budget')
            for member in members:
                if member.isfile():issues.extend(inspect_payload(name+'!'+member.name,t.extractfile(member).read(),depth+1))
    return issues

def git(*args,root=ROOT):
    r=subprocess.run(['git','-C',str(root),*args],capture_output=True,timeout=40)
    if r.returncode:raise RuntimeError('Git inspection failed: '+r.stderr.decode(errors='replace')[:300])
    return r.stdout

def check_index(root=ROOT):
    paths=git('ls-files','-z',root=root).decode().split('\0');issues=[];count=0
    for path in filter(None,paths):
        data=git('show',':'+path,root=root);issues.extend(inspect_payload(path,data));count+=1
    return {'files':count,'issues':issues}

def check_history(root=ROOT):
    lines=git('rev-list','--objects','--all',root=root).decode().splitlines();mapping={}
    for line in lines:
        p=line.split(' ',1)
        if len(p)==2:mapping.setdefault(p[0],[]).append(p[1])
    # Batch only type/size metadata. Manuscript contents never reach tool output.
    ids=list(mapping);proc=subprocess.run(['git','-C',str(root),'cat-file','--batch-check=%(objectname) %(objecttype) %(objectsize)'],input=('\n'.join(ids)+'\n').encode(),capture_output=True,timeout=40)
    if proc.returncode:raise RuntimeError('Object metadata inspection failed')
    issues=[];count=0
    for line in proc.stdout.decode().splitlines():
        oid,kind,size=line.split()
        if kind!='blob':continue
        if int(size)>MAX_BYTES:raise ValueError('Historical blob exceeds inspection budget')
        raw=git('cat-file','blob',oid,root=root);count+=1
        for name in mapping[oid]:issues.extend(inspect_payload(name,raw))
    return {'blobs':count,'issues':issues}

def main():
    p=argparse.ArgumentParser();p.add_argument('--index',action='store_true');p.add_argument('--history',action='store_true');p.add_argument('--archive',type=Path,action='append',default=[]);p.add_argument('--root',type=Path,default=ROOT);a=p.parse_args()
    if not any([a.index,a.history,a.archive]):a.index=a.history=True
    results={}
    if a.index:results['index']=check_index(a.root)
    if a.history:results['history']=check_history(a.root)
    for archive in a.archive:results['archive:'+archive.name]={'issues':inspect_payload(archive.name,archive.read_bytes())}
    issues=[issue for r in results.values() for issue in r['issues']]
    print(json.dumps({'status':'blocked' if issues else 'passed','checks':results},ensure_ascii=False,indent=2))
    if issues:raise SystemExit(1)
if __name__=='__main__':main()
