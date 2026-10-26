"""Port measured CUDA edits while preserving S18's dependency protocol."""
import argparse,difflib,hashlib,importlib.util,json,sys
from pathlib import Path
import pvsum_build
H=Path(__file__).resolve().parent;R=H.parents[3];o=H/'page_ring'
ap=argparse.ArgumentParser();ap.add_argument('candidate',type=Path);a=ap.parse_args()
spec=importlib.util.spec_from_file_location('p5b_s18_source',a.candidate);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
def apply(c,patch):
    lines=patch.splitlines(True);i=2;hunks=[]
    while i<len(lines):
        assert lines[i].startswith('@@');i+=1;before=[];after=[]
        while i<len(lines) and not lines[i].startswith('@@'):
            s=lines[i];i+=1
            if s[0] in ' -':before.append(s[1:])
            if s[0] in ' +':after.append(s[1:])
        hunks.append((''.join(before),''.join(after)))
    for before,after in reversed(hunks):
        assert c.count(before)==1,(before[:100],c.count(before));c=c.replace(before,after)
    return c
result={}
for name,key in [('R8','ATTN_PORT_CUDA_SRC'),('D','VQ8_CUDA')]:
    old=getattr(m,key)
    new=apply(old,(o/'R8.cuda.patch').read_text()) if name=='R8' else pvsum_build.cuda(old)
    for text in ('mb_init(&kfull[tid],4);','mb_wait(&full[','griddepcontrol.wait;'):
        assert old.count(text)==new.count(text),(name,text)
    p=o/(name+'_S18.cuda.patch');p.write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/S18_'+key+'.cu',tofile='b/S18_'+key+'.cu')))
    (o/(name+'_S18.cu')).write_text(new)
    result[name]=dict(S18_source=str(a.candidate),S18_source_sha256=hashlib.sha256(a.candidate.read_bytes()).hexdigest(),
        before_cuda_sha256=hashlib.sha256(old.encode()).hexdigest(),after_cuda_sha256=hashlib.sha256(new.encode()).hexdigest(),
        patch=str(p.relative_to(R)),patch_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),existing_dependency_counts_preserved=True,
        correctness='S15 arm has native TASK proof; S18 composition remains P4 validation')
(o/'S18_portability.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
