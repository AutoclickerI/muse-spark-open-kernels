"""Apply the eea9d87b N16G4 arm onto frozen S12 final, retaining real math.

Usage: python3 patch_s12_light.py source.py output.py
The pinned light builder and PERMW/wide siblings must be next to this script.
"""
import ast
import hashlib
from pathlib import Path
import sys
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H))
import light_build
import patch_s12_permw

BASE='240248b2c27c890bf5055a77e87d2fc9c4429b747a04d714e51d5f67e936bb3e'
BUILDER='4e9afa80994deb60c7ff3f0e563063f79b76dbc8eb1f6704961e5b96d307b64a'


def patch(text):
    assert hashlib.sha256(text.encode()).hexdigest()==BASE
    assert hashlib.sha256((H/'light_build.py').read_bytes()).hexdigest()==BUILDER
    light_build.main()
    lab=(H/'light_payload.py').read_text()
    assert hashlib.sha256(lab.encode()).hexdigest()=='ce1ecf2eb5e7478ac59c0d99ab8a33f6a35c2e32182b4e272b1f580ba1b261e6'
    s=patch_s12_permw.patch(text)
    tree=ast.parse(s);lines=s.splitlines(keepends=True);repls=[]
    cuda=light_build.value(lab,'OP_CUDA')[3]
    node=next(n for n in ast.parse(lab).body if isinstance(n,ast.FunctionDef) and n.name=='nf_light_plan')
    plan=ast.get_source_segment(lab,node).replace('def nf_light_plan(data,base,native,streams,light_grid):','def nf_plan(data,base,native,streams):\n    light_grid=4')
    for n in tree.body:
        if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='NF_CUDA':repls.append((n.lineno,n.end_lineno,'NF_CUDA='+repr(cuda)+'\n'))
        if isinstance(n,ast.FunctionDef) and n.name=='nf_plan':repls.append((n.lineno,n.end_lineno,plan+'\n'))
    assert len(repls)==2
    for a,b,code in sorted(repls,reverse=True):lines[a-1:b]=[code]
    s=''.join(lines)
    s=light_build.once(s,"state['nf_streams']=(torch.cuda.Stream(device=device),torch.cuda.Event(),torch.cuda.Event())",
                      "state['nf_streams']=(torch.cuda.Stream(device=device),torch.cuda.Event(),torch.cuda.Event(),torch.cuda.Stream(device=device),torch.cuda.Event())")
    compile(s,'<S12 N16G4>','exec');return s


if __name__=='__main__':
    src,dst=map(Path,sys.argv[1:]);out=patch(src.read_text());dst.write_text(out)
    print(dst,hashlib.sha256(out.encode()).hexdigest())
