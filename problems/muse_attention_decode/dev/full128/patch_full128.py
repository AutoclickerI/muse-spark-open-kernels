"""Replace only frozen S11's full-verify native vq8 cell; paired H200 evidence must select the row settings."""
import ast,hashlib,sys,importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent
_spec=importlib.util.spec_from_file_location('_p5_full128_source',HERE/'build.py')
build=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(build)
BASE_SHA='acda6c621b1f3b885f2111ea1b4482ae1d733e2f2493db2b7f0b852ccf66dfb7'

def replace_cell(src,stages=(4,4)):
    assert len(stages)==2 and all(x in (3,4) for x in stages)
    cuda=build.source();tag=hashlib.sha256(cuda.encode()).hexdigest()[:12]
    lines=src.splitlines(True)
    for n in reversed(ast.parse(src).body):
        if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='VQ8_CUDA':lines[n.lineno-1:n.end_lineno]=['VQ8_CUDA='+repr(cuda)+'\n']
        if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='_VQ8_ROWS':lines[n.lineno-1:n.end_lineno]=['_VQ8_ROWS='+repr({(16,8,False):(stages[0],8),(32,8,False):(stages[1],8)})+'\n']
    out=''.join(lines);old='muse_vq8_c262e4c5f9c3_{arch}';assert old in out
    return out.replace(old,'muse_vq8_'+tag+'_{arch}')

def patch(src,stages=(4,4)):
    assert hashlib.sha256(src.encode()).hexdigest()==BASE_SHA,'apply onto frozen S11 acda6c62'
    return replace_cell(src,stages)

if __name__=='__main__':
    out=patch(Path(sys.argv[1]).read_text());p=Path(sys.argv[2]);p.write_text(out);print(p,hashlib.sha256(out.encode()).hexdigest())
