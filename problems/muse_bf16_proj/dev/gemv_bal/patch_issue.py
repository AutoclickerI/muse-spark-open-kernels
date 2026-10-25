"""Apply only paired whole-layer winners of the GEMV issue-path experiment."""
import ast
import hashlib
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROWS={}
TRIAL_ROWS={**{(t,4608,8192):(6,(128,2,0,0)) for t in (1,8,16,32)},
            (32,8192,4096):(2,(128,4,0,1)),
            **{(t,1280,8192):(10,(128,4,0,1)) for t in (1,16)}}


def block():
    src=(HERE/'issue_kernel.py').read_text();lines=src.splitlines();out=[]
    for n in ast.parse(src).body:
        if isinstance(n,ast.FunctionDef) and n.name in ('_gi_kernel','gi_run'):
            first=min([d.lineno for d in n.decorator_list]+[n.lineno])
            out.append('\n'.join(lines[first-1:n.end_lineno]))
    assert len(out)==2
    return '\n\n'.join(out)


def patch(src,rows=None):
    rows=ROWS if rows is None else rows
    anchor='def _gb_gemv(x, w, choice):\n'
    assert src.count(anchor)==1
    src=src.replace(anchor,anchor+"    hit=GI_ROWS.get((x.shape[0],w.shape[0],w.shape[1]))\n"
                    "    if hit is not None:\n        s,cfg=hit\n        return gi_run(x,w,s,cfg),s\n")
    return src+'\n\n'+block()+'\nGI_ROWS = '+repr(rows)+'\n'


if __name__=='__main__':
    s=patch(Path(sys.argv[1]).read_text(),TRIAL_ROWS if '--trial' in sys.argv[3:] else ROWS)
    Path(sys.argv[2]).write_text(s)
    print(sys.argv[2],hashlib.sha256(s.encode()).hexdigest())
