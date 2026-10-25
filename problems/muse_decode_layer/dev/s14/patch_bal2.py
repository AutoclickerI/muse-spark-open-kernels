"""Pool/router BK128 NS2 from d3666d18 / a645a795; T1 keeps S13's coded path."""
import ast
from pathlib import Path

def patch(src):
    text=Path(__file__).with_name('issue_frozen.py').read_text();lines=text.splitlines();parts=[]
    for n in ast.parse(text).body:
        if isinstance(n,ast.FunctionDef) and n.name in ('_gi_kernel','gi_run'):
            a=min([d.lineno for d in n.decorator_list]+[n.lineno]);parts.append('\n'.join(lines[a-1:n.end_lineno]))
    assert len(parts)==2
    old='def _gb_gemv(x, w, choice):\n'
    assert src.count(old)==1
    src=src.replace(old,old+"    if x.shape[0] in (8,16,32) and tuple(w.shape)==(4608,8192):\n        return gi_run(x,w,6,(128,2,0,0)),6\n")
    return src+'\n\n'+'\n\n'.join(parts)+'\n'
