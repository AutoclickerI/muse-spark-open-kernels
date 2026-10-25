"""Build producer-glue paired/timeline payload with the verified selected S11 setup."""
import ast
import hashlib
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
sys.path.insert(0,str(HERE))
from patch import patch


def pieces(src,names):
    ls=src.splitlines();out=[]
    for n in ast.parse(src).body:
        if isinstance(n,ast.FunctionDef) and n.name in names:
            out.append('\n'.join(ls[n.lineno-1:n.end_lineno]))
    assert len(out)==len(names)
    return '\n\n'.join(out)


source=(REPO/'problems/muse_decode_layer/cand_r3_best7_s11_final.py').read_text()
assert hashlib.sha256(source.encode()).hexdigest().startswith('acda6c62')
timeline=REPO/'problems/muse_decode_layer/dev/timeline'
ab=(REPO/'problems/muse_decode_layer/dev/s10/ablate_lab.py.inc').read_text()
helpers=pieces(ab,('ab_copy','ab_graph','ab_tensors','ab_gate','ab_time','ab_summary'))
tl_helpers=pieces((timeline/'lab_body.py').read_text(),('tl_activity','tl_times','tl_union','tl_summarize','tl_node_bytes'))
for stats,hybrid in ((False,False),(True,False),(True,True)):
    body='import base64,gc,hashlib,json,linecache,lzma,math,statistics,time,types\nimport torch\nfrom torch.utils.cpp_extension import load_inline\n'
    body+='PG_SOURCE = '+repr(patch(source,stats=stats,hybrid=hybrid))+'\nTL_ACTIVITY = '+repr((timeline/'activity.cpp').read_text())+'\nTL_REPS=5\n'
    body+=(timeline/'selected_prepare.py').read_text()+'\n\n'+helpers+'\n\n'+tl_helpers+'\n\n'
    body+=(HERE/'lab_body.py').read_text()
    compile(body,'lab_producer_glue','exec')
    path=HERE/('lab_producer_hybrid.py' if hybrid else 'lab_producer_stats.py' if stats else 'lab_producer_glue.py');path.write_text(body)
    print(path,hashlib.sha256(body.encode()).hexdigest())
