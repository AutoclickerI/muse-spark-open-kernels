from pathlib import Path
import ast
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]

def block():
    graph=(HERE.parent/'verify/library.py').read_text()
    graph='import ctypes as ct\n'+graph[graph.index('class _PgGraph:'):graph.index('PG_RAW =')]
    pairs=(HERE.parent/'verify/lab_library_tail.py').read_text().split('\ndef lab(')[0].split('\ndef vp_check')[0]
    parts=[(HERE/n).read_text() for n in ('kernels.py.inc','runtime.py.inc','oracle.py.inc')]
    # The counter/publish kernels use the same code on sm120 and sm90; no Hopper-only launch glue.
    body='\n'.join(parts+[graph,pairs])
    if (HERE/'install.py.inc').exists():body+='\n'+(HERE/'install.py.inc').read_text()
    p=HERE/'projection_glue.py';p.write_text(body);print(p)
    return body

def lab():
    body=block()+'\n'+(HERE/'control.py.inc').read_text()+'\n'+(HERE/'lab.py.inc').read_text()
    path=ROOT/'prof/lab/verify_projection_glue.py';path.write_text(body);print(path)
    return path

if __name__=='__main__':lab()
