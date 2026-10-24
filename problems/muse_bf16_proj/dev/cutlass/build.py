"""Build self-contained lab and decode drop-in from the same CUTLASS source."""
from pathlib import Path
import ast
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]

def format_source(name,value):
    return name+' = (\n'+''.join('    '+repr(value[i:i+96])+'\n' for i in range(0,len(value),96))+')\n'

def build():
    source=(HERE/'kernel.cu').read_text()
    core=(HERE/'runtime.py.inc').read_text()+'\n'+format_source('VGC_SOURCE',source)
    graph=(HERE.parent/'verify/library.py').read_text()
    pairs=(HERE.parent/'verify/lab_library_tail.py').read_text().split('\ndef lab(')[0]
    common=core+'\n'+graph+'\n'+pairs
    # Preserve exact source strings while respecting serving's 400-char line cap.
    tree=ast.parse(common);lines=common.splitlines(keepends=True)
    for node in reversed(tree.body):
        if isinstance(node,ast.Assign) and isinstance(node.value,ast.Constant) and isinstance(node.value.value,str):
            if max(map(len,lines[node.lineno-1:node.end_lineno]))>400:
                lines[node.lineno-1:node.end_lineno]=[format_source(node.targets[0].id,node.value.value)]
    common=''.join(lines)
    graph_only='import ctypes as ct\nimport time,statistics\n'+graph[
        graph.index('class _PgGraph:'):graph.index('PG_RAW =')]
    drop=HERE/'projection_cutlass.py'
    drop.write_text(core+'\n'+graph_only+'\n'+pairs+'\n'+(HERE/'install.py.inc').read_text())
    lab=ROOT/'prof/lab/verify_proj_cutlass.py'
    lab.write_text(common+'\n'+(HERE/'gemv.py.inc').read_text()+'\n'+(HERE/'lab.py.inc').read_text())
    print(drop);print(lab)

if __name__=='__main__':build()
