"""Package only our source and existing cold paired lab helpers."""
import ast
from pathlib import Path
H=Path(__file__).resolve().parent
own=H.parent/'coded_dense'
helper=(own/'lab_body.py').read_text();tree=ast.parse(helper);lines=helper.splitlines(keepends=True)
needed={'_gb_kernel','gb_run','_cl_graph','_cl_times','cl_weights'}
pre='import torch\nimport triton\nimport triton.language as tl\nimport math,statistics,time\n'
for n in tree.body:
    if isinstance(n,ast.FunctionDef) and n.name in needed:
        begin=min([n.lineno]+[x.lineno for x in n.decorator_list])-1
        pre+=''.join(lines[begin:n.end_lineno])+'\n\n'
sources={'shared':(own/'direct_body.py').read_text().split('\n"""Lossless register-A WGMMA experiment;',1)[0], 'reg':(H/'register_x.py').read_text()}
(H/'register_x_lab.py').write_text(pre+'\nSOURCES = '+repr(sources)+'\n'+(H/'lab_body.py.inc').read_text())
print(H/'register_x_lab.py')
