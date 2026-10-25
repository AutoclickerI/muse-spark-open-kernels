"""Build the self-contained balanced GEMV issue-path lab."""
import ast
import hashlib
from pathlib import Path

HERE=Path(__file__).resolve().parent
EP=HERE.parent/'verify_epilogue'
src=(EP/'sweep_body.py').read_text()
lines=src.splitlines();helper=[]
for n in ast.parse(src).body:
    if isinstance(n,ast.FunctionDef) and n.name in ('ve_capture','ve_timing'):
        helper.append('\n'.join(lines[n.lineno-1:n.end_lineno]))
assert len(helper)==2
out=(HERE/'issue_kernel.py').read_text()+'\nimport statistics\n'+'\n\n'.join(helper)+'\n\n'+(HERE/'issue_body.py').read_text()
path=HERE/'lab_gemv_issue.py'
path.write_text(out)
print(path,hashlib.sha256(out.encode()).hexdigest())
layer=(EP/'lab_verify_layer.py').read_text().split('def lab(deadline, weights, state=None):',1)[0]
layer+='\n'+(HERE/'smallm_prepare.py').read_text()+'\n'+(HERE/'issue_kernel.py').read_text()+'\n'+(HERE/'issue_layer_body.py').read_text()
path=HERE/'lab_gemv_issue_layer.py';path.write_text(layer)
print(path,hashlib.sha256(layer.encode()).hexdigest())
