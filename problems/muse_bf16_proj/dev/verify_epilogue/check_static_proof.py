"""A proof must not hide a new CUDA error or a changed cached shader."""
import ast
import importlib.util
import json
from pathlib import Path
import tempfile

HERE=Path(__file__).resolve().parent
p=HERE.parent/'glue_preload'
spec=importlib.util.spec_from_file_location('p7screen',HERE/'screen.py')
screen=importlib.util.module_from_spec(spec);spec.loader.exec_module(screen)
tree=ast.parse((HERE/'screen.py').read_text())
fn=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='compile_or_reuse')
ns=dict(screen.__dict__,proof=json.loads((p/'static_proof.json').read_text()),original_static=screen._ORIGINAL_HOPPER_COMPILE)
exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'<proof-check>','exec'),ns)
source=(p/'d1_trial.py').read_text()
bad='#include <cuda_runtime.h>\n__global__ void invalid(){gp_deliberate_compile_error();}\n'
with tempfile.TemporaryDirectory(prefix='p7_proof_test_') as tmp:
    path=Path(tmp)/'new_cuda.py';path.write_text(source+'\nBAD_CUDA='+repr(bad)+'\n')
    rows=ns['compile_or_reuse'](path);by={r['name']:r for r in rows}
    assert not by['BAD_CUDA']['ok'] and all(r['ok'] for r in rows if r['name']!='BAD_CUDA')
    path=Path(tmp)/'changed_cuda.py';nodes=ast.parse(source)
    node=next(n for n in nodes.body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='GF_CUDA')
    lines=source.splitlines(True);changed=''.join(lines[:node.lineno-1])+'GF_CUDA='+repr(bad)+'\n'+''.join(lines[node.end_lineno:])
    path.write_text(changed);rows=ns['compile_or_reuse'](path);by={r['name']:r for r in rows}
    assert not by['GF_CUDA']['ok'] and all(r['ok'] for r in rows if r['name']!='GF_CUDA')
print('PASS new CUDA errors and changed cached CUDA errors both compile and fail')
