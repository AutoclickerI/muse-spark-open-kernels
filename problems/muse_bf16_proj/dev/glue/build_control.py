"""Extract the exact cand_r3_best projection/glue chain without MoE/attention weights."""
import ast
import hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
p=ROOT/'problems/muse_decode_layer/cand_r3_best.py';source=p.read_text();tree=ast.parse(source)
expected='812f5f369d042c9bb7f80c4e7198bc4883922539c619c10f1e49622f1fe7aad2'
assert hashlib.sha256(p.read_bytes()).hexdigest()==expected,'owner base changed; review before rebuilding'
nodes={}
for n in tree.body:
    if isinstance(n,(ast.FunctionDef,ast.ClassDef)):nodes[n.name]=n
    elif isinstance(n,ast.Assign):
        for t in n.targets:
            for v in ast.walk(t):
                if isinstance(v,ast.Name):nodes[v.id]=n
wanted=set()
def visit(name):
    if name in wanted or name not in nodes:return
    wanted.add(name)
    for v in ast.walk(nodes[name]):
        if isinstance(v,ast.Name) and isinstance(v.ctx,ast.Load):visit(v.id)
for n in ('_prefix_variant','project','boundary','route_and_norm','_seed_v9_gemv_choices',
          '_tune_projections','_tune_pdl','_tune_measured_lt','_tune_prefix'):visit(n)
rename={name:'_gbc_'+name for name in wanted}
class Rename(ast.NodeTransformer):
    def visit_Name(self,node):node.id=rename.get(node.id,node.id);return node
    def visit_FunctionDef(self,node):
        node.name=rename.get(node.name,node.name);return self.generic_visit(node)
    def visit_ClassDef(self,node):
        node.name=rename.get(node.name,node.name);return self.generic_visit(node)
selected={id(nodes[name]) for name in wanted}
body=[Rename().visit(n) for n in tree.body if id(n) in selected]
preamble='import torch\nimport torch.nn.functional as F\nfrom torch.utils.cpp_extension import load_inline\nimport triton\nimport triton.language as tl\nimport ctypes as ct\nimport time,statistics\n'
result=preamble+ast.unparse(ast.fix_missing_locations(ast.Module(body=body,type_ignores=[])))+'\n'
(HERE/'control.py.inc').write_text(result)
print('control definitions',len(body),'bytes',len(result),'base',expected)
