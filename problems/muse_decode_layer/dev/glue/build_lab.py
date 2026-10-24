from pathlib import Path
import ast,copy
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).parent
base=(ROOT/'problems/muse_decode_layer/cand_r3_best5.py').read_text();tree=ast.parse(base)
funcs={'_pdl_start','_pdl_wait','_launch','_group_barrier','_boundary_kernel','_chunks','boundary','_route_norm_kernel','route_and_norm'}
assigns={'HIDDEN','D_EXPERT','N_EXPERTS','TOP_K','LOCAL_EXPERTS','POST_EPS','RMS_EPS','ROUTER_MULT','PDL'}
nodes=[]
for n in tree.body:
 if isinstance(n,ast.FunctionDef) and n.name in funcs:nodes.append(n)
 if isinstance(n,ast.Assign) and any(isinstance(k,ast.Name) and k.id in assigns for k in n.targets):nodes.append(n)
class Control(ast.NodeTransformer):
 def visit_Name(self,n):
  if n.id in ('boundary','route_and_norm'):n.id='control_'+n.id
  return n
 def visit_FunctionDef(self,n):
  if n.name in ('boundary','route_and_norm'):n.name='control_'+n.name
  return self.generic_visit(n)
nodes=Control().visit(ast.Module(nodes,type_ignores=[])).body
extra=ast.parse((HERE/'kernels.py.inc').read_text());nodes+=extra.body
names={n.name for n in nodes if isinstance(n,ast.FunctionDef)}
names|={k.id for n in nodes if isinstance(n,ast.Assign) for k in n.targets if isinstance(k,ast.Name)}
class Prefix(ast.NodeTransformer):
 def visit_Name(self,n):
  if n.id in names:n.id='glue_'+n.id.lstrip('_')
  return n
 def visit_FunctionDef(self,n):
  n.name='glue_'+n.name.lstrip('_');return self.generic_visit(n)
 def visit_Global(self,n):
  n.names=['glue_'+x.lstrip('_') if x in names else x for x in n.names];return n
module=Prefix().visit(ast.Module(nodes,type_ignores=[]));ast.fix_missing_locations(module)
payload='import torch\nimport triton\nimport triton.language as tl\n\n'+ast.unparse(module)+'\n\n'+(HERE/'lab_body.py.inc').read_text()
out=ROOT/'prof/lab/decode_glue.py';out.write_text(payload);print(out)
