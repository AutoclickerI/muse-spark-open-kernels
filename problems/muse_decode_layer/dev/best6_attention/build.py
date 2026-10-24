"""Change only best6's attention sampling/selection and layer metadata dispatch."""
from pathlib import Path
import ast,copy,hashlib,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).parent
CAND=ROOT/'problems/muse_decode_layer/cand_r3_best6.py'
REF=ROOT/'problems/muse_decode_layer/reference.py'
BOARD=ROOT/'runs/muse_decode_layer/20261023T185641Z_submit_0e94268cf5654012.py'
# Keep a pinned source, so rebuilding never accumulates edits and the expert
# portfolio remains exactly the accepted preceding construction.
FROZEN=HERE/'base_c582.py'
if not FROZEN.exists():
 original=subprocess.check_output(['git','show','313a282b8d1393bfdad928716becb53986f4dd1a:problems/muse_decode_layer/cand_r3_best6.py'],cwd=ROOT)
 assert hashlib.sha256(original).hexdigest()=='c582f3427684b69b67b13be3f1bcb9f43c68fc8b32e9556bffe4377687d85491'
 FROZEN.write_bytes(original)
s=FROZEN.read_text();assert hashlib.sha256(FROZEN.read_bytes()).hexdigest()=='c582f3427684b69b67b13be3f1bcb9f43c68fc8b32e9556bffe4377687d85491'
ref=REF.read_text();t=ast.parse(ref)
defs={n.name:n for n in t.body if isinstance(n,ast.FunctionDef)}
def assigned(n):return {x.id for target in n.targets for x in ast.walk(target) if isinstance(x,ast.Name)}
assigns={name:n for n in t.body if isinstance(n,ast.Assign) for name in assigned(n)}
needed=set();todo=['generate_input']
while todo:
 name=todo.pop()
 if name in needed:continue
 node=defs.get(name,assigns.get(name))
 if node is None:continue
 needed.add(name)
 if isinstance(node,ast.Assign):needed.update(assigned(node))
 for n in ast.walk(node):
  if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Load) and (n.id in defs or n.id in assigns) and n.id not in needed:todo.append(n.id)
class Prefix(ast.NodeTransformer):
 def __init__(self):self.scopes=[]
 def visit_Name(self,n):
  if n.id in needed and not any(n.id in scope for scope in self.scopes):n.id='_ap_'+n.id.lstrip('_')
  return n
 def visit_FunctionDef(self,n):
  if n.name in needed:n.name='_ap_'+n.name.lstrip('_')
  # Defaults/decorators use the enclosing scope; the body uses this scope.
  n.decorator_list=[self.visit(x) for x in n.decorator_list]
  n.args.defaults=[self.visit(x) for x in n.args.defaults]
  n.args.kw_defaults=[self.visit(x) if x else None for x in n.args.kw_defaults]
  local={a.arg for a in (*n.args.posonlyargs,*n.args.args,*n.args.kwonlyargs)}
  local|={x.id for x in ast.walk(n) if isinstance(x,ast.Name) and isinstance(x.ctx,ast.Store)}
  self.scopes.append(local);n.body=[self.visit(x) for x in n.body];self.scopes.pop();return n
nodes=[]
for n in t.body:
 if isinstance(n,ast.FunctionDef) and n.name in needed:nodes.append(copy.deepcopy(n))
 elif isinstance(n,ast.Assign) and assigned(n)&needed:nodes.append(copy.deepcopy(n))
module=Prefix().visit(ast.Module(nodes,type_ignores=[]));ast.fix_missing_locations(module)
# The generated candidate embeds the public synthetic generator; it never
# imports or reads the reference module in the judge.
s+='\n\n# Public synthetic input generation, copied for prepare-only tuning.\nimport math,functools\n'+ast.unparse(module)+'\n'
board_tree=ast.parse(BOARD.read_text())
control=copy.deepcopy(next(n for n in board_tree.body if isinstance(n,ast.FunctionDef) and n.name=='attention_fa3'))
control.name='_attention_fa3_public';ast.fix_missing_locations(control)
s+='\n\n# Exact board-control FA3 call: preserve caller metadata, including q8 None.\n'+ast.unparse(control)+'\n'
a=s.index('def _tune_attention_portfolio(');b=s.index('\ndef stage_attention',a)
s=s[:a]+(HERE/'tuner.py.inc').read_text()+'\n\n'+s[b:]
s=s.replace("    if label == 'layer':\n        return _layer_stage_attention(qkv, data, state)",
'''    if label == 'layer':
        if data['config']['q_len']==8 and _attn_port_hopper(qkv[0].device):
            q,gate=qk_prep(qkv,data,state,q_fp8=True)
            return _attention_fa3_public(q,gate,data,state)
        return _layer_stage_attention(qkv, data, state)''',1)
# Synthetic admission/forced fixtures carry the same metadata field as public inputs.
s=s.replace('                stream=torch.cuda.current_stream(device))',
            '                stream=torch.cuda.current_stream(device),scheduler_metadata=None)',1)
# Save compact public-draw timing decisions last, alongside the MoE summary.
s=s.replace('    _best6_summary(state)\n', '    _best6_summary(state)\n    _attention_public_summary(state)\n',1)
CAND.write_text(s)
row=dict(sha256=hashlib.sha256(CAND.read_bytes()).hexdigest(),base_sha256=hashlib.sha256(FROZEN.read_bytes()).hexdigest(),
         reference_sha256=hashlib.sha256(REF.read_bytes()).hexdigest(),board_sha256=hashlib.sha256(BOARD.read_bytes()).hexdigest(),
         public_definitions=sorted(needed),changed_functions=['prepare','_attn_port_variant','_attn_port_synthetic','_tune_attention_portfolio'])
(HERE/'build_manifest.json').write_text(json.dumps(row,indent=2)+'\n')
print(CAND,row['sha256'],len(needed),'public generator definitions')
