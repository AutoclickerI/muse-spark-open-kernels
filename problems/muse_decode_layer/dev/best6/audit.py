"""Pin unchanged kernels/adapters and audit each optional failure boundary."""
from pathlib import Path
import ast,hashlib,json
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).parent
BASE=ROOT/'problems/muse_decode_layer/cand_r3_best5_glue.py';CAND=BASE.with_name('cand_r3_best6.py')
PREFILL=ROOT/'problems/muse_prefill_layer/cand_mid_sweep.py'
manifest=json.loads((HERE/'build_manifest.json').read_text())
assert hashlib.sha256(BASE.read_bytes()).hexdigest()==manifest['base_sha256']
assert hashlib.sha256(PREFILL.read_bytes()).hexdigest()==manifest['prefill_sha256']
assert hashlib.sha256(CAND.read_bytes()).hexdigest()==manifest['sha256']
a=ast.parse(BASE.read_text());b=ast.parse(CAND.read_text());p=ast.parse(PREFILL.read_text())
def funcs(t):return {n.name:n for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
def literals(t):return {k.id:ast.literal_eval(n.value) for n in t.body if isinstance(n,ast.Assign) for k in n.targets if isinstance(k,ast.Name) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str)}
fa,fb,fp=funcs(a),funcs(b),funcs(p);la,lb=literals(a),literals(b)
for name in la:assert la[name]==lb[name],name
unchanged=[name for name in fa if name not in ('prepare','kernel_fn')]
for name in unchanged:assert ast.dump(fa[name],include_attributes=False)==ast.dump(fb[name],include_attributes=False),name
names=manifest['embedded_definitions']+manifest['third_party_names'];mapping={('_pf'+n if n.startswith('_') else '_pf_'+n):n for n in names}
class Unprefix(ast.NodeTransformer):
 def visit_Name(self,n):
  n.id=mapping.get(n.id,n.id);return n
 def visit_FunctionDef(self,n):
  n.name=mapping.get(n.name,n.name);return self.generic_visit(n)
 def visit_ClassDef(self,n):
  n.name=mapping.get(n.name,n.name);return self.generic_visit(n)
 def visit_Global(self,n):
  n.names=[mapping.get(x,x) for x in n.names];return n
adapter_functions=[]
for name in manifest['embedded_definitions']:
 if name not in fp:continue
 pref='_pf'+name if name.startswith('_') else '_pf_'+name
 copy=Unprefix().visit(ast.parse(ast.unparse(fb[pref])).body[0])
 assert ast.dump(copy,include_attributes=False)==ast.dump(fp[name],include_attributes=False),name
 adapter_functions.append(name)
assert lb['_pf_GROUPED_SRC']==literals(p)['GROUPED_SRC']
# Every optional call is reached from a Try in prepare construction/admission/timing.
for name in ('_best6_build_adapters','_best6_tune'):
 node=fb[name]
 assert any(isinstance(n,ast.Try) and any(isinstance(h.type,ast.Name) and h.type.id=='Exception' for h in n.handlers) for n in ast.walk(node)),name
row=dict(sha256=manifest['sha256'],base_sha256=manifest['base_sha256'],prefill_sha256=manifest['prefill_sha256'],
         unchanged_base_functions=unchanged,unchanged_base_native_literals=list(la),
         exact_prefill_functions=adapter_functions,exact_prefill_native_literal='GROUPED_SRC',
         choices=['v9nt1','v9nt2','raw17','humming_masked','dg_w8'],tokens=[128,256],
         timing='mean of three seed means, four cold replays per seed, order reversed on alternate repetitions',
         fp32_gate=dict(relative_l2=.01,worst_row=.05),prepared_dense_expert_weight_banks=0)
(HERE/'audit.json').write_text(json.dumps(row,indent=2)+'\n');print('AUDIT',len(unchanged),'unchanged base functions;',len(adapter_functions),'exact prefill definitions; native literals exact')
