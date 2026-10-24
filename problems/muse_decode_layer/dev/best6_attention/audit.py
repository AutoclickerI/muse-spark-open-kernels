"""Verify the attention-only delta and exact public generator provenance."""
from pathlib import Path
import ast,copy,hashlib,json
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).parent
CAND=ROOT/'problems/muse_decode_layer/cand_r3_best6.py';BASE=HERE/'base_c582.py';REF=ROOT/'problems/muse_decode_layer/reference.py'
manifest=json.loads((HERE/'build_manifest.json').read_text())
assert hashlib.sha256(CAND.read_bytes()).hexdigest()==manifest['sha256']
assert hashlib.sha256(BASE.read_bytes()).hexdigest()==manifest['base_sha256']
assert hashlib.sha256(REF.read_bytes()).hexdigest()==manifest['reference_sha256']
def funcs(t):return {n.name:n for n in t.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
def literals(t):return {x.id:n.value.value for n in t.body if isinstance(n,ast.Assign) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str) for x in n.targets if isinstance(x,ast.Name)}
a,b,p=map(lambda q:ast.parse(q.read_text()),(BASE,CAND,REF));fa,fb,fp=funcs(a),funcs(b),funcs(p)
changed=set(manifest['changed_functions']);unchanged=[]
for name,node in fa.items():
 if name in changed:continue
 assert ast.dump(node,include_attributes=False)==ast.dump(fb[name],include_attributes=False),name
 unchanged.append(name)
for name,value in literals(a).items():assert value==literals(b)[name],name
mapping={'_ap_'+name.lstrip('_'):name for name in manifest['public_definitions']}
class Undo(ast.NodeTransformer):
 def visit_Name(self,n):n.id=mapping.get(n.id,n.id);return n
 def visit_FunctionDef(self,n):n.name=mapping.get(n.name,n.name);return self.generic_visit(n)
public=[]
for name in manifest['public_definitions']:
 if name not in fp:continue
 node=Undo().visit(copy.deepcopy(fb['_ap_'+name.lstrip('_')]))
 assert ast.dump(node,include_attributes=False)==ast.dump(fp[name],include_attributes=False),name
 public.append(name)
# No reference-module imports or computation/check helpers enter the candidate.
assert all('reference' not in (n.module or '') for n in ast.walk(b) if isinstance(n,ast.ImportFrom))
assert set(public)=={'generate_input','_longest','_lengths','_rms','_quant','_fill_cache','_stream_basis','_fa3_scheduler_metadata'}
assert 'kernel_fn' in unchanged and '_best6_tune' in unchanged and '_best6_build_adapters' in unchanged
board_path=ROOT/'runs/muse_decode_layer/20261023T185641Z_submit_0e94268cf5654012.py'
control=funcs(ast.parse(board_path.read_text()))['attention_fa3'];actual=copy.deepcopy(fb['_attention_fa3_public']);actual.name='attention_fa3'
assert ast.dump(control,include_attributes=False)==ast.dump(actual,include_attributes=False)
result=dict(sha256=manifest['sha256'],unchanged_functions=unchanged,unchanged_native_literals=list(literals(a)),
            exact_public_generator_functions=public,board_fa3_control_exact=True,
            draws=3,sampling='fixed at B1; public agentic spread at B16/B32; ctx80000, page1, soft caches',
            selection='mean of per-draw medians; incomplete trials never selected',q1_and_q8_same_sampling=True)
(HERE/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print('ATTENTION_AUDIT',len(unchanged),'unchanged functions;',len(public),'exact public generator helpers; all native literals unchanged')
