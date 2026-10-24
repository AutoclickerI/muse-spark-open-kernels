"""Require unchanged best6 compute and exact old-glue/streaming provenance."""
from pathlib import Path
import argparse,ast,copy,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).parent
parser=argparse.ArgumentParser();parser.add_argument('--manifest',default='manifest.json')
parser.add_argument('--out',default='audit_local.json');args=parser.parse_args()
manifest=json.loads((HERE/args.manifest).read_text())
base=ROOT/'problems/muse_decode_layer/cand_r3_best6.py';candidate=ROOT/manifest['source']
assert hashlib.sha256(candidate.read_bytes()).hexdigest()==manifest['sha256']
def definitions(p):return {n.name:n for n in ast.parse(p.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
a,b=definitions(base),definitions(candidate);unchanged=[]
from precision import apply as preserve_raw_scales, CHANGED, REMOVED
corrected=ast.parse(preserve_raw_scales(base.read_text()))
expected={n.name:n for n in corrected.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
for name,node in a.items():
    if name in ('prepare','kernel_fn') or name in REMOVED:continue
    assert ast.dump(expected[name],include_attributes=False)==ast.dump(b[name],include_attributes=False),name
    if name not in CHANGED:unchanged.append(name)
def literals(p):
    return {n.targets[0].id:ast.literal_eval(n.value) for n in ast.parse(p.read_text()).body if isinstance(n,ast.Assign)
        and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str)}
for name,value in literals(base).items():
    if name=='_pf_GROUPED_SRC':
        wanted=next(ast.literal_eval(n.value) for n in corrected.body if isinstance(n,ast.Assign)
                    and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
        assert literals(candidate)[name]==wanted
    else:assert literals(candidate)[name]==value,name
old=definitions(base.with_name('cand_r3_best5.py'));undo={v:k for k,v in manifest['old_glue_names'].items()}
class Undo(ast.NodeTransformer):
    def visit_FunctionDef(self,node):node.name=undo.get(node.name,node.name);return self.generic_visit(node)
    def visit_Name(self,node):node.id=undo.get(node.id,node.id);return node
for oldname,newname in manifest['old_glue_names'].items():
    node=Undo().visit(copy.deepcopy(b[newname]))
    assert ast.dump(node,include_attributes=False)==ast.dump(old[oldname],include_attributes=False),oldname
sp=ROOT/'problems/muse_bf16_proj/dev/streaming'
assert literals(candidate)['SP_CUDA']==(sp/'kernel.cu').read_text()
for name,node in definitions(sp/'runtime.py.inc').items():
    assert ast.dump(node,include_attributes=False)==ast.dump(b[name],include_attributes=False),name
sys.path.insert(0,str(ROOT/'tools'));from okc_run import rejected_words
assert not rejected_words(candidate.read_text())
result=dict(sha256=manifest['sha256'],unchanged_functions=unchanged,raw_scale_helpers=sorted(CHANGED),
            removed_scale_recoders=sorted(REMOVED),native_changed_only=['_pf_GROUPED_SRC'],
            exact_old_glue_helpers=len(undo),stream_kernel_exact=True,stream_runtime_exact=True,rejected_words=[])
(HERE/args.out).write_text(json.dumps(result,indent=2)+'\n')
print('BEST7_AUDIT',len(unchanged),'unchanged best6 helpers;',len(undo),'exact old glue helpers; exact streaming drop-in; no rejected words')
