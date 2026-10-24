"""Build best7 by adding dispatch settings to the H200-verified best6 source."""
from pathlib import Path
import ast,copy,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).parent
BASE=ROOT/'problems/muse_decode_layer/cand_r3_best6.py'
OLD=BASE.with_name('cand_r3_best5.py');OUT=BASE.with_name('cand_r3_best7.py')
SP=ROOT/'problems/muse_bf16_proj/dev/streaming'
P7=ROOT/'problems/muse_bf16_proj/dev/l2_overlap'
CHAIN_LAB=ROOT/'problems/muse_prefill_layer/lab/lab_persistent_dense_chain.py'
base=BASE.read_text();old=OLD.read_text()
assert hashlib.sha256(base.encode()).hexdigest()=='4516cb979dac623a6b6a2347be54cb5ecb416be23d7631aeae092a403beb9528'
assert hashlib.sha256(old.encode()).hexdigest()=='a651af25cb44550932f497b57eb82fb32c0fbe07c1f21a8592bd0982f937c150'
from precision import apply as preserve_raw_scales
names={'_chunks','_group_barrier','_boundary_kernel','boundary','_route_norm_kernel','route_and_norm','_qk_prep_kernel','qk_prep'}
mapping={name:'_best7_old_'+name.lstrip('_') for name in names}
class Rename(ast.NodeTransformer):
    def visit_FunctionDef(self,node):
        node.name=mapping.get(node.name,node.name);return self.generic_visit(node)
    def visit_Name(self,node):node.id=mapping.get(node.id,node.id);return node
nodes=[copy.deepcopy(n) for n in ast.parse(old).body if isinstance(n,ast.FunctionDef) and n.name in names]
control=ast.fix_missing_locations(Rename().visit(ast.Module(nodes,type_ignores=[])))
settings=(HERE/'settings.py.inc').read_text()
source=preserve_raw_scales(base).replace('Best6: glue base with FP32-gated prefill expert adapters at M128/256.',
                    'Best7: receipt-controlled dispatch for the decode-layer stage portfolio.',1)
where=source.index('\n\nclass Weights')
source=source[:where]+'\n\n'+settings+source[where:]
a=source.index('def kernel_fn(data,state):');b=source.index('\n\ndef prepare(weights):',a)
source=source[:a]+(HERE/'kernel_fn.py.inc').read_text()+source[b:]
source=source.replace("    phases['pdl_prefix']=time.monotonic()-started-sum(phases.values())",
    "    _best7_prepare_settings(state,weights,device)\n    phases['pdl_prefix']=time.monotonic()-started-sum(phases.values())",1)
source=source.replace("print(f'BEST_SHAPE b=", "print(f'BEST6_BASE_SHAPE b=",1)
source=source.replace('    _attention_public_summary(state)\n','    _attention_public_summary(state)\n    _best7_summary(state)\n',1)
source+='\n\n# Exact best5 glue controls, with private function names.\n'+ast.unparse(control)+'\n'
source+='\n\n# Projection lane drop-in; constructed only for table-selected shapes.\nSP_CUDA = '+repr((SP/'kernel.cu').read_text())+'\n'
source+=(SP/'runtime.py.inc').read_text()+'\n\n'+(HERE/'dispatch.py.inc').read_text()
source+='\n\n# P7 within-call side-stream transport, table-selected and captured in the root graph.\n'+(P7/'prefetch.py').read_text()
chain_tree=ast.parse(CHAIN_LAB.read_text())
chain_cuda=next(ast.literal_eval(n.value) for n in chain_tree.body
                 if isinstance(n,ast.Assign) and any(getattr(t,'id',None)=='CUDA_SRC' for t in n.targets))
source+='\n\n# P7 deterministic persistent dense chain; disabled until its H200 gate.\nDENSE_CHAIN_CUDA='+repr(chain_cuda)+'\n'+(HERE/'dense_chain.py.inc').read_text()
sys.path.insert(0,str(ROOT/'tools'));from okc_run import rejected_words
assert not rejected_words(source),rejected_words(source)
OUT.write_text(source)
manifest=dict(source=str(OUT.relative_to(ROOT)),sha256=hashlib.sha256(source.encode()).hexdigest(),
    base_sha256=hashlib.sha256(base.encode()).hexdigest(),old_sha256=hashlib.sha256(old.encode()).hexdigest(),
    stream_cuda_sha256=hashlib.sha256((SP/'kernel.cu').read_bytes()).hexdigest(),
    stream_runtime_sha256=hashlib.sha256((SP/'runtime.py.inc').read_bytes()).hexdigest(),
    settings_sha256=hashlib.sha256(settings.encode()).hexdigest(),old_glue_names=mapping,
    precision_builder_sha256=hashlib.sha256((HERE/'precision.py').read_bytes()).hexdigest(),
    prefetch_sha256=hashlib.sha256((P7/'prefetch.py').read_bytes()).hexdigest())
(HERE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(OUT,manifest['sha256'])
