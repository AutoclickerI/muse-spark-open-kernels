"""Build from reviewed kernels only; never import or execute donor drivers."""
import ast,base64,copy,hashlib,json,lzma,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3]
DONORS=Path('/data2/jihoon/muse_kernel/review/moe-public-Ury55uMA')
def sha(s):return hashlib.sha256(s.encode() if isinstance(s,str) else s).hexdigest()
def val(s,name):
 n=next(n for n in ast.parse(s).body if isinstance(n,ast.Assign) and any(getattr(t,'id','')==name for t in n.targets));return ast.literal_eval(n.value)
def fun(s,name):return ast.get_source_segment(s,next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name==name))
def once(s,a,b):assert s.count(a)==1,(a,s.count(a));return s.replace(a,b)
def make_operator(s):
 # Retain all reviewed device kernels and repackers; replace the driver with a
 # list of closures. Allocation and tensor ABI stay identical to the donor.
 t=ast.parse(s);kernels=s[:s.index('def _decode1_direct')]
 class Plan(ast.NodeTransformer):
  def __init__(self):self.i=0
  def visit_Expr(self,n):
   if isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Subscript):
    name=n.value.func.value.id;label='route' if name.startswith('_route') else 'finish' if name in ('_wsum_ks','_post_expert_norm_weighted_sum_kernel') else 'w13' if name.startswith(('_w13','_d1')) or any(k.arg=='GU' and isinstance(k.value,ast.Constant) and k.value.value for k in n.value.keywords) else 'w2'
    self.i+=1;f='launch_'+str(self.i)
    return [ast.FunctionDef(name=f,args=ast.arguments(posonlyargs=[],args=[],kwonlyargs=[],kw_defaults=[],defaults=[]),body=[n],decorator_list=[]),ast.parse(f"steps.append(({label!r},{f}))").body[0]]
   return n
  def visit_Return(self,n):
   if isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='_decode1_direct':
    n.value.func.id='operator_plan1';return n
   return ast.Return(ast.parse("dict(steps=steps,out=out,owners=locals())",mode='eval').body)
 for src,new in [(fun(s,'_decode1_direct'),'operator_plan1'),(fun(s,'kernel_fn'),'operator_plan')]:
  n=ast.parse(src).body[0];n.name=new;n=Plan().visit(n);n.body.insert(1 if isinstance(n.body[0],ast.Expr) and isinstance(n.body[0].value,ast.Constant) else 0,ast.parse('steps=[]').body[0]);kernels+='\n'+ast.unparse(ast.fix_missing_locations(n))+'\n'
 kernels+='\n'+fun(s,'_w13_regcap')+'\n'
 return kernels

def make_lee_x(s):
 cu=val(s,'CUDA_X');cu=cu[:cu.index('static int g_gridX')]
 cu=cu.replace('#include <torch/extension.h>','#include <cuda_runtime.h>\n#include <vector>\n#include <stdexcept>').replace('#include <ATen/cuda/CUDAContext.h>','')
 # An explicit template splits exactly the same queue and math for diagnostics
 # and S19's large-M ABI. The fused instantiation remains the original queue.
 cu=once(cu,'__global__ void __launch_bounds__(NTH, 1) moe_x(P p)', 'template<int PHASE>\n__global__ void __launch_bounds__(NTH, 1) moe_x(P p)')
 cu=once(cu,'for (int g = 0; g < 2; g++)  // all W13 units', 'for (int g = (PHASE == 1 ? 1 : 0); g < (PHASE == 0 ? 1 : 2); g++)  // all W13 units')
 cu=once(cu,'if (g && !(depok >> e & 1))', 'if (PHASE == -1 && g && !(depok >> e & 1))')
 cu=once(cu,'if (lane == 0) while (ldacq(p.ctr + C_W2) < nw2)', 'if (PHASE != 0 && lane == 0) while (ldacq(p.ctr + C_W2) < nw2)')
 a=cu.index('  for (int it = blockIdx.x');b=cu.index('\n#ifdef PROF\n  PSET(7',a);finish=cu[a:b]
 cu=cu[:a]+'  if constexpr(PHASE == -1) {\n'+finish+'\n  }\n'+cu[b:]
 # Standalone finalization uses the original per-warp vector loop unchanged.
 cu+='\n__global__ void finish_x(P p) { const int wid=threadIdx.x>>5,lane=threadIdx.x&31;\n'+finish+'\n}\n'
 cu+='\n'+(H/'lee_host.cu.inc').read_text()
 return cu

def main():
 out=H/'out';out.mkdir(exist_ok=True)
 op=(DONORS/'operatorstack/muse_moe_w4a16/submission.py').read_text();lee=(DONORS/'lee101/muse_moe_w4a16/submission_v43d.py').read_text()
 notices='operatorstack/open-kernel-challenge-solutions @ 6d9b77656b6c295e421be9e1424b0f8f99299b0a\n'+(DONORS/'operatorstack/LICENSE').read_text()+'\n'+(DONORS/'operatorstack/NOTICE').read_text()+'\nlee101/openkernels @ 601c65413496cf63f942b4f15ca087839ef2f5cc; MoE v43d\nModified by Jihoon: reviewed kernel extraction, explicit ABI/phase launchers, no donor tuning or diagnostics.\nNo upstream Lee NOTICE file exists at this pin.\n'+(DONORS/'lee101/LICENSE').read_text()
 (H/'NOTICES.txt').write_text(notices)
 head='\n'.join('# '+l for l in notices.splitlines())+'\n'
 ops=head+make_operator(op);(H/'operator.py.inc').write_text(ops.replace('\n# \n','\n#\n'))
 x=make_lee_x(lee);(H/'lee_x.cu.inc').write_text(x)
 f=val(lee,'CUDA_F');f=f[:f.index('static int g_gridF')].replace('#include <torch/extension.h>','#include <cuda_runtime.h>\n#include <vector>\n#include <stdexcept>').replace('#include <ATen/cuda/CUDAContext.h>','')+(H/'lee_f_host.cu.inc').read_text();(H/'lee_f.cu.inc').write_text(f)
 repack='import torch\n_ROT = ((0,0),(3,1),(6,6),(-5,5))\n'+'\n\n'.join(fun(lee,n) for n in ('_place','_repack','_repack2','_repackX'))
 # Pin the actual source's rotation table rather than infer its sign convention.
 repack=once(repack,'_ROT = ((0,0),(3,1),(6,6),(-5,5))',next(l for l in lee.splitlines() if l.startswith('_ROT =')))
 lee_source=head+repack+'\n'+(H/'lee_runtime.py.inc').read_text()
 (H/'lee_repack.py.inc').write_text(lee_source.replace('\n# \n','\n#\n'))
 s=(R/'problems/muse_decode_layer/cand_r3_best7_s19_final.py').read_text();t=ast.parse(s);n=next(n for n in t.body if isinstance(n,ast.Assign) and any(getattr(t,'id','')=='_source' for t in n.targets));blob=next(n.value for n in ast.walk(n.value) if isinstance(n,ast.Constant) and isinstance(n.value,str) and len(n.value)>10000);base=lzma.decompress(base64.b85decode(blob)).decode()
 frozen=json.loads((H.parent/'s19/frozen.json').read_text());assert sha(base)==frozen['expanded_sha256'] and sha(s)==frozen['shipped_sha256']
 ck='\n'.join(l for p in ('checks.py','nvfp4.py','reference.py') for l in (R/'problems/muse_moe_w4a16'/p).read_text().splitlines() if not l.startswith(('from checks import','from nvfp4 import','from __future__ import')))
 # The same random_weights generator makes only the local 32 experts, at the
 # original precision. Input generation still has all 256 experts.
 ck=ck.replace('leading=(N_EXPERTS,)','leading=(32,)')
 B=dict(base=base,operator=ops,lee=lee_source,lee_x=x,lee_f=f,checker=ck,notice=notices)
 body='import base64,gc,hashlib,json,linecache,lzma,math,statistics,struct,time,types\nimport torch\nBUNDLE=json.loads(lzma.decompress(base64.b85decode('+repr(base64.b85encode(lzma.compress(json.dumps(B).encode(),preset=9)).decode())+')))\n'+(H/'runtime.py.inc').read_text()+'\n'+(H/'lab.py.inc').read_text()
 compile(body,'<moe-public>','exec');(out/'public_moe.py').write_text(body)
 manifest=dict(donors={'operatorstack':dict(commit='6d9b77656b6c295e421be9e1424b0f8f99299b0a',source_sha256=sha(op)),'lee101':dict(commit='601c65413496cf63f942b4f15ca087839ef2f5cc',source_sha256=sha(lee))},identities={k:sha(v) for k,v in B.items()},payload_sha256=sha(body),seeds=[183528941+104729*i for i in range(8)],weight_seed=183528937,rounds=4)
 (H/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(manifest))
if __name__=='__main__':main()
