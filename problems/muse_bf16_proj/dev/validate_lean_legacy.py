"""Validate the lightweight ABI for original doubled-K incumbent kernels before receipt pruning."""
import ast,hashlib,importlib.util,json,os,sys
from pathlib import Path
import torch
from torch.utils.cpp_extension import load_inline
P=Path(__file__).resolve().parents[1];sys.path[:0]=[str(P),str(P/'dev')]
import reference as ref
import build_portfolio_v2 as builder
import lean_codegen
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--arch',default='12.0');ap.add_argument('--build-only',action='store_true');a=ap.parse_args()
base=(P/'cand_fused_lean.py').read_text();v=lean_codegen.values_of(base);raw=v['RAW_CUDA']
old=(P/'cand_portfolio.py').read_text();ov=lean_codegen.values_of(old);ns={**ov};ns['HOP_CONFIGS']=((64,64,4),)
for name in ('_wgmma_fn','_hop_source'):
 node=next(n for n in ast.parse(old).body if isinstance(n,ast.FunctionDef) and n.name==name);exec(ast.get_source_segment(old,node),ns)
legacy=lean_codegen.device_only(ns['_hop_source'](),'muse_wg').replace('muse_wg','muse_legacy')
oldhost=(P/'dev/lean_old_host.cuh').read_text().replace('LEAN_OLD_CASES','LOLD(64,64,4,true) LOLD(64,64,4,false)')
raw+='\n'+legacy+'\n'+oldhost
os.environ['TORCH_CUDA_ARCH_LIST']=a.arch
name='proj_lean_old_'+hashlib.sha1(raw.encode()).hexdigest()[:10]+'_'+a.arch.replace('.','_')
path=load_inline(name=name,cpp_sources='',cuda_sources=raw,functions=None,is_python_module=False,no_implicit_headers=True,
                 extra_cuda_cflags=['-O3','-std=c++20','--expt-relaxed-constexpr'],extra_ldflags=['-lcublasLt'],verbose=False)
print(json.dumps({'module':name,'arch':a.arch,'build':'ok'}),flush=True)
if a.build_only:sys.exit(0)
code=(P/'dev/lean_module.py').read_text();scope={'torch':torch,'os':os};exec(code,scope);native=scope['_LeanModule'](path)
weights=ref.generate_weights(ref.WEIGHTS_SEED);router=weights['router'].float();hi=router.bfloat16();lo=(router-hi.float()).bfloat16();wp=torch.cat((hi,lo),1).contiguous();cases=0
for layer in ('router','qkv_gate'):
 for m in (9,33,128,130,256):
  data=ref.generate_input(m=m,layer=layer,seed=5667,dist='outliers');x=data['x'];n=data['config']['n'];tiles=(n//64)*((m+63)//64)
  work=torch.empty(tiles*17*64*64,device='cuda',dtype=torch.float32);counts=torch.zeros(tiles,device='cuda',dtype=torch.uint32)
  out=torch.empty((m,n),device='cuda',dtype=torch.float32 if layer=='router' else torch.bfloat16);w=wp if layer=='router' else weights[layer]
  for mode,split in ((0,1),(0,8),(1,8),(1,16),(2,132),(2,264)):
   native.old_hopper(w,x,out,work,counts,64,64,4,mode,split);torch.cuda.synchronize()
   g=torch.cuda.CUDAGraph()
   with torch.cuda.graph(g):native.old_hopper(w,x,out,work,counts,64,64,4,mode,split)
   for seed in (5667,7812,41201):
    fresh=ref.generate_input(m=m,layer=layer,seed=seed,dist='outliers');x.copy_(fresh['x']);g.replay();torch.cuda.synchronize()
    ok,msg=ref.check_implementation(data,out,weights);assert ok,(layer,m,mode,split,msg)
    assert not bool(counts.view(torch.int32).any());cases+=1
print(json.dumps({'lean_legacy_cases':cases,'ok':True}),flush=True)
