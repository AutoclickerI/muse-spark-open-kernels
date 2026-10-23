"""Validate only the pruned, dtype/hierarchy-specialized fused source for v2.
This requires no H200 receipt and makes no claim about which existing path wins.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import torch
from torch.utils.cpp_extension import load_inline

P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'dev'));sys.path.insert(0,str(P))
import build_portfolio_v2 as builder
import reference as ref
ap=argparse.ArgumentParser();ap.add_argument('--arch',default='12.0');ap.add_argument('--build-only',action='store_true');args=ap.parse_args()
configs=builder.local_fused_choices()
for layer,m,config in [('router',128,(128,64,3,3,32,160)),('router',256,(256,64,2,3,32,160))]:
    if config not in configs[(layer,m)]:configs[(layer,m)].append(config)
source,specs=builder.fused_source([(c,layer=='router') for (layer,m),row in configs.items() for c in row])
prefix='''#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <stdint.h>
namespace cg=cooperative_groups;
'''
cpp='''void hopper_pair(torch::Tensor W,torch::Tensor X,torch::Tensor Out,torch::Tensor Work,torch::Tensor Counts,
                 int64_t bt,int64_t bk,int64_t st,int64_t mode,int64_t split,int64_t threads);'''
name='proj_v2_pair_'+hashlib.sha1(source.encode()).hexdigest()[:10]+'_'+args.arch.replace('.','_')
os.environ['TORCH_CUDA_ARCH_LIST']=args.arch
mod=load_inline(name=name,cpp_sources=cpp,cuda_sources=prefix+source,functions=['hopper_pair'],extra_cuda_cflags=['-O3','-std=c++20','--expt-relaxed-constexpr','-Xptxas=-v'],verbose=False)
print(json.dumps({'module':name,'specializations':len(specs),'arch':args.arch}),flush=True)
if args.build_only:sys.exit(0)
weights=ref.generate_weights(ref.WEIGHTS_SEED);router=weights['router'].float();hi=router.bfloat16();lo=(router-hi.float()).bfloat16()
wr=torch.cat([hi,lo],1).contiguous();cases=0
for (layer,tuned_m),row in configs.items():
    for config in row:
        bt,bk,st,mode,split,threads=config
        smem=max(st*(64*(2 if layer=='router' else 1)+bt)*bk*2+st*16,bt*68*4)
        if smem>torch.cuda.get_device_properties(0).shared_memory_per_block_optin:continue
        for m in (9,33,128,130,256):
            data=ref.generate_input(m=m,layer=layer,seed=9223,dist='outliers');x=data['x'];n=data['config']['n']
            tiles=(n//64)*((m+bt-1)//bt)
            work=torch.empty(tiles*17*bt*64,device='cuda',dtype=torch.float32);counts=torch.zeros(tiles,device='cuda',dtype=torch.uint32)
            w=wr if layer=='router' else weights[layer]
            out=torch.empty((m,n),device='cuda',dtype=torch.float32 if layer=='router' else torch.bfloat16)
            def call():mod.hopper_pair(w,x,out,work,counts,*config)
            call();torch.cuda.synchronize()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):call()
            for seed in (9223,56371,17999):
                fresh=ref.generate_input(m=m,layer=layer,seed=seed,dist='outliers');x.copy_(fresh['x']);graph.replay();torch.cuda.synchronize()
                ok,msg=ref.check_implementation(data,out,weights);assert ok,(layer,config,m,msg)
                assert not bool(counts.view(torch.int32).any()),'counter reset failed'
                cases+=1
print(json.dumps({'pruned_pair_cases':cases,'specializations':len(specs),'ok':True}),flush=True)
