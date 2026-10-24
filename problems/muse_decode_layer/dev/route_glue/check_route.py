"""Native route/MoE prologue vs best7 on changed PR/EP owners under graph replay."""
from pathlib import Path
import json
import sys
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'tools'))
import local_eval


def main():
    torch.backends.cuda.matmul.allow_tf32=False
    # The whole lab installs the exact real packed expert bank before this
    # optional hook; it compares fused and separate routing with identical PR.
    m=local_eval.load_module('route_reference',ROOT/'problems/muse_decode_layer/cand_r3_best7_scale.py')
    from torch.utils.cpp_extension import load_inline
    import hashlib
    raw='#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <cuda_bf16.h>\n#include <math_constants.h>\n'+(HERE/'prologue.cuh').read_text()+'''
__global__ void probe(const float* p,const float* bias,const float* pre,int* ids,int* lids,float* wts,int* local,__nv_bfloat16* x,int M,int S){
  extern __shared__ char mem[];int* private_ids=(int*)mem;float* weights=(float*)(private_ids+M*8);float* inv=weights+M*8;
  RouteGlue rg={p,bias,pre,S,(int64_t)M*4608,4608,ids,lids,wts,local};
  rg_prepare(rg,M,private_ids,weights,inv);
  for(int j=threadIdx.x;j<M*4096;j+=256)x[j]=rg_value(rg,j/4096,j%4096,inv[j/4096]);
}
void check(torch::Tensor p,torch::Tensor bias,torch::Tensor pre,torch::Tensor ids,torch::Tensor lids,torch::Tensor wts,torch::Tensor local,torch::Tensor x){
  int M=p.size(1),S=p.size(0);probe<<<1,256,M*(8*8+4),at::cuda::getCurrentCUDAStream()>>>(p.data_ptr<float>(),bias.data_ptr<float>(),pre.data_ptr<float>(),ids.data_ptr<int>(),lids.data_ptr<int>(),wts.data_ptr<float>(),local.data_ptr<int>(),(__nv_bfloat16*)x.data_ptr(),M,S);
}
'''
    mod=load_inline(name='route_glue_probe_'+hashlib.sha256(raw.encode()).hexdigest()[:12],
        cpp_sources='void check(torch::Tensor p,torch::Tensor bias,torch::Tensor pre,torch::Tensor ids,torch::Tensor lids,torch::Tensor wts,torch::Tensor local,torch::Tensor x);',cuda_sources=raw,functions=['check'],extra_cuda_cflags=['-O3','-std=c++20'],verbose=False)
    gen=torch.Generator(device='cuda').manual_seed(9301);rows=[];worst=0
    for M in (1,8,16,32,128,256):
      for S in (1,2,8,16):
        p=torch.empty((S,M,4608),device='cuda',dtype=torch.float32)
        bias=torch.randn(256,device='cuda',generator=gen)*.055;pre=1+torch.randn(4096,device='cuda',generator=gen)*.1
        ids=torch.empty((M,8),device='cuda',dtype=torch.int32);lids=torch.empty_like(ids);wts=torch.empty((M,8),device='cuda',dtype=torch.float32)
        local=torch.tensor([0],device='cuda',dtype=torch.int32);x=torch.empty((M,4096),device='cuda',dtype=torch.bfloat16)
        fn=lambda:mod.check(p,bias,pre,ids,lids,wts,local,x)
        p.normal_(generator=gen);fn();graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):fn()
        for draw in range(3):
            p.normal_(generator=gen);p.mul_(.7/S**.5);local.fill_(draw*96);graph.replay()
            ref=m.route_and_norm((p,S),dict(router_bias=bias,pre_expert_w=pre),dict(local_start=local))
            assert torch.equal(ids,ref[1]),(M,S,draw,'route IDs')
            assert torch.equal(lids,ref[3]),(M,S,draw,'local IDs')
            for a,b in ((x,ref[0]),(wts,ref[2])):
                err=float((a.float()-b.float()).norm()/b.float().norm());worst=max(worst,err)
                assert err<.001,(M,S,draw,err)
        rows.append(dict(tokens=M,splits=S,draws=3,route_flips=0))
    result=dict(checks=len(rows)*3,max_l2=worst,rows=rows)
    (HERE/'route_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'}))


if __name__=='__main__':main()
