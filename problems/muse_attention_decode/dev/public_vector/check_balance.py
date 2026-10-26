"""Execute the exact two-lane balanced map and query-sharing barrier on GPU0.

No WGMMA/ring stand-in: this isolates the native-only cross-request/idle-peer
prologue that the one-lane local attention screen cannot reach.
"""
import hashlib,json
from pathlib import Path
import torch
from torch.utils.cpp_extension import load_inline
H=Path(__file__).resolve().parent;O=H/'out';s=(O/'lee_F.cu').read_text()
a=s.index('attnp(A a, WF f) {');a=s.index('  const int wg',a);z=s.index('  const int b = bq',a);mapping=s[a:z]
assert 'first==last' in mapping
a=s.index('template<typename V,typename Params>');z=s.index('\nstruct WF',a);reader=s[a:z]
src=r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
struct QView{const void*data;int S,W,RS;int64_t SS;};
struct WF{const __nv_bfloat16*qkv;QView qv;};
struct A{const int*seq;int B,nsplit,window;};
#define PV_SHARE 1
'''+reader+r'''
__global__ void prologue(A a,WF f,__nv_bfloat16*out,int*map){
 constexpr int NL=2;constexpr bool BAL=true;
'''+mapping+r'''
 if(prod)return;
 const int wt=threadIdx.x&127,h=(threadIdx.x>>7)&1;
 if(threadIdx.x==LN*256)map[blockIdx.x*2+LN]=spare?-1:bq;
 if(spare)return;
 __shared__ __align__(16) __nv_bfloat16 q[2][8*16*64];
 const int token=(wt>>1)>>3,head=h*8+((wt>>1)&7),dim=(wt&1)*32;
 const int local=(token*16+head)*64+dim,slot=shareQ?0:LN;
 if(!shareQ || LN==0){
  #pragma unroll
  for(int j=0;j<4;j++)*(uint4*)(q[slot]+local+8*j)=qread<uint4>(f,f.qkv+((bq*8+token)*2304+head*64+dim+8*j));
 }
 if(shareQ)asm volatile("bar.sync 13, 512;":::"memory");
 else asm volatile("bar.sync %0, 256;"::"r"(9+LN):"memory");
 #pragma unroll
 for(int j=0;j<4;j++)*(uint4*)(out+(blockIdx.x*2+LN)*8*16*64+local+8*j)=*(const uint4*)(q[slot]+local+8*j);
}
void run(torch::Tensor p,torch::Tensor seq,torch::Tensor out,torch::Tensor map,int64_t ns){
 A a{seq.data_ptr<int>(),(int)seq.numel(),(int)ns,0};WF f{(const __nv_bfloat16*)p.data_ptr(),{p.data_ptr(),(int)p.size(0),2304,2304,p.stride(0)}};
 prologue<<<132,640,0,at::cuda::getCurrentCUDAStream()>>>(a,f,(__nv_bfloat16*)out.data_ptr(),map.data_ptr<int>());
}
'''
mod=load_inline(name='p5b_balanced_q_'+hashlib.sha256(src.encode()).hexdigest()[:12],cpp_sources='void run(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t);',cuda_sources=src,functions=['run'],extra_cuda_cflags=['-O3'])
rows=[]
for b in (1,16,32):
 for seed in (0,1,4,13,42):
  torch.manual_seed(seed);seq=torch.randint(1,100000,(b,),device='cuda',dtype=torch.int32)
  if seed==0:
   # Short contexts force an odd active-lane total and an idle peer. The
   # expected queries still come from the independent ordered-sum reference.
   seq.fill_(9);seq[0]=8
  p=torch.randn(3,b*8,2304,device='cuda');out=torch.full((264,8,16,64),float('nan'),device='cuda',dtype=torch.bfloat16);mapping=torch.full((264,),-99,device='cuda',dtype=torch.int32)
  mod.run(p,seq,out,mapping,min(64,2*((264+b-1)//b)));torch.cuda.synchronize()
  want=p[0].clone()
  for i in (1,2):want+=p[i]
  want=want.bfloat16().reshape(b,8,2304)[...,:1024].reshape(b,8,16,64)
  valid=mapping>=0;assert torch.equal(out[valid].view(torch.int16),want[mapping[valid].long()].view(torch.int16))
  pairs=mapping.reshape(-1,2);cross=((pairs[:,0]>=0)&(pairs[:,1]>=0)&(pairs[:,0]!=pairs[:,1])).sum().item();single=((pairs[:,0]>=0)^(pairs[:,1]>=0)).sum().item()
  rows.append(dict(batch=b,seed=seed,cross_request_CTAs=cross,single_active_lane_CTAs=single,bit_exact=True))
print(rows,flush=True)
assert sum(r['cross_request_CTAs'] for r in rows)>0 and sum(r['single_active_lane_CTAs'] for r in rows)>0
(H/'balance_validation.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(src.encode()).hexdigest(),cases=rows),indent=2)+'\n');print(rows)
