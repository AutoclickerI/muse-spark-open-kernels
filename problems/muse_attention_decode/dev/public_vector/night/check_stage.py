"""GPU test of the exact NL2 staged-reader prologue and temporary lifetime.

Only the unused history-ring address span is remapped to fit sm120 shared
memory; the balanced map, copy, barriers and vector readers are the new code.
"""
import hashlib,json
from pathlib import Path
import torch
from torch.utils.cpp_extension import load_inline
H=Path(__file__).resolve().parent;O=H/'out';s=(O/'stage.cu').read_text()
a=s.index('attnp(A a, WF f) {');a=s.index('  const int wg',a);z=s.index('  const int b = bq',a);mapping=s[a:z]
a=s.index('  const uint32_t rawq=');z=s.index('  if (EQ &&',a);copy=s[a:z].replace('shareQ?sb+RS*16384+8192:vf0','shareQ?sb:vf0')
a=s.index('template<typename V,typename Params>');z=s.index('\nstruct WF',a);reader=s[a:z]
src=r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
struct QView{const void*data;int S,W,RS;int64_t SS;uint32_t stage;int first;};
struct WF{const __nv_bfloat16*qkv;QView qv;};
struct A{const int*seq;int B,nsplit,window;};
#define PV_SHARE 1
'''+reader+r'''
__global__ void prologue(A a,WF f,__nv_bfloat16*out,int*map){
 constexpr int NL=2;constexpr bool BAL=true;
'''+mapping+r'''
 if(prod)return;
 const int tid=threadIdx.x-256*LN;
 if(tid==0)map[blockIdx.x*2+LN]=spare?-1:bq;
 if(spare)return;
 const int b=bq;
 __shared__ __align__(16) __nv_bfloat16 temp[16+2*8*1280];
 // Zero is the inactive-stage sentinel; every tested staged pointer is live.
 const uint32_t sb=(uint32_t)__cvta_generic_to_shared(temp+16),vf0=sb+LN*8*1280*2;
'''+copy+r'''
 for(int i=tid;i<8*2304/8;i+=256){
  int tok=i/(2304/8),c=(i%(2304/8))*8;
  *(uint4*)(out+(blockIdx.x*2+LN)*8*2304+i*8)=qread<uint4>(f,f.qkv+(b*8+tok)*2304+c);
 }
 if(shareQ)asm volatile("bar.sync 6, %0;"::"n"(256*NL):"memory");
 else asm volatile("bar.sync %0,256;"::"r"(6+LN):"memory");
 f.qv.stage=0;
 // Exactly as the attention mainloop, storage is now free for V widening.
 if(!shareQ || LN==0)for(int i=tid;i<8*1280/8;i+=256)asm volatile("st.shared.v4.u32 [%0],{0,0,0,0};"::"r"(rawq+i*16):"memory");
 if(shareQ)asm volatile("bar.sync 6, %0;"::"n"(256*NL):"memory");
 else asm volatile("bar.sync %0,256;"::"r"(6+LN):"memory");
 for(int i=tid;i<8*1024/8;i+=256){
  int tok=i/(1024/8),c=1280+(i%(1024/8))*8;
  *(uint4*)(out+264*8*2304+(blockIdx.x*2+LN)*8*1024+i*8)=qread<uint4>(f,f.qkv+(b*8+tok)*2304+c);
 }
}
void run(torch::Tensor p,torch::Tensor seq,torch::Tensor out,torch::Tensor map,int64_t ns,int64_t bf){
 A a{seq.data_ptr<int>(),(int)seq.numel(),(int)ns,0};WF f{(const __nv_bfloat16*)p.data_ptr(),{p.data_ptr(),bf?0:(int)p.size(0),2304,2304,p.stride(0),0,0}};
 prologue<<<132,640,0,at::cuda::getCurrentCUDAStream()>>>(a,f,(__nv_bfloat16*)out.data_ptr(),map.data_ptr<int>());
}
'''
mod=load_inline(name='p5b_stage_lifetime_'+hashlib.sha256(src.encode()).hexdigest()[:12],cpp_sources='void run(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t);',cuda_sources=src,functions=['run'],extra_cuda_cflags=['-O3'])
rows=[]
for b in (1,16,32):
 for seed in (0,1,4):
  for bf in (False,True):
   torch.manual_seed(seed);seq=torch.randint(1,100000,(b,),device='cuda',dtype=torch.int32)
   if seed==0:seq.fill_(9);seq[0]=8
   p=torch.randn(1 if bf else 3,b*8,2304,device='cuda',dtype=torch.bfloat16 if bf else torch.float32)
   out=torch.full((264*8*(2304+1024),),float('nan'),device='cuda',dtype=torch.bfloat16);mapping=torch.full((264,),-99,device='cuda',dtype=torch.int32)
   mod.run(p,seq,out,mapping,min(64,2*((264+b-1)//b)),bf);torch.cuda.synchronize()
   want=p[0].float()
   for i in range(1,p.shape[0]):want+=p[i].float()
   want=want.bfloat16().reshape(b,8,2304);valid=mapping>=0;indices=mapping[valid].long()
   got=out[:264*8*2304].reshape(264,8,2304)[valid];gate=out[264*8*2304:].reshape(264,8,1024)[valid]
   assert torch.equal(got.view(torch.int16),want[indices].view(torch.int16))
   assert torch.equal(gate.view(torch.int16),want[indices,:,1280:].view(torch.int16))
   pairs=mapping.reshape(-1,2);cross=((pairs[:,0]>=0)&(pairs[:,1]>=0)&(pairs[:,0]!=pairs[:,1])).sum().item();idle=((pairs[:,0]>=0)^(pairs[:,1]>=0)).sum().item()
   rows.append(dict(batch=b,seed=seed,BF16=bf,cross_request_CTAs=cross,single_active_lane_CTAs=idle,bit_exact=True,gate_after_overwrite_exact=True))
print(rows,flush=True);assert sum(r['cross_request_CTAs'] for r in rows)>0 and sum(r['single_active_lane_CTAs'] for r in rows)>0
(H/'stage_validation.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(src.encode()).hexdigest(),stage_cuda_sha256=hashlib.sha256(s.encode()).hexdigest(),cases=rows),indent=2)+'\n')
