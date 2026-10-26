"""Exercise the exact native readers against ordered FP32 sum -> BF16."""
import hashlib,json
from pathlib import Path
import torch
from torch.utils.cpp_extension import load_inline
H=Path(__file__).resolve().parent;O=H/'out';b=json.loads((O/'bundle.json').read_text())
def function(s,needle):
 a=s.index(needle);z=s.index('{',a);depth=1;j=z+1
 while depth:
  depth+=(s[j]=='{')-(s[j]=='}');j+=1
 return s[a:j]
cs='template<int N>\n'+function(b['native_cs'],'DI void input_pairs')
pa='template<int N>\n'+function(b['native_partial'],'DEVI void proj_pairs')
lee=b['modules']['lee_F'];import ast
cu=ast.literal_eval(next(n.value for n in ast.parse(lee).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='CUDA'))
qr='template<typename V,typename Params>\n'+function(cu,'__device__ __forceinline__ V qread')
header=r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#define DI __device__ __forceinline__
#define DEVI DI
struct VP{const void* qkv;int L,proj_splits,stride;int64_t split_stride;};
struct Params{const __nv_bfloat16* qkv;int proj_splits,qkv_stride;int64_t split_stride;};
struct QView{const void* data;int S,W,RS;int64_t SS;};
struct WF{const __nv_bfloat16* qkv;QView qv;};
'''
tail=r'''
__global__ void readers(const void*src,__nv_bfloat16*out,int s,int bf){
 int lane=threadIdx.x&31,w=threadIdx.x>>5;int tk[5],col[5];bool valid[5];float2 x[5];__nv_bfloat162 y[5];
 for(int i=0;i<5;i++){tk[i]=w+4*i;col[i]=2*lane;valid[i]=true;}
 VP p{src,1,bf?0:s,2304,20*2304};input_pairs<5>(p,tk,col,valid,x);
 Params q{(const __nv_bfloat16*)src,bf?0:s,2304,20*2304};proj_pairs<5>(q,tk,col,y);
 for(int i=0;i<5;i++){
  *(__nv_bfloat162*)(out+(tk[i]*64+2*lane))=__floats2bfloat162_rn(x[i].x,x[i].y);
  *(__nv_bfloat162*)(out+20*64+tk[i]*64+2*lane)=y[i];
 }
 int gt[4],gc[4];__nv_bfloat162 g[4];
 for(int i=0;i<4;i++){gt[i]=w;gc[i]=(lane&7)*8+2*i;}
 proj_pairs<4>(q,gt,gc,g);
 if(lane<8)for(int i=0;i<4;i++)*(__nv_bfloat162*)(out+40*64+w*64+gc[i])=g[i];
 if(lane<8){
  WF f{(const __nv_bfloat16*)src,{src,bf?0:s,2304,2304,20*2304}};
  const uint4 v=qread<uint4>(f,f.qkv+w*2304+lane*8);
  *(uint4*)(out+44*64+w*64+lane*8)=v;
 }
}
void run(torch::Tensor p,torch::Tensor out,int64_t s,int64_t bf){readers<<<1,128,0,at::cuda::getCurrentCUDAStream()>>>(p.data_ptr(),(__nv_bfloat16*)out.data_ptr(),s,bf);}
'''
src=header+cs+'\n'+pa+'\n'+qr+'\n'+tail
mod=load_inline(name='p5b_vector_reader_'+hashlib.sha256(src.encode()).hexdigest()[:12],cpp_sources='void run(torch::Tensor,torch::Tensor,int64_t,int64_t);',cuda_sources=src,functions=['run'],extra_cuda_cflags=['-O3'])
rows=[]
for s in (1,3,7,10):
 for bf in (False,True):
  torch.manual_seed(113+s);p=torch.randn(s if not bf else 1,20,2304,device='cuda',dtype=torch.float32 if not bf else torch.bfloat16)
  ref=p[0].float()
  for j in range(1,p.shape[0]):ref=ref+p[j].float()
  ref=ref.bfloat16();out=torch.full((48*64,),float('nan'),device='cuda',dtype=torch.bfloat16)
  mod.run(p,out,s,bf);torch.cuda.synchronize()
  want=torch.cat((ref[:,:64].flatten(),ref[:,:64].flatten(),ref[:4,:64].flatten(),ref[:4,:64].flatten()))
  ok=torch.equal(out.view(torch.int16),want.view(torch.int16));rows.append(dict(s=s,bf=bf,bit_exact=ok));assert ok,rows[-1]
(H/'reader_validation.json').write_text(json.dumps(rows,indent=2)+'\n');print('READERS',rows)
