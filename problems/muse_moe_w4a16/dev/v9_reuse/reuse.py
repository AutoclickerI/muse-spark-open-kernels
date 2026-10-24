"""V9 fragment reuse with independent channel-tile CTAs; no prepared dequantized bank.

mode 0: decode into registers once per tile and feed up to 128 token rows.
mode 1: decode K=2048 into shared BF16, ldmatrix, retain row accumulators.
mode 2 (Hopper): decode all K into shared BF16, ldmatrix, process row blocks serially.
Scales are byte-for-byte E4M3 layout repacks. Global scales remain FP32.
"""
import hashlib
import torch
import triton
import triton.language as tl

REUSE_CUDA = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <stdint.h>
namespace {
constexpr int HD=4096, NW=4, TH=128, TB=36864, WB=32768;
__device__ __forceinline__ uint32_t rotr(uint32_t x,int s){return __funnelshift_r(x,x,s);}
__device__ __forceinline__ void dq8(uint32_t A,uint32_t B,uint32_t* o){
 o[0]=A&0x81C081C0u; o[1]=rotr(A,4)&0x81C081C0u;
 o[2]=rotr(A,10)&0x81C081C0u; o[3]=rotr(B,2)&0x81C081C0u;
 o[4]=rotr(B,6)&0x81C081C0u; o[5]=rotr(B,12)&0x81C081C0u;
 uint32_t W=(A&0x60306030u)|(B&~0x60306030u);
 o[6]=rotr(W,24)&0x81C081C0u; o[7]=rotr(W,30)&0x81C081C0u;
}
// Raw E4M3 -> BF16(s * 2^118), including zero, sign and finite subnormals.
__device__ __forceinline__ uint32_t sf_bits(uint32_t raw){
 uint32_t mag=raw&127,b=0x7700u+(mag<<4);
 if(mag<8){if(mag){int k=31-__clz(mag);b=((236+k)<<7)|((mag<<(7-k))&127);}else b=0;}
 b|=(raw&128)<<8;return b|(b<<16);
}
// Each byte is one E4M3; normal conversion is byte-permute + IMAD, with exact subnormal handling.
__device__ __forceinline__ uint32_t sf_dup(uint32_t sw,int shift){return sf_bits((sw>>shift)&255u);}
__device__ __forceinline__ uint32_t mul2(uint32_t a,uint32_t b){
 uint32_t d;asm("mul.rn.bf16x2 %0,%1,%2;":"=r"(d):"r"(a),"r"(b));return d;
}
__device__ __forceinline__ void mma(float* c,uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3,uint32_t b0,uint32_t b1){
 asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
 :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}
__device__ __forceinline__ uint4 load_a(const uint32_t* p){
 uint4 a;uint32_t s=(uint32_t)__cvta_generic_to_shared(p);
 asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];":"=r"(a.x),"=r"(a.y),"=r"(a.z),"=r"(a.w):"r"(s));return a;
}
__device__ __forceinline__ void decoded(const uint8_t* tile,int wg,int g,int lane,int s,uint4 wv,uint4& a){
 const uint8_t* sc=tile+WB+(wg*32+lane)*16+g*2;
 uint32_t s0=sf_bits(sc[0]),s1=sf_bits(sc[1]),o[16];
 dq8(wv.x,wv.y,o);dq8(wv.z,wv.w,o+8);
 const uint32_t* q=o+(s>>1)*8+(s&1)*4;
 a=make_uint4(mul2(q[0],s0),mul2(q[1],s1),mul2(q[2],s0),mul2(q[3],s1));
}
template<int NT>
__device__ __forceinline__ void load_bs(uint32_t (&bf)[NT][8],const __nv_bfloat16* x,const int* pairs,int pos,int ne,int rowbase,int pass,int wg,int g,int lane){
 #pragma unroll
 for(int j=0;j<NT;++j){int row=rowbase+j*8+(lane>>2);uint4 v0=make_uint4(0,0,0,0),v1=v0;
  if(row<ne){int p=pos+row,src=pass?p:pairs[p]/8;const uint4* q=(const uint4*)(x+(size_t)src*HD+wg*512+g*64+(lane&3)*16);v0=__ldg(q);v1=__ldg(q+1);}
  bf[j][0]=v0.x;bf[j][1]=v0.y;bf[j][2]=v0.z;bf[j][3]=v0.w;
  bf[j][4]=v1.x;bf[j][5]=v1.y;bf[j][6]=v1.z;bf[j][7]=v1.w;
 }
}
// One deterministic routing pass, used by both GEMMs. Hot experts first reduce the tail.
__global__ void route_k(const int* ids,int* counts,int* offs,int* order,int* pairs,int* pos,int* tasks,int M,int start,int nt){
 __shared__ int n[32],off[32],run[32],wc[4][32],ord[32];
 int t=threadIdx.x,l=t&31,w=t>>5,P=M*8;
 if(t<32)n[t]=0;__syncthreads();
 for(int i=t;i<P;i+=TH){int e=ids[i]-start;pos[i]=-1;if(e>=0&&e<32)atomicAdd(n+e,1);}
 __syncthreads();
 if(w==0){int v=n[l];for(int d=1;d<32;d<<=1){int a=__shfl_up_sync(~0u,v,d);if(l>=d)v+=a;}
  off[l]=v-n[l];run[l]=off[l];counts[l]=n[l];offs[l]=off[l];if(l==31)offs[32]=v;
  int rank=0;for(int e=0;e<32;++e)rank+=(n[e]>n[l]||(n[e]==n[l]&&e<l));ord[rank]=l;}
 __syncthreads();
 for(int base=0;base<P;base+=TH){wc[w][l]=0;__syncthreads();int i=base+t,e=i<P?ids[i]-start:-1;
  if(e<0||e>=32)e=-1;unsigned mask=__match_any_sync(~0u,e);int rank=__popc(mask&((1u<<l)-1));
  if(e>=0&&rank==0)wc[w][e]=__popc(mask);__syncthreads();
  if(t<32){int cur=run[t];for(int k=0;k<4;++k){int a=wc[k][t];wc[k][t]=cur;cur+=a;}run[t]=cur;}
  __syncthreads();if(e>=0){int p=wc[w][e]+rank;pos[i]=p;pairs[p]=i;}__syncthreads();
 }
 if(t==0){int u=0;tasks[0]=0;for(int a=0;a<32;++a){order[a]=ord[a];u+=(n[ord[a]]+8*nt-1)/(8*nt);tasks[a+1]=u;}}
}
// Each unit is (expert, row group, output-channel tile). No cross-CTA spin waits.
// Trace: start/end clocks,busy,dequant, MMA+activation cycles,tasks,rowblocks,max rows,
//        packed weight bytes,activation payload bytes,dequant K16 fragments,reserved.
template<int NT,int MODE>
__global__ void gemm_k(const uint8_t* bank,const float* gs,const __nv_bfloat16* x,__nv_bfloat16* inter,float* y,
 const int* counts,const int* offs,const int* order,const int* pairs,const int* tasks,int pass,int grid,unsigned long long* trace){
 extern __shared__ __align__(128) uint32_t sm[];
 int t=threadIdx.x,l=t&31,w=t>>5,ntile=pass?256:512;
 unsigned long long begin=trace?clock64():0,busy=0,dqcy=0,mmcy=0,njob=0,nrow=0,hot=0,ab=0,frag=0;
 int total=tasks[32]*ntile;
 for(int task=blockIdx.x;task<total;task+=grid){
  unsigned long long t0=trace?clock64():0;
  int group=task/ntile,m=task%ntile,a=0;
  for(int d=16;d;d>>=1)if(tasks[a+d]<=group)a+=d;
  int e=order[a],nb=group-tasks[a],ne=min(NT*8,counts[e]-nb*NT*8),off=offs[e]+nb*NT*8;
  const uint8_t* tile=bank+((size_t)e*ntile+m)*TB;
  float acc[NT][4];
  #pragma unroll
  for(int j=0;j<NT;++j)for(int k=0;k<4;++k)acc[j][k]=0;
  if constexpr(MODE==0){
   #pragma unroll
   for(int h=0;h<2;++h){int wg=w+4*h;
    #pragma unroll
    for(int g=0;g<8;++g){uint32_t bf[NT][8];load_bs(bf,x,pairs,off,ne,0,pass,wg,g,l);uint4 v=__ldg((const uint4*)tile+(wg*8+g)*32+l);
     #pragma unroll
     for(int s=0;s<4;++s){uint4 av;decoded(tile,wg,g,l,s,v,av);
      #pragma unroll
      for(int j=0;j<NT;++j)if(j*8<ne){mma(acc[j],av.x,av.y,av.z,av.w,bf[j][2*s],bf[j][2*s+1]);}
     }
    }
   }
  }else if constexpr(MODE==1){
   #pragma unroll
   for(int h=0;h<2;++h){int wg=w+4*h;unsigned long long td=trace?clock64():0;
    #pragma unroll
    for(int g=0;g<8;++g){uint4 v=__ldg((const uint4*)tile+(wg*8+g)*32+l);
     #pragma unroll
     for(int s=0;s<4;++s){uint4 av;decoded(tile,wg,g,l,s,v,av);uint32_t* p=sm+((w*8+g)*4+s)*128+(l>>2)*4+(l&3);
      p[0]=av.x;p[32]=av.y;p[64]=av.z;p[96]=av.w;
     }
    }
    __syncthreads();unsigned long long tm=trace?clock64():0;if(trace)dqcy+=tm-td;
    #pragma unroll
    for(int g=0;g<8;++g){uint32_t bf[NT][8];load_bs(bf,x,pairs,off,ne,0,pass,wg,g,l);
     #pragma unroll
     for(int s=0;s<4;++s){const uint32_t* p=sm+((w*8+g)*4+s)*128+(l>>3)*32+(l&7)*4;uint4 av=load_a(p);
      #pragma unroll
      for(int j=0;j<NT;++j)if(j*8<ne){mma(acc[j],av.x,av.y,av.z,av.w,bf[j][2*s],bf[j][2*s+1]);}
     }
    }
    __syncthreads();if(trace)mmcy+=clock64()-tm;
   }
  }else{
   unsigned long long td=trace?clock64():0;
   #pragma unroll
   for(int h=0;h<2;++h){int wg=w+4*h;
    #pragma unroll
    for(int g=0;g<8;++g){uint4 v=__ldg((const uint4*)tile+(wg*8+g)*32+l);
     #pragma unroll
     for(int s=0;s<4;++s){uint4 av;decoded(tile,wg,g,l,s,v,av);uint32_t* p=sm+((wg*8+g)*4+s)*128+(l>>2)*4+(l&3);
      p[0]=av.x;p[32]=av.y;p[64]=av.z;p[96]=av.w;
     }
    }
   }
   __syncthreads();unsigned long long tm=trace?clock64():0;if(trace)dqcy+=tm-td;
   // NT=1 here means all row blocks, serially, after a single full-tile decode.
   for(int j=0;j<(ne+7)/8;++j){float c[4]={0,0,0,0};
    #pragma unroll
    for(int h=0;h<2;++h){int wg=w+4*h;
     #pragma unroll
     for(int g=0;g<8;++g){uint32_t bf[1][8];load_bs(bf,x,pairs,off,ne,j*8,pass,wg,g,l);
      #pragma unroll
      for(int s=0;s<4;++s){uint4 av=load_a(sm+((wg*8+g)*4+s)*128+(l>>3)*32+(l&7)*4);
       mma(c,av.x,av.y,av.z,av.w,bf[0][2*s],bf[0][2*s+1]);
      }
     }
    }
    // Reduction buffer is separate: cached weights survive every row block.
    float* red=(float*)(sm+32768);((float4*)red)[w*32+l]=make_float4(c[0],c[1],c[2],c[3]);__syncthreads();
    int work=(pass?16:8)*8;
    if(t<work){int i=t%(pass?16:8),n=t/(pass?16:8),ls=(i&7)*4+(n>>1),rg=(n&1)+((i>>3)<<1);float v=0,u=0;
     #pragma unroll
     for(int ww=0;ww<4;++ww){v+=red[(ww*32+ls)*4+rg];if(!pass)u+=red[(ww*32+ls)*4+rg+2];}
     int row=j*8+n;if(row<ne){float sc=gs[e]*256.f;v*=sc;
      if(pass)y[(size_t)(off+row)*HD+m*16+i]=v;
      else{u*=sc;inter[(size_t)(off+row)*HD+m*8+i]=__float2bfloat16(v/(1.f+__expf(-v))*u);}
     }
    }
    __syncthreads();
   }
   if(trace)mmcy+=clock64()-tm;
  }
  if constexpr(MODE!=2){
   __syncthreads();float* red=(float*)sm;
   #pragma unroll
   for(int j=0;j<NT;++j)((float4*)red)[(w*32+l)*NT+j]=make_float4(acc[j][0],acc[j][1],acc[j][2],acc[j][3]);
   __syncthreads();int width=pass?16:8;
   for(int z=t;z<NT*8*width;z+=TH){int i=z%width,n=z/width,j=n>>3,col=n&7,ls=(i&7)*4+(col>>1),rg=j*4+(col&1)+((i>>3)<<1);float v=0,u=0;
    #pragma unroll
    for(int ww=0;ww<4;++ww){v+=red[(ww*32+ls)*NT*4+rg];if(!pass)u+=red[(ww*32+ls)*NT*4+rg+2];}
    if(n<ne){float sc=gs[e]*256.f;v*=sc;if(pass)y[(size_t)(off+n)*HD+m*16+i]=v;
     else{u*=sc;inter[(size_t)(off+n)*HD+m*8+i]=__float2bfloat16(v/(1.f+__expf(-v))*u);}}
   }
   __syncthreads();
  }
  if(trace){busy+=clock64()-t0;++njob;nrow+=(ne+7)/8;hot=max(hot,(unsigned long long)ne);ab+=(unsigned long long)ne*4096*2;frag+=256;}
 }
 if(trace&&t==0){auto p=trace+(size_t)blockIdx.x*12;p[0]=begin;p[1]=clock64();p[2]=busy;p[3]=dqcy;p[4]=mmcy;p[5]=njob;p[6]=nrow;p[7]=hot;p[8]=njob*TB;p[9]=ab;p[10]=frag;p[11]=MODE;}
}
}
void reuse_route(torch::Tensor ids,torch::Tensor counts,torch::Tensor offs,torch::Tensor order,torch::Tensor pairs,torch::Tensor pos,torch::Tensor tasks,int64_t start,int64_t nt){
 route_k<<<1,128,0,at::cuda::getCurrentCUDAStream()>>>(ids.data_ptr<int>(),counts.data_ptr<int>(),offs.data_ptr<int>(),order.data_ptr<int>(),pairs.data_ptr<int>(),pos.data_ptr<int>(),tasks.data_ptr<int>(),ids.size(0),start,nt);C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void reuse_gemm(torch::Tensor bank,torch::Tensor gs,torch::Tensor x,torch::Tensor inter,torch::Tensor y,torch::Tensor counts,torch::Tensor offs,torch::Tensor order,torch::Tensor pairs,torch::Tensor tasks,torch::Tensor trace,int64_t pass,int64_t nt,int64_t mode,int64_t grid){
 auto stream=at::cuda::getCurrentCUDAStream();auto tr=trace.numel()?reinterpret_cast<unsigned long long*>(trace.data_ptr<int64_t>()):nullptr;
 #define GO(NT,MODE) {int sm=(MODE==2?133120:(MODE==1?65536:NT*2048));C10_CUDA_CHECK(cudaFuncSetAttribute(gemm_k<NT,MODE>,cudaFuncAttributeMaxDynamicSharedMemorySize,sm));gemm_k<NT,MODE><<<grid,128,sm,stream>>>(bank.data_ptr<uint8_t>(),gs.data_ptr<float>(),(const __nv_bfloat16*)x.data_ptr(),(__nv_bfloat16*)inter.data_ptr(),y.data_ptr<float>(),counts.data_ptr<int>(),offs.data_ptr<int>(),order.data_ptr<int>(),pairs.data_ptr<int>(),tasks.data_ptr<int>(),pass,grid,tr);}
 // Full shared tile processes one row group of at most 256 rows sequentially.
 if(mode==2){GO(32,2);}
 else if(mode==1){if(nt==4){GO(4,1);}else if(nt==8){GO(8,1);}else if(nt==16){GO(16,1);}else{TORCH_CHECK(false,"shared NT must be 4/8/16");}}
 else if(mode==0){if(nt==1){GO(1,0);}else if(nt==2){GO(2,0);}else if(nt==4){GO(4,0);}else if(nt==8){GO(8,0);}else if(nt==16){GO(16,0);}else{TORCH_CHECK(false,"register NT must be 1/2/4/8/16");}}
 else{TORCH_CHECK(false,"unknown reuse mode");}
 #undef GO
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
'''
REUSE_CPP = r'''
#include <torch/extension.h>
void reuse_route(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t);
void reuse_gemm(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t);
'''
_REUSE_MOD=None


def reuse_module():
    global _REUSE_MOD
    if _REUSE_MOD is None:
        from torch.utils.cpp_extension import load_inline
        tag=hashlib.sha256(REUSE_CUDA.encode()).hexdigest()[:12]
        _REUSE_MOD=load_inline(name='muse_v9_reuse_'+tag,cpp_sources=REUSE_CPP,cuda_sources=REUSE_CUDA,
                              functions=['reuse_route','reuse_gemm'],extra_cuda_cflags=['-O3','-std=c++20','-lineinfo'])
    return _REUSE_MOD


@triton.jit
def reuse_norm(Y,P,O,R,P_MAX:tl.constexpr,H:tl.constexpr=4096):
    p=tl.program_id(0)
    if p<tl.load(O+32):
        k=tl.arange(0,H)
        v=tl.load(Y+p*H+k)*(1+tl.load(P+k).to(tl.float32))
        tl.store(R+p,tl.rsqrt(tl.sum(v*v,0)/H+1e-8))


@triton.jit
def reuse_combine(Y,P,W,POS,R,OUT,H:tl.constexpr=4096):
    t=tl.program_id(0);k=tl.arange(0,H);f=1+tl.load(P+k).to(tl.float32)
    z=tl.full((H,),0,tl.float32)
    for j in range(8):
        p=tl.load(POS+t*8+j);ok=p>=0
        v=tl.load(Y+tl.maximum(p,0)*H+k,ok,0)
        r=tl.load(R+tl.maximum(p,0),ok,0);w=tl.load(W+t*8+j)
        z+=v*f*r*w
    tl.store(OUT+t*H+k,z)


def reuse_work(data,bank,choice,trace=False):
    nt,mode,mult=choice
    if mode==2:nt=32
    x=data['hidden_states'];m=x.shape[0];dev=x.device;p=m*8
    counts=torch.empty(32,device=dev,dtype=torch.int32);offs=torch.empty(33,device=dev,dtype=torch.int32)
    order=torch.empty(32,device=dev,dtype=torch.int32);pairs=torch.empty(p,device=dev,dtype=torch.int32)
    pos=torch.empty(p,device=dev,dtype=torch.int32);tasks=torch.empty(33,device=dev,dtype=torch.int32)
    inter=torch.empty((p,4096),device=dev,dtype=torch.bfloat16)
    y=torch.empty((p,4096),device=dev,dtype=torch.float32);inv=torch.empty(p,device=dev,dtype=torch.float32)
    out=torch.empty_like(x);grid=bank['grid']*mult
    stamps=[torch.empty((grid,12),device=dev,dtype=torch.int64) if trace else torch.empty(0,device=dev,dtype=torch.int64) for _ in range(2)]
    mod=bank['reuse_mod']
    def route():
        mod.reuse_route(data['topk_ids'],counts,offs,order,pairs,pos,tasks,data['config']['local_start'],nt)
    def first():
        mod.reuse_gemm(bank['w13p'],bank['gs13'],x,inter,y,counts,offs,order,pairs,tasks,stamps[0],0,nt,mode,grid)
    def second():
        mod.reuse_gemm(bank['w2p'],bank['gs2'],inter,inter,y,counts,offs,order,pairs,tasks,stamps[1],1,nt,mode,grid)
    def norm():
        reuse_norm[(p,)](y,bank['pnw'],offs,inv,p,num_warps=4)
    def combine():
        reuse_combine[(m,)](y,bank['pnw'],data['topk_weights'],pos,inv,out,num_warps=4)
        return out
    return (route,first,second,norm,combine),stamps


def reuse_call(data,bank,choice):
    phases,_=reuse_work(data,bank,choice)
    for f in phases:out=f()
    return out


_DQ_SLOTS = (("A", 0), ("A", 4), ("A", 10), ("B", 2), ("B", 6), ("B", 12), ("W", 24), ("W", 30))
_LA_MASK = 0x60306030


def _bit_table():
    """(slot, half, code bit, word, bit position) for the kernel's dq8 decode."""
    dbit = {3: 15, 2: 8, 1: 7, 0: 6}
    la = {p for p in range(32) if (_LA_MASK >> p) & 1}
    table, used = [], {"A": set(), "B": set()}
    for k, (src, rot) in enumerate(_DQ_SLOTS):
        for h in (0, 1):
            for b in range(4):
                p = (dbit[b] + 16 * h + rot) % 32
                w = src if src != "W" else ("A" if p in la else "B")
                assert p not in used[w]
                used[w].add(p)
                table.append((k, h, b, w, p))
    assert len(used["A"]) == 32 and len(used["B"]) == 32
    return table


_TABLE = _bit_table()


def _repack(wq: torch.Tensor, sf: torch.Tensor, rowmap: torch.Tensor) -> torch.Tensor:
    """One expert: packed E2M1 [R, K/2] u8 + E4M3 scales [R, K/16] -> [T, 36864] u8 tiles."""
    R, K = wq.shape[0], wq.shape[1] * 2
    assert K == 4096
    T = rowmap.shape[0]
    codes = torch.stack((wq & 0xF, wq >> 4), -1).reshape(R, K // 16, 16)
    u = sf.view(torch.uint8)  # E4M3 bytes: layout change only
    codes = codes.reshape(R, K)[rowmap.reshape(-1)]
    # k = 512 w + 64 g + 16 q + 4 (2 p + sl) + 2 j + h ; tile row = 8 ri + rr
    ct = codes.view(T, 2, 8, 8, 8, 4, 2, 2, 2, 2)                           # T ri rr w g q p sl j h
    ct = ct.permute(0, 3, 4, 2, 5, 6, 7, 8, 1, 9).reshape(T, 8, 8, 32, 2, 8, 2).to(torch.int64)
    words = torch.zeros((2, T, 8, 8, 32, 2), dtype=torch.int64, device=wq.device)
    for k, h, b, w, p in _TABLE:
        words[0 if w == "A" else 1] |= ((ct[..., k, h] >> b) & 1) << p
    words = words.permute(1, 2, 3, 4, 5, 0).contiguous()                    # T w g lane p {A,B}
    wbytes = words.view(torch.uint8).view(T, 8, 8, 32, 2, 2, 8)[..., :4].reshape(T, 32768)
    us = u[rowmap.reshape(-1)].view(T, 2, 8, 8, 8, 4)                       # T ri rr w g q
    sbytes = us.permute(0, 3, 2, 5, 4, 1).reshape(T, 4096)                  # T w (rr q) g ri
    return torch.cat((wbytes, sbytes), 1).contiguous()



def reuse_prepare(weights):
    dev=weights['w13'].device
    t=torch.arange(512,device=dev);i=torch.arange(8,device=dev)
    maps=(torch.cat((8*t[:,None]+i,4096+8*t[:,None]+i),1),torch.arange(4096,device=dev).view(256,16))
    result={}
    for name,nt,rowmap in zip(('w13','w2'),(512,256),maps):
        dst=torch.empty((32,nt,36864),device=dev,dtype=torch.uint8)
        for e in range(32):dst[e]=_repack(weights[name][e],weights[name+'_sf'][e],rowmap)
        result[name+'p']=dst
    result.update(gs13=weights['w13_gs'].float().contiguous(),gs2=weights['w2_gs'].float().contiguous(),
                  pnw=weights['post_norm_w'].contiguous(),grid=torch.cuda.get_device_properties(dev).multi_processor_count,reuse_mod=reuse_module())
    return result


def _reuse_fp32_impl(data,weights):
    """Independent agreement oracle; scratch weights die here and never enter returned state."""
    x=data['hidden_states'];ids=data['topk_ids'];tw=data['topk_weights'];start=data['config']['local_start']
    lut=torch.tensor((0.,.5,1.,1.5,2.,3.,4.,6.),device=x.device)
    out=torch.zeros_like(x,dtype=torch.float32)
    for e in range(32):
        t,j=torch.where(ids==e+start)
        if t.numel()==0:continue
        def dq(name):
            q=weights[name][e]
            code=torch.stack((q&15,q>>4),-1).flatten(-2).long()
            return lut[code&7]*torch.where((code&8)!=0,-1.,1.)*weights[name+'_sf'][e].float().repeat_interleave(16,-1)*weights[name+'_gs'][e]
        a=x[t].float()@dq('w13').t()
        a=(torch.nn.functional.silu(a[:,:4096])*a[:,4096:]).bfloat16()
        y=a.float()@dq('w2').t()
        y*=1+weights['post_norm_w'].float()
        y*=torch.rsqrt(y.square().mean(-1,keepdim=True)+1e-8)
        out.index_add_(0,t,y*tw[t,j,None])
    return out.bfloat16()


def reuse_relative(got,want):
    return float((got.float()-want.float()).norm()/want.float().norm().clamp_min(1e-12))


def reuse_fp32(data,weights):
    old=torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32=False
        return _reuse_fp32_impl(data,weights)
    finally:
        torch.backends.cuda.matmul.allow_tf32=old
