"""V9 activation-cached channel chunks on a hot-first device work queue.
Physical launch capacity is fixed for graph capture; route counts choose active CTAs and jobs.
Raw NVFP4/E4M3 bank; per-call counts, sorting and all expert arithmetic stay on the GPU.
"""
import hashlib
import torch
import triton
import triton.language as tl

BAL_CUDA = '\n#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n\nnamespace {\n\nconstexpr int HD = 4096;\nconstexpr int NLOC = 32;\nconstexpr int TILE_BYTES = 36864;\nconstexpr int TILE_W_BYTES = 32768;\nconstexpr int NT1 = 512;\nconstexpr int NT2 = 256;\nconstexpr int NCW = 8;                 // consumer warps\nconstexpr int NCT = NCW * 32;          // consumer threads\nconstexpr int NTHR = NCT;             // thread 0 also prefetches\nconstexpr uint32_t DQ_MASK = 0x81C081C0u;\nconstexpr uint32_t LA_MASK = 0x60306030u;\n// persistent counters\nenum { C_DONE = 0, C_ZERO = 32, C_BAR = 33, C_GEN = 34 };\n\n__device__ __forceinline__ uint32_t smem_u32(const void* p) {\n  return static_cast<uint32_t>(__cvta_generic_to_shared(p));\n}\n__device__ __forceinline__ void mbar_init(uint64_t* b, int cnt) {\n  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(b)), "r"(cnt));\n}\n__device__ __forceinline__ bool mbar_try(uint64_t* b, uint32_t parity) {\n  uint32_t ok;\n  asm volatile("{\\n .reg .pred p;\\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\\n selp.u32 %0, 1, 0, p;\\n}\\n"\n               : "=r"(ok) : "r"(smem_u32(b)), "r"(parity) : "memory");\n  return ok != 0;\n}\n__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t parity) {\n  while (!mbar_try(b, parity)) {}\n}\n__device__ __forceinline__ void mbar_arrive(uint64_t* b) {\n  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(b)) : "memory");\n}\n__device__ __forceinline__ void mbar_expect_tx(uint64_t* b, uint32_t bytes) {\n  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(b)), "r"(bytes) : "memory");\n}\n__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* b, uint64_t pol) {\n  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"\n               :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(b)), "l"(pol) : "memory");\n}\n__device__ __forceinline__ void bar_consumers() {\n  asm volatile("bar.sync 1, %0;" :: "n"(NCT) : "memory");\n}\n__device__ __forceinline__ int ld_acquire(const int* p) {\n  int v;\n  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");\n  return v;\n}\n__device__ __forceinline__ uint32_t rotr(uint32_t x, int s) { return __funnelshift_r(x, x, s); }\n__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {\n  uint32_t d;\n  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));\n  return d;\n}\n__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,\n                                         uint32_t b0, uint32_t b1) {\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"\n               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])\n               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));\n}\n// 16 E2M1 codes (two words) -> 8 BF16x2 registers holding value * 2^-126 (exact).\n__device__ __forceinline__ void dq8(uint32_t A, uint32_t B, uint32_t* o) {\n  o[0] = A & DQ_MASK;\n  o[1] = rotr(A, 4) & DQ_MASK;\n  o[2] = rotr(A, 10) & DQ_MASK;\n  o[3] = rotr(B, 2) & DQ_MASK;\n  o[4] = rotr(B, 6) & DQ_MASK;\n  o[5] = rotr(B, 12) & DQ_MASK;\n  const uint32_t W = (A & LA_MASK) | (B & ~LA_MASK);\n  o[6] = rotr(W, 24) & DQ_MASK;\n  o[7] = rotr(W, 30) & DQ_MASK;\n}\n\n// Raw E4M3 -> BF16(s * 2^118), including zero, sign and finite subnormals.\n__device__ __forceinline__ uint32_t sf_bits(uint32_t raw){\n uint32_t mag=raw&127, exp=mag>>3, m=mag&7, b=0;\n if(exp) b=((exp+238)<<7)|(m<<4);\n else if(m){int k=31-__clz(m);b=((236+k)<<7)|((m<<(7-k))&127);}\n b|=(raw&128)<<8; return b|(b<<16);\n}\n\n__global__ void balance_route_k(const int* ids,int* counts,int* offs,int* order,int* pairs,int* pos,int* tasks,int M,int start,int nt,int chunk,int hot,int cap){\n __shared__ int n[32],off[32],run[32],wc[4][32],ord[32];\n int t=threadIdx.x,l=t&31,w=t>>5,P=M*8;\n if(t<32)n[t]=0;__syncthreads();\n for(int i=t;i<P;i+=128){int e=ids[i]-start;pos[i]=-1;if(e>=0&&e<32)atomicAdd(n+e,1);}\n __syncthreads();\n if(w==0){int v=n[l];for(int d=1;d<32;d<<=1){int a=__shfl_up_sync(~0u,v,d);if(l>=d)v+=a;}\n  off[l]=v-n[l];run[l]=off[l];counts[l]=n[l];offs[l]=off[l];if(l==31)offs[32]=v;\n  int rank=0;for(int e=0;e<32;++e)rank+=(n[e]>n[l]||(n[e]==n[l]&&e<l));ord[rank]=l;}\n __syncthreads();\n for(int base=0;base<P;base+=128){wc[w][l]=0;__syncthreads();int i=base+t,e=i<P?ids[i]-start:-1;\n  if(e<0||e>=32)e=-1;unsigned mask=__match_any_sync(~0u,e);int rank=__popc(mask&((1u<<l)-1));\n  if(e>=0&&rank==0)wc[w][e]=__popc(mask);__syncthreads();\n  if(t<32){int cur=run[t];for(int k=0;k<4;++k){int a=wc[k][t];wc[k][t]=cur;cur+=a;}run[t]=cur;}\n  __syncthreads();if(e>=0){int p=wc[w][e]+rank;pos[i]=p;pairs[p]=i;}__syncthreads();\n }\n if(t==0){int u1=0,u2=0;tasks[0]=0;tasks[33]=0;\n  for(int a=0;a<32;++a){int e=hot?ord[a]:a;order[a]=e;int nb=(n[e]+8*nt-1)/(8*nt);\n   u1+=nb*((512+chunk-1)/chunk);u2+=nb*((256+chunk/2-1)/(chunk/2));tasks[a+1]=u1;tasks[33+a+1]=u2;}\n  tasks[66]=min(cap,u1);tasks[67]=min(cap,u2);tasks[68]=0;tasks[69]=0;\n }\n}\n\nstruct Job {int e,nb,m0,m1,ne,off,task;};\ntemplate<int NT>\n__global__ void __launch_bounds__(NTHR, NT==1?2:1) balance_gemm_k(\n const __nv_bfloat16* x,const uint8_t* bank,const float* gs,__nv_bfloat16* inter,float* y,\n const int* counts,const int* offs,const int* order,const int* pairs,int* plan,int pass,int chunk,int nstage){\n constexpr int NB=8*NT,RED=4*NT;\n extern __shared__ __align__(128) uint8_t sm[];\n uint8_t* tiles=sm;float* red=(float*)(sm+nstage*TILE_BYTES);\n uint64_t* bars=(uint64_t*)(red+NCW*32*RED);Job* job=(Job*)(bars+nstage);\n int tid=threadIdx.x,lane=tid&31,warp=tid>>5,q=lane&3;\n int total=plan[pass?65:32],active=plan[66+pass];if(blockIdx.x>=active)return;\n uint64_t pol=0;uint32_t ph[2]={0,0};\n if(tid==0){for(int s=0;s<nstage;++s)mbar_init(bars+s,1);asm volatile("fence.mbarrier_init.release.cluster;":::"memory");}\n asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;":"=l"(pol));\n __syncthreads();\n while(true){\n  if(tid==0){job->task=atomicAdd(plan+68+pass,1);if(job->task<total){\n   int a=0;const int* pref=plan+(pass?33:0);\n   for(int d=16;d;d>>=1)if(pref[a+d]<=job->task)a+=d;\n   int e=order[a],ntile=pass?256:512,nchunk=(ntile+chunk-1)/chunk,rel=job->task-pref[a];\n   int nb=rel/nchunk,m0=(rel%nchunk)*chunk;\n   job->e=e;job->nb=nb;job->m0=m0;job->m1=min(m0+chunk,ntile);job->ne=min(NB,counts[e]-nb*NB);job->off=offs[e]+nb*NB;\n  }}\n  __syncthreads();if(job->task>=total)break;\n  const int e=job->e,off=job->off,ne=job->ne,m0=job->m0,m1=job->m1;\n  const int ntile=pass?256:512;\n  uint32_t Bf[NT][8][8];\n  #pragma unroll\n  for(int j=0;j<NT;++j){int row=j*8+(lane>>2);const __nv_bfloat16* xp=nullptr;\n   if(row<ne){int p=off+row;xp=x+(size_t)(pass?p:pairs[p]/8)*HD;}\n   #pragma unroll\n   for(int g=0;g<8;++g){uint4 a=make_uint4(0,0,0,0),b=a;if(xp){const uint4* bp=(const uint4*)(xp+warp*512+g*64+q*16);if(pass){a=__ldcg(bp);b=__ldcg(bp+1);}else{a=__ldg(bp);b=__ldg(bp+1);}}\n    Bf[j][g][0]=a.x;Bf[j][g][1]=a.y;Bf[j][g][2]=a.z;Bf[j][g][3]=a.w;\n    Bf[j][g][4]=b.x;Bf[j][g][5]=b.y;Bf[j][g][6]=b.z;Bf[j][g][7]=b.w;\n   }\n  }\n  if(tid==0){for(int s=0;s<nstage&&m0+s<m1;++s){mbar_expect_tx(bars+s,TILE_BYTES);bulk_g2s(tiles+s*TILE_BYTES,bank+((size_t)e*ntile+m0+s)*TILE_BYTES,TILE_BYTES,bars+s,pol);}}\n  __syncthreads();\n  for(int m=m0;m<m1;++m){int st=(m-m0)%nstage;mbar_wait(bars+st,ph[st]);ph[st]^=1;\n   const uint8_t* tile=tiles+st*TILE_BYTES;float acc[NT][4];\n   #pragma unroll\n   for(int j=0;j<NT;++j)for(int k=0;k<4;++k)acc[j][k]=0;\n   const uint4* wq=(const uint4*)tile+warp*8*32+lane;\n   uint4 scv=((const uint4*)(tile+TILE_W_BYTES))[warp*32+lane];uint32_t scw[4]={scv.x,scv.y,scv.z,scv.w};\n   #pragma unroll\n   for(int g=0;g<8;++g){uint4 wv=wq[g*32];uint32_t sw=scw[g>>1];\n    uint32_t s0=sf_bits((sw>>((g&1)*16))&255),s1=sf_bits((sw>>((g&1)*16+8))&255),o[16];dq8(wv.x,wv.y,o);dq8(wv.z,wv.w,o+8);\n    #pragma unroll\n    for(int s=0;s<4;++s){const uint32_t* os=o+(s>>1)*8+(s&1)*4;\n     uint32_t a0=hmul2(os[0],s0),a1=hmul2(os[1],s1),a2=hmul2(os[2],s0),a3=hmul2(os[3],s1);\n     #pragma unroll\n     for(int j=0;j<NT;++j)if(j*8<ne)mma16816(acc[j],a0,a1,a2,a3,Bf[j][g][2*s],Bf[j][g][2*s+1]);\n    }\n   }\n   float* rp=red+(warp*32+lane)*RED;\n   #pragma unroll\n   for(int j=0;j<NT;++j)((float4*)rp)[j]=make_float4(acc[j][0],acc[j][1],acc[j][2],acc[j][3]);\n   __syncthreads();\n   if(tid==0&&m+nstage<m1){mbar_expect_tx(bars+st,TILE_BYTES);bulk_g2s(tiles+st*TILE_BYTES,bank+((size_t)e*ntile+m+nstage)*TILE_BYTES,TILE_BYTES,bars+st,pol);}\n   int width=pass?16:8;\n   for(int z=tid;z<NB*width;z+=NTHR){int i=z%width,n=z/width,j=n>>3,col=n&7,ls=(i&7)*4+(col>>1),rg=j*4+(col&1)+((i>>3)<<1);float v=0,u=0;\n    #pragma unroll\n    for(int w=0;w<NCW;++w){v+=red[(w*32+ls)*RED+rg];if(!pass)u+=red[(w*32+ls)*RED+rg+2];}\n    if(n<ne){float sc=gs[e]*256.f;v*=sc;if(pass)y[(size_t)(off+n)*HD+m*16+i]=v;\n     else{u*=sc;inter[(size_t)(off+n)*HD+m*8+i]=__float2bfloat16(v/(1.f+__expf(-v))*u);}}\n   }\n   __syncthreads();\n  }\n }\n}\nsize_t balance_smem(int nt,int stages){return stages*TILE_BYTES+NCW*32*nt*4*sizeof(float)+128;}\n}\nint64_t balance_capacity(int64_t nt,int64_t stages){\n int optin,dev,n;C10_CUDA_CHECK(cudaGetDevice(&dev));C10_CUDA_CHECK(cudaDeviceGetAttribute(&optin,cudaDevAttrMaxSharedMemoryPerBlockOptin,dev));\n int sm=balance_smem(nt,stages);TORCH_CHECK(sm<=optin,"balanced shared memory exceeds device capacity");\n #define CAP(N) C10_CUDA_CHECK(cudaFuncSetAttribute(balance_gemm_k<N>,cudaFuncAttributeMaxDynamicSharedMemorySize,sm));C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,balance_gemm_k<N>,NTHR,sm));\n if(nt==1){CAP(1);}else if(nt==2){CAP(2);}else{CAP(4);}\n #undef CAP\n return n;\n}\nvoid balance_route(torch::Tensor ids,torch::Tensor counts,torch::Tensor offs,torch::Tensor order,torch::Tensor pairs,torch::Tensor pos,torch::Tensor plan,int64_t start,int64_t nt,int64_t chunk,int64_t hot,int64_t cap){\n balance_route_k<<<1,128,0,at::cuda::getCurrentCUDAStream()>>>(ids.data_ptr<int>(),counts.data_ptr<int>(),offs.data_ptr<int>(),order.data_ptr<int>(),pairs.data_ptr<int>(),pos.data_ptr<int>(),plan.data_ptr<int>(),ids.size(0),start,nt,chunk,hot,cap);C10_CUDA_KERNEL_LAUNCH_CHECK();\n}\nvoid balance_gemm(torch::Tensor x,torch::Tensor bank,torch::Tensor gs,torch::Tensor inter,torch::Tensor y,torch::Tensor counts,torch::Tensor offs,torch::Tensor order,torch::Tensor pairs,torch::Tensor plan,int64_t pass,int64_t nt,int64_t chunk,int64_t stages,int64_t cap){\n auto stream=at::cuda::getCurrentCUDAStream();int sm=balance_smem(nt,stages);\n #define RUN(N) balance_gemm_k<N><<<cap,NTHR,sm,stream>>>((const __nv_bfloat16*)x.data_ptr(),bank.data_ptr<uint8_t>(),gs.data_ptr<float>(),(__nv_bfloat16*)inter.data_ptr(),y.data_ptr<float>(),counts.data_ptr<int>(),offs.data_ptr<int>(),order.data_ptr<int>(),pairs.data_ptr<int>(),plan.data_ptr<int>(),pass,chunk,stages)\n if(nt==1){RUN(1);}else if(nt==2){RUN(2);}else{RUN(4);}\n #undef RUN\n C10_CUDA_KERNEL_LAUNCH_CHECK();\n}\n'
BAL_CPP=r"""
#include <torch/extension.h>
int64_t balance_capacity(int64_t,int64_t);
void balance_route(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t,int64_t);
void balance_gemm(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t,int64_t);
"""
_BAL_MOD=None
BAL_CHOICES=((1,16,1),(1,32,1),(2,16,1),(2,32,1),(2,64,1),(2,32,0))


def balanced_label(c):return f'b{c[0]}c{c[1]}h{c[2]}'


def balanced_module():
    global _BAL_MOD
    if _BAL_MOD is None:
        from torch.utils.cpp_extension import load_inline
        tag=hashlib.sha256(BAL_CUDA.encode()).hexdigest()[:12]
        _BAL_MOD=load_inline(name='muse_balance_'+tag,cpp_sources=BAL_CPP,cuda_sources=BAL_CUDA,
                            functions=['balance_capacity','balance_route','balance_gemm'],extra_cuda_cflags=['-O3','-std=c++20','-lineinfo'])
    return _BAL_MOD


def balanced_prepare(bank):
    out=dict(bank);out['balanced_mod']=balanced_module();out['balanced_caps']={}
    for nt in (1,2,4):out['balanced_caps'][nt]=bank['grid']*out['balanced_mod'].balance_capacity(nt,2)
    return out


@triton.jit
def balanced_norm(Y,P,O,R,H:tl.constexpr=4096):
    p=tl.program_id(0)
    if p<tl.load(O+32):
        k=tl.arange(0,H);v=tl.load(Y+p*H+k)*(1+tl.load(P+k).to(tl.float32))
        tl.store(R+p,tl.rsqrt(tl.sum(v*v,0)/H+1e-8))


@triton.jit
def balanced_combine(Y,P,W,POS,R,OUT,H:tl.constexpr=4096):
    t=tl.program_id(0);k=tl.arange(0,H);f=1+tl.load(P+k).to(tl.float32);z=tl.full((H,),0,tl.float32)
    for j in range(8):
        p=tl.load(POS+t*8+j);ok=p>=0;v=tl.load(Y+tl.maximum(p,0)*H+k,ok,0)
        r=tl.load(R+tl.maximum(p,0),ok,0);w=tl.load(W+t*8+j);z+=v*f*r*w
    tl.store(OUT+t*H+k,z)


def balanced_work(data,bank,choice):
    nt,chunk,hot=choice;x=data['hidden_states'];m=x.shape[0];p=m*8;dev=x.device
    counts=torch.empty(32,device=dev,dtype=torch.int32);offs=torch.empty(33,device=dev,dtype=torch.int32)
    order=torch.empty(32,device=dev,dtype=torch.int32);pairs=torch.empty(p,device=dev,dtype=torch.int32);pos=torch.empty_like(pairs)
    plan=torch.empty(72,device=dev,dtype=torch.int32);inter=torch.empty((p,4096),device=dev,dtype=torch.bfloat16)
    y=torch.empty((p,4096),device=dev,dtype=torch.float32);inv=torch.empty(p,device=dev,dtype=torch.float32);out=torch.empty_like(x)
    mod=bank['balanced_mod'];cap=bank['balanced_caps'][nt]
    def route():mod.balance_route(data['topk_ids'],counts,offs,order,pairs,pos,plan,data['config']['local_start'],nt,chunk,hot,cap)
    def first():mod.balance_gemm(x,bank['w13p'],bank['gs13'],inter,y,counts,offs,order,pairs,plan,0,nt,chunk,2,cap)
    def second():mod.balance_gemm(inter,bank['w2p'],bank['gs2'],inter,y,counts,offs,order,pairs,plan,1,nt,chunk//2,2,cap)
    def norm():balanced_norm[(p,)](y,bank['pnw'],offs,inv,num_warps=4)
    def combine():
        balanced_combine[(m,)](y,bank['pnw'],data['topk_weights'],pos,inv,out,num_warps=4)
        return out
    return (route,first,second,norm,combine),dict(counts=counts,offs=offs,order=order,plan=plan)


def balanced_call(data,bank,choice):
    phases,_=balanced_work(data,bank,choice)
    for f in phases:out=f()
    return out
