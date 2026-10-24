"""M128 recoded fused MoE cell. Enable only after its paired H200 receipt wins."""
import functools
import hashlib
import torch
from torch.utils.cpp_extension import load_inline
def m128_scales_mw_repack_planes(q):
    """A permutation of all four bits of every nibble, matching RS register ownership.

    Two words encode the 16 codes each lane needs for M16/K32. The extraction
    table is the incumbent V9 bit-plane bijection, used here with WGMMA layout.
    """
    e, n, k2 = q.shape
    k = k2 * 2
    dst = torch.empty((e, n // 16, k // 32, 32, 2), device=q.device, dtype=torch.int32)
    slots = (('A', 0), ('A', 4), ('A', 10), ('B', 2), ('B', 6), ('B', 12), ('W', 24), ('W', 30))
    mask = 1613783088
    for expert in range(e):
        codes = torch.stack((q[expert] & 15, q[expert] >> 4), -1).reshape(n // 16, 2, 8, k // 32, 2, 2, 4, 2)
        codes = codes.permute(0, 3, 2, 6, 4, 5, 1, 7).reshape(n // 16, k // 32, 32, 8, 2).to(torch.int64)
        words = torch.zeros((n // 16, k // 32, 32, 2), device=q.device, dtype=torch.int64)
        for slot, (src, rot) in enumerate(slots):
            for half in (0, 1):
                for bit, dbit in enumerate((6, 7, 8, 15)):
                    p = (dbit + 16 * half + rot) % 32
                    word = (0 if mask >> p & 1 else 1) if src == 'W' else 0 if src == 'A' else 1
                    words[..., word] |= (codes[..., slot, half] >> bit & 1) << p
        dst[expert] = words.to(torch.int32)
    return dst

def m128_scales_repair_plan(data, bank, which=0, grid=528):
    nt, kt, st, sk, adaptive = M128_SCALES_CONFIGS[which]
    x = data['hidden_states']
    m = x.shape[0]
    dev = x.device
    cap = m * 8 + 32 * nt
    gc = cap // nt
    c = torch.empty(32, device=dev, dtype=torch.int32)
    off = torch.empty(33, device=dev, dtype=torch.int32)
    order = torch.empty_like(c)
    task = torch.empty_like(off)
    slots = torch.empty((32, m), device=dev, dtype=torch.int32)
    ids = data['topk_ids'].to(torch.int32).contiguous()
    tw = data['topk_weights'].float().contiguous()
    pos = torch.empty_like(ids)
    clear = gc * 193 + m
    state = torch.empty(clear + m, device=dev, dtype=torch.int32)
    act = torch.empty((cap, 4096), device=dev, dtype=torch.bfloat16)
    p13 = torch.empty((sk, cap, 8192) if sk > 1 else (1,), device=dev, dtype=torch.float32)
    p2 = torch.empty((sk, cap, 4096) if sk > 1 else (1,), device=dev, dtype=torch.float32)
    y = torch.empty((cap, 4096), device=dev, dtype=torch.float32)
    ss = torch.empty((cap, 64), device=dev, dtype=torch.float32)
    inv = torch.empty(cap, device=dev, dtype=torch.float32)
    out = torch.empty_like(x)
    common = [c, off, order, task, slots, act]
    tail = [ids, tw, pos, bank['pnw'], y, ss, inv, state, out]
    v13 = [bank['w13'], bank['w13_sf'], bank['w13_gs'], x] + common + [p13] + tail
    v2 = [bank['w2'], bank['w2_sf'], bank['w2_gs'], act] + common + [p2] + tail
    start = data['config'].get('local_start', 0)

    def route():
        bank['mod'].ft_route(ids, c, off, order, task, slots, pos, state, out, nt, start, clear)

    def first():
        bank['mod'].ft_run(v13, 1, which, grid, start)

    def second():
        bank['mod'].ft_run(v2, 0, which, grid, start)
        return out
    return {'steps': (route, first, second), 'v13': v13, 'v2': v2}

def m128_scales_encode_scales(sf: torch.Tensor):
    """E4M3 block scales -> (u8 code, neg mask, zero mask). Code: |s| = 2^(e5 - 21) * (1 + m/8),
    byte = e5 << 3 | m; the kernel builds BF16 bits 0x7000 | byte << 4 = |s| * 2^118. Exact for
    every nonzero finite E4M3 value (subnormals included)."""
    s = sf.float()
    if not torch.isfinite(s).all():
        raise ValueError('non-finite NVFP4 block scale')
    a = s.abs()
    nz = a > 0
    m, e = torch.frexp(torch.where(nz, a, torch.ones_like(a)))
    mant = ((m * 2 - 1) * 8).round().to(torch.int32)
    e5 = (e - 1 + 21).to(torch.int32)
    assert int(e5[nz].min()) >= 0 and int(e5[nz].max()) <= 30 and (int(mant.max()) <= 7)
    code = torch.where(nz, e5 << 3 | mant, torch.zeros_like(e5))
    back = torch.exp2((code >> 3).float() - 21) * (1 + (code & 7).float() / 8)
    assert torch.equal(torch.where(nz, back, torch.zeros_like(back)), a)
    return (code.to(torch.uint8), s < 0, ~nz)
M128_SCALES_CONFIGS = ((16, 256, 4, 1, False), (64, 128, 4, 1, False), (64, 128, 4, 1, True), (128, 64, 4, 1, True), (128, 128, 2, 1, True), (128, 64, 4, 1, False))
M128_SCALES_SCALE_CUDA = '#include <cuda_runtime.h>\n#include <vector>\n#include <stdexcept>\n#define WX_CUDA_CHECK(expr) do {cudaError_t wx_status=(expr);if(wx_status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(wx_status));}while(0)\n#define WX_CHECK(cond,...) do {if(!(cond))throw std::runtime_error("packed pipeline argument check");}while(0)\n// Raw E4M3 stays raw in the bank. A format lookup and packed 32-bit fragments\n// replace per-scalar conversion/branches. The lookup is made inside each CTA.\n\n\n#include <cuda_bf16.h>\n#include <stdint.h>\n\nnamespace msrec {\nconstexpr int BK=64; // descriptor atom remains K64; a pipeline stage holds several atoms.\n__device__ __forceinline__ unsigned smaddr(const void*p){return (unsigned)__cvta_generic_to_shared(p);}\n__device__ __forceinline__ void async_copy(void*d,const void*s,int bytes,int valid){\n unsigned dst=smaddr(d);\n if(bytes==16)asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;"::"r"(dst),"l"(s),"r"(valid?16:0):"memory");\n else if(bytes==8)asm volatile("cp.async.ca.shared.global [%0],[%1],8;"::"r"(dst),"l"(s):"memory");\n else asm volatile("cp.async.ca.shared.global [%0],[%1],4;"::"r"(dst),"l"(s):"memory");\n}\n__device__ __forceinline__ unsigned mul2(unsigned a,unsigned b){\n unsigned c;asm("mul.rn.bf16x2 %0,%1,%2;":"=r"(c):"r"(a),"r"(b));return c;\n}\n__device__ __forceinline__ unsigned rr(unsigned a,int n){return __funnelshift_r(a,a,n);}\n__device__ __forceinline__ void dq(unsigned a,unsigned b,unsigned raw,unsigned*out){\n unsigned w=(a&0x60306030u)|(b&~0x60306030u),mask=0x81c081c0u;\n unsigned s0=__byte_perm(raw,0u,0x4040u)*16u+0x70007000u;\n unsigned s1=__byte_perm(raw,0u,0x4141u)*16u+0x70007000u;\n unsigned s2=__byte_perm(raw,0u,0x4242u)*16u+0x70007000u;\n unsigned s3=__byte_perm(raw,0u,0x4343u)*16u+0x70007000u;\n out[0]=mul2(a&mask,s0);out[1]=mul2(rr(a,4)&mask,s1);\n out[2]=mul2(rr(a,10)&mask,s0);out[3]=mul2(rr(b,2)&mask,s1);\n out[4]=mul2(rr(b,6)&mask,s2);out[5]=mul2(rr(b,12)&mask,s3);\n out[6]=mul2(rr(w,24)&mask,s2);out[7]=mul2(rr(w,30)&mask,s3);\n}\n__device__ __forceinline__ void keep(const unsigned*a){\n asm volatile(""::"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),\n  "r"(a[4]),"r"(a[5]),"r"(a[6]),"r"(a[7]),"r"(a[8]),"r"(a[9]),\n  "r"(a[10]),"r"(a[11]),"r"(a[12]),"r"(a[13]),"r"(a[14]),"r"(a[15]));\n}\ntemplate<int NT>__device__ __forceinline__ void mma(float*c,const unsigned*a,const __nv_bfloat16*xs,int b){\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\n uint64_t d=(uint64_t)((smaddr(xs)&0x3ffff)>>4)|((uint64_t)64<<32)|((uint64_t)1<<62);\n d+=b*2;\n if constexpr(NT==16){\n asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "\n "{%0,%1,%2,%3,%4,%5,%6,%7},{%8,%9,%10,%11},%12,p,1,1,0;}"\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7])\n :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(d));\n }else if constexpr(NT==32){\n asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "\n "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15},{%16,%17,%18,%19},%20,p,1,1,0;}"\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7]),\n  "+f"(c[8]),"+f"(c[9]),"+f"(c[10]),"+f"(c[11]),"+f"(c[12]),"+f"(c[13]),"+f"(c[14]),"+f"(c[15])\n :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(d));\n }else if constexpr(NT==64){\n asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "\n "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"\n "{%32,%33,%34,%35},%36,p,1,1,0;}"\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7]),\n  "+f"(c[8]),"+f"(c[9]),"+f"(c[10]),"+f"(c[11]),"+f"(c[12]),"+f"(c[13]),"+f"(c[14]),"+f"(c[15]),\n  "+f"(c[16]),"+f"(c[17]),"+f"(c[18]),"+f"(c[19]),"+f"(c[20]),"+f"(c[21]),"+f"(c[22]),"+f"(c[23]),\n  "+f"(c[24]),"+f"(c[25]),"+f"(c[26]),"+f"(c[27]),"+f"(c[28]),"+f"(c[29]),"+f"(c[30]),"+f"(c[31])\n :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(d));\n }else{\n asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16 "\n "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63},{%64,%65,%66,%67},%68,p,1,1,0;}"\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7]),"+f"(c[8]),"+f"(c[9]),"+f"(c[10]),"+f"(c[11]),"+f"(c[12]),"+f"(c[13]),"+f"(c[14]),"+f"(c[15]),"+f"(c[16]),"+f"(c[17]),"+f"(c[18]),"+f"(c[19]),"+f"(c[20]),"+f"(c[21]),"+f"(c[22]),"+f"(c[23]),"+f"(c[24]),"+f"(c[25]),"+f"(c[26]),"+f"(c[27]),"+f"(c[28]),"+f"(c[29]),"+f"(c[30]),"+f"(c[31]),"+f"(c[32]),"+f"(c[33]),"+f"(c[34]),"+f"(c[35]),"+f"(c[36]),"+f"(c[37]),"+f"(c[38]),"+f"(c[39]),"+f"(c[40]),"+f"(c[41]),"+f"(c[42]),"+f"(c[43]),"+f"(c[44]),"+f"(c[45]),"+f"(c[46]),"+f"(c[47]),"+f"(c[48]),"+f"(c[49]),"+f"(c[50]),"+f"(c[51]),"+f"(c[52]),"+f"(c[53]),"+f"(c[54]),"+f"(c[55]),"+f"(c[56]),"+f"(c[57]),"+f"(c[58]),"+f"(c[59]),"+f"(c[60]),"+f"(c[61]),"+f"(c[62]),"+f"(c[63])\n :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(d));\n }\n#else\n int lane=threadIdx.x&31;\n #pragma unroll\n for(int j=0;j<NT/8;++j){int r=j*8+(lane>>2),k=b*16+(lane&3)*2;\n  unsigned b0=*(const unsigned*)(xs+r*BK+(k^(((r+(smaddr(xs)>>7))&7)*8)));\n  unsigned b1=*(const unsigned*)(xs+r*BK+((k+8)^(((r+(smaddr(xs)>>7))&7)*8)));\n  float*p=c+4*j;\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"\n  :"+f"(p[0]),"+f"(p[1]),"+f"(p[2]),"+f"(p[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));\n }\n#endif\n}\ntemplate<int NT,bool OVERLAP,bool ADAPT>__device__ __forceinline__ void mma_group(float*c,const unsigned*a,const __nv_bfloat16*xs,int active){\n#pragma unroll\nfor(int b=0;b<4;++b){\n if constexpr(ADAPT){\n  if(active<=16)mma<16>(c,a+b*4,xs,b);\n  else if(active<=32)mma<32>(c,a+b*4,xs,b);\n  else if(active<=64)mma<64>(c,a+b*4,xs,b);\n  else mma<128>(c,a+b*4,xs,b);\n }else mma<NT>(c,a+b*4,xs,b);\n}\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\nasm volatile("wgmma.commit_group.sync.aligned;":::"memory");\nif constexpr(OVERLAP)asm volatile("wgmma.wait_group.sync.aligned 1;":::"memory");\nelse asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");\n#endif\n}\ntemplate<int NT,int KT>__device__ __forceinline__ void load_stage(\n unsigned char*q,unsigned char*s,__nv_bfloat16*bs,const int*Q,const unsigned char*S,\n const __nv_bfloat16*X,const int*src,int e,int mt,int N,int tile,int active){\n int t=threadIdx.x,w=t>>5,l=t&31;\n #pragma unroll\n for(int g=0;g<KT/32;++g){size_t z=((size_t)e*(N/16)+mt*4+w)*128+tile*(KT/32)+g;\n  async_copy(q+(g*128+t)*8,Q+z*64+l*2,8,1);\n  if(l<8)async_copy(s+g*128+w*32+l*4,S+z*32+l*4,4,1);\n }\n #pragma unroll\n for(int c=0;c<KT/64;++c){\n  #pragma unroll\n  for(int j=0;j<NT/16;++j)if(j*16<active){int r=t/8+j*16,k=(t&7)*8;\n   async_copy(bs+c*NT*64+r*64+(k^((r&7)*8)),\n              X+(size_t)max(src[j],0)*4096+tile*KT+c*64+k,16,src[j]>=0);\n  }\n }\n asm volatile("cp.async.commit_group;":::"memory");\n}\ntemplate<int S,int NS>__device__ __forceinline__ void cp_wait(int k){\n if(k==NS-1)asm volatile("cp.async.wait_group 0;":::"memory");\n else if constexpr(S>3){\n  if(k==NS-2)asm volatile("cp.async.wait_group 1;":::"memory");\n  else if constexpr(S>4){\n   if(k==NS-3)asm volatile("cp.async.wait_group 2;":::"memory");\n   else if constexpr(S>5){\n    if(k==NS-4)asm volatile("cp.async.wait_group 3;":::"memory");\n    else if constexpr(S>6){\n     if(k==NS-5)asm volatile("cp.async.wait_group 4;":::"memory");\n     else if constexpr(S>7){\n      if(k==NS-6)asm volatile("cp.async.wait_group 5;":::"memory");\n      else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\n     }else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\n    }else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\n   }else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\n  }else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\n }else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\n}\n__device__ __forceinline__ int publish(int*p){\n int old;asm volatile("atom.acq_rel.gpu.global.add.u32 %0,[%1],1;":"=r"(old):"l"(p):"memory");return old;\n}\n__global__ void route_init(const int*ids,int*C,int*OFF,int*ORDER,int*TASK,int*SLOTS,int*POS,\n int*state,__nv_bfloat16*out,int M,int NT,int start,int clear_count){\n int t=threadIdx.x,all=blockIdx.x*blockDim.x+t,step=gridDim.x*blockDim.x;\n for(int i=all;i<clear_count;i+=step)state[i]=0;\n for(int i=all;i<M*4096/8;i+=step)reinterpret_cast<uint4*>(out)[i]=make_uint4(0,0,0,0);\n if(blockIdx.x)return;\n __shared__ int count[32];if(t<32)count[t]=0;__syncthreads();\n for(int i=t;i<M*8;i+=256){int e=ids[i]-start;POS[i]=-1;\n  if(e>=0&&e<32){int rank=atomicAdd(count+e,1);SLOTS[e*M+rank]=i;POS[i]=rank;}\n }\n for(int token=t;token<M;token+=256){int n=0;\n  #pragma unroll\n  for(int j=0;j<8;++j){int e=ids[token*8+j]-start;n+=(e>=0&&e<32);}\n  state[clear_count+token]=n;\n }\n __syncthreads();\n if(t<32){int n=count[t],pad=(n+NT-1)/NT*NT;C[t]=n;\n  int scan=pad;\n  #pragma unroll\n  for(int d=1;d<32;d*=2){int a=__shfl_up_sync(~0u,scan,d);if(t>=d)scan+=a;}\n  OFF[t]=scan-pad;if(t==31)OFF[32]=scan;\n  int key=n*32+31-t;\n  #pragma unroll\n  for(int k=2;k<=32;k*=2){\n   #pragma unroll\n   for(int j=k/2;j;j/=2){int other=__shfl_xor_sync(~0u,key,j);\n    bool ascending=(t&k)==0,low=(t&j)==0;key=(ascending==low)?min(key,other):max(key,other);\n   }\n  }\n  // Descending keys after reversing the ascending bitonic result.\n  key=__shfl_sync(~0u,key,31-t);ORDER[t]=31-key%32;\n  int blocks=(key/32+NT-1)/NT,ends=blocks;\n  #pragma unroll\n  for(int d=1;d<32;d*=2){int a=__shfl_up_sync(~0u,ends,d);if(t>=d)ends+=a;}\n  TASK[t+1]=ends;if(t==0)TASK[0]=0;\n }\n}\ntemplate<int NT,int WIDTH,int KT,int S,int SPLIT>\n__device__ __forceinline__ void core(const int*Q,const unsigned char*SCALE,const __nv_bfloat16*X,\n const int*src,int e,int mt,int N,int split,unsigned char*qs,unsigned char*ss,\n __nv_bfloat16*bs,float*acc){\n constexpr int QB=64*KT/2,SB=64*KT/16,NS=4096/KT/SPLIT;\n constexpr int active=WIDTH;\n int t=threadIdx.x,w=t>>5,l=t&31;\n  unsigned af[2][16]={};\n  #pragma unroll\n  for(int j=0;j<NT/2;++j)asm volatile("":"+f"(acc[j]));\n  #pragma unroll\n  for(int i=0;i<min(S-1,NS);++i)\n   load_stage<NT,KT>(qs+i*QB,ss+i*SB,bs+i*NT*KT,Q,SCALE,X,src,e,mt,N,split*NS+i,active);\n  // K64 has one MMA group per stage. Alternate register-A banks across\n  // stages so wait1 retires a bank before its next conversion overwrites it.\n  // K128/K256 already alternate within each stage; their schedule is unchanged.\n  static_assert(KT>=128 || NS%2==0,"K64 paired stages require an even K count");\n  #pragma unroll 1\n  for(int kbase=0;kbase<NS;kbase+=(KT==64?2:1)){\n   #pragma unroll\n   for(int ksub=0;ksub<(KT==64?2:1);++ksub){\n   int k=kbase+ksub,st=k%S;cp_wait<S,NS>(k);__syncthreads();\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\n   asm volatile("fence.proxy.async.shared::cta;":::"memory");\n#endif\n   #pragma unroll 1\n   for(int pair=0;pair<(KT>=128?KT/128:1);++pair){\n    #pragma unroll\n    for(int sub=0;sub<(KT>=128?2:1);++sub){int chunk=pair*2+sub,reg=KT==64?ksub:sub;\n     #pragma unroll\n     for(int g=0;g<2;++g){int z=chunk*2+g;uint2 q=*(const uint2*)(qs+st*QB+(z*128+t)*8);\n      unsigned sf=*(const unsigned*)(ss+st*SB+z*128+w*32+(l>>2)*4);dq(q.x,q.y,sf,af[reg]+g*8);}\n     keep(af[reg]);\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\n     asm volatile("wgmma.fence.sync.aligned;":::"memory");\n#endif\n     mma_group<WIDTH,true,false>(acc,af[reg],bs+st*NT*KT+chunk*NT*64,active);keep(af[reg^1]);\n     if(chunk==0&&k+S-1<NS){int dst=(k+S-1)%S;\n      load_stage<NT,KT>(qs+dst*QB,ss+dst*SB,bs+dst*NT*KT,Q,SCALE,X,src,e,mt,N,split*NS+k+S-1,active);}\n    }\n   }\n   }\n  }\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\n  asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");\n#endif\n  keep(af[0]);keep(af[1]);\n}\ntemplate<int NT,int KT,int S,int SPLIT,bool ADAPT,bool FIRST>\n__global__ void __launch_bounds__(128,2) fused(const int*Q,const unsigned char*SCALE,const float*GS,\n const __nv_bfloat16*X,const int*C,const int*OFF,const int*ORDER,const int*TASK,const int*SLOTS,\n const int*IDS,const float*TW,const int*POS,const __nv_bfloat16*PNW,float*PART,__nv_bfloat16*ACT,\n float*Y,float*SS,float*INV,int*STATE,__nv_bfloat16*OUT,int M,int cap,int start){\n constexpr int QB=64*KT/2,SB=64*KT/16,NS=4096/KT/SPLIT,N=FIRST?8192:4096,MT=N/64;\n constexpr int BS=((S*(QB+SB)+1023)&~1023);\n extern __shared__ __align__(128) unsigned char rawsm[];\n unsigned char*sm=(unsigned char*)(((uintptr_t)rawsm+1023)&~(uintptr_t)1023);\n unsigned char*qs=sm,*ss=qs+S*QB;auto*bs=(__nv_bfloat16*)(sm+BS);\n __shared__ int last,flags[NT];\n __shared__ __nv_bfloat16 epi[FIRST?64*NT:1];__shared__ float rowsum[FIRST?1:4*NT];\n int t=threadIdx.x,w=t>>5,l=t&31,gcap=cap/NT;\n int*c1=STATE,*c2=c1+gcap*128,*group_ready=c2+gcap*64,*token_ready=group_ready+gcap,*local_count=token_ready+M;\n \n int total=TASK[32]*MT*SPLIT;\n for(int unit=blockIdx.x;unit<total;unit+=gridDim.x){\n  int task=unit/SPLIT,split=unit%SPLIT,group=task%TASK[32],mt=task/TASK[32],ix=0;\n  #pragma unroll\n  for(int d=16;d;d>>=1)if(TASK[ix+d]<=group)ix+=d;\n  task=group*MT+mt;\n  int e=ORDER[ix],row=(group-TASK[ix])*NT,count=C[e],offset=OFF[e];\n  int remain=min(NT,count-row),active=ADAPT?(remain<=16?16:remain<=32?32:remain<=64?64:128):NT;\n  int src[NT/16];\n  #pragma unroll\n  for(int j=0;j<NT/16;++j){int r=row+t/8+j*16;src[j]=r<count?(FIRST?SLOTS[e*M+r]/8:offset+r):-1;}\n  float acc[NT/2]={};\n  if constexpr(ADAPT){\n   if(active<=16)core<NT,16,KT,S,SPLIT>(Q,SCALE,X,src,e,mt,N,split,qs,ss,bs,acc);\n   else if(active<=32)core<NT,32,KT,S,SPLIT>(Q,SCALE,X,src,e,mt,N,split,qs,ss,bs,acc);\n   else if constexpr(NT>=64){\n    if(active<=64)core<NT,64,KT,S,SPLIT>(Q,SCALE,X,src,e,mt,N,split,qs,ss,bs,acc);\n    else if constexpr(NT>=128)core<NT,128,KT,S,SPLIT>(Q,SCALE,X,src,e,mt,N,split,qs,ss,bs,acc);\n   }\n  }else core<NT,NT,KT,S,SPLIT>(Q,SCALE,X,src,e,mt,N,split,qs,ss,bs,acc);\n  if constexpr(SPLIT>1){\n   #pragma unroll\n   for(int j=0;j<NT/2;++j){int r=row+(l&3)*2+(j&1)+(j>>2)*8;\n    int c=mt*64+w*16+(l>>2)+((j>>1)&1)*8;\n    if(r<count)PART[((size_t)split*cap+offset+r)*N+c]=acc[j];\n   }\n   __threadfence();__syncthreads();\n   if(t==0)last=(publish((FIRST?c1:c2)+task)==SPLIT-1);__syncthreads();\n   if(!last)continue;\n   #pragma unroll\n   for(int j=0;j<NT/2;++j){int r=row+(l&3)*2+(j&1)+(j>>2)*8;\n    int c=mt*64+w*16+(l>>2)+((j>>1)&1)*8;float sum=0;\n    if(r<count){\n     #pragma unroll\n     for(int sk=0;sk<SPLIT;++sk)sum+=PART[((size_t)sk*cap+offset+r)*N+c];\n    }\n    acc[j]=sum;\n   }\n  }\n  if constexpr(FIRST){\n   #pragma unroll\n   for(int j=0;j<NT/2;++j){int r=(l&3)*2+(j&1)+(j>>2)*8;\n    int c=w*16+(l>>2)+((j>>1)&1)*8;float gs=GS[e+(c>=32?32:0)]*256.f;\n    epi[c*NT+r]=__float2bfloat16_rn(acc[j]*gs);\n   }\n   __syncthreads();\n   #pragma unroll\n   for(int i=t;i<NT*32;i+=128){int r=i/32,c=i%32;\n    float g=__bfloat162float(epi[c*NT+r]),u=__bfloat162float(epi[(c+32)*NT+r]);\n    if(row+r<count)ACT[(size_t)(offset+row+r)*4096+mt*32+c]=__float2bfloat16_rn(g/(1.f+__expf(-g))*u);\n   }\n  }else{\n   float gs=GS[e]*256.f;\n   #pragma unroll\n   for(int j=0;j<NT/2;++j){int r=row+(l&3)*2+(j&1)+(j>>2)*8;\n    int c=mt*64+w*16+(l>>2)+((j>>1)&1)*8;\n    float z=acc[j]*gs*(1.f+__bfloat162float(PNW[c]));acc[j]=z;\n    if(r<count)Y[(size_t)(offset+r)*4096+c]=z;\n   }\n   #pragma unroll\n   for(int i=0;i<NT/8;++i){float a=acc[4*i]*acc[4*i]+acc[4*i+2]*acc[4*i+2];\n    float b=acc[4*i+1]*acc[4*i+1]+acc[4*i+3]*acc[4*i+3];\n    #pragma unroll\n    for(int d=4;d<=16;d*=2){a+=__shfl_xor_sync(~0u,a,d);b+=__shfl_xor_sync(~0u,b,d);}\n    if(l<4){rowsum[w*NT+i*8+2*l]=a;rowsum[w*NT+i*8+2*l+1]=b;}\n   }\n   __syncthreads();\n   if(t<NT&&row+t<count){float sum=0;\n    #pragma unroll\n    for(int ww=0;ww<4;++ww)sum+=rowsum[ww*NT+t];\n    SS[(size_t)(offset+row+t)*64+mt]=sum;\n   }\n   __threadfence();__syncthreads();\n   if(t==0)last=(publish(group_ready+group)==63);__syncthreads();\n   if(last){\n    for(int r=w;r<NT;r+=4)if(row+r<count){float sum=SS[(size_t)(offset+row+r)*64+l]+SS[(size_t)(offset+row+r)*64+l+32];\n     #pragma unroll\n     for(int d=16;d;d/=2)sum+=__shfl_down_sync(~0u,sum,d);\n     if(l==0)INV[offset+row+r]=rsqrtf(sum/4096.f+1.e-8f);\n    }\n    __threadfence();__syncthreads();\n    if(t<NT){flags[t]=0;if(row+t<count){int tok=SLOTS[e*M+row+t]/8;\n      flags[t]=(publish(token_ready+tok)==local_count[tok]-1);}}\n    __syncthreads();\n    for(int r=w;r<NT;r+=4)if(flags[r]){int tok=SLOTS[e*M+row+r]/8;\n     for(int c=l;c<4096;c+=32){float sum=0;\n      #pragma unroll\n      for(int j=0;j<8;++j){int ee=IDS[tok*8+j]-start;\n       if(ee>=0&&ee<32){int p=OFF[ee]+POS[tok*8+j];sum+=Y[(size_t)p*4096+c]*INV[p]*TW[tok*8+j];}}\n      OUT[(size_t)tok*4096+c]=__float2bfloat16_rn(sum);\n     }\n    }\n   }\n  }\n  __syncthreads();\n }\n}\n#define FT_CASES(F) \\\n F(3,128,64,4,1,true) \\\n F(4,128,128,2,1,true) \\\n F(5,128,64,4,1,false)\n}\nint64_t msrec_init(){int dev,opt;WX_CUDA_CHECK(cudaGetDevice(&dev));WX_CUDA_CHECK(cudaDeviceGetAttribute(&opt,cudaDevAttrMaxSharedMemoryPerBlockOptin,dev));\n #define INIT(ID,NT,KT,S,SK,AD) {constexpr int bytes=1023+((S*(64*KT/2+64*KT/16)+1023)&~1023)+S*NT*KT*2;WX_CUDA_CHECK(cudaFuncSetAttribute(msrec::fused<NT,KT,S,SK,AD,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));WX_CUDA_CHECK(cudaFuncSetAttribute(msrec::fused<NT,KT,S,SK,AD,false>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes));}\n FT_CASES(INIT)\n #undef INIT\n return opt;\n}\nstd::vector<int64_t> msrec_info(int64_t which,int64_t first){\n #define INFO(ID,NT,KT,S,SK,AD) case ID:{constexpr int bytes=1023+((S*(64*KT/2+64*KT/16)+1023)&~1023)+S*NT*KT*2;auto info=[&]<bool F>(){cudaFuncAttributes a{};int blocks;WX_CUDA_CHECK(cudaFuncGetAttributes(&a,msrec::fused<NT,KT,S,SK,AD,F>));WX_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,msrec::fused<NT,KT,S,SK,AD,F>,128,bytes));return std::vector<int64_t>{a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,bytes,blocks};};return first?info.template operator()<true>():info.template operator()<false>();}\n switch(which){FT_CASES(INFO) default:WX_CHECK(false,"unknown fused configuration");}\n #undef INFO\n}\nvoid msrec_route(std::vector<int64_t>v,int64_t M,int64_t nt,int64_t start,int64_t clear_count,int64_t stream){\n msrec::route_init<<<128,256,0,(cudaStream_t)stream>>>((const int*)v[0],(int*)v[1],(int*)v[2],(int*)v[3],(int*)v[4],(int*)v[5],(int*)v[6],(int*)v[7],(__nv_bfloat16*)v[8],M,nt,start,clear_count);WX_CUDA_CHECK(cudaGetLastError());\n}\nvoid msrec_run(std::vector<int64_t> v,int64_t first,int64_t which,int64_t grid,int64_t start,int64_t M,int64_t cap,int64_t stream_value){\n WX_CHECK(v.size()==20);auto stream=(cudaStream_t)stream_value;\n #define GO(ID,NT,KT,S,SK,AD) case ID:{constexpr int bytes=1023+((S*(64*KT/2+64*KT/16)+1023)&~1023)+S*NT*KT*2; \\\n auto launch=[&]<bool FIRST>(){msrec::fused<NT,KT,S,SK,AD,FIRST><<<grid,128,bytes,stream>>>((int*)v[0],(uint8_t*)v[1],(float*)v[2],(const __nv_bfloat16*)(void*)v[3],(int*)v[4],(int*)v[5],(int*)v[6],(int*)v[7],(int*)v[8],(int*)v[11],(float*)v[12],(int*)v[13],(const __nv_bfloat16*)(void*)v[14],(float*)v[10],(__nv_bfloat16*)(void*)v[9],(float*)v[15],(float*)v[16],(float*)v[17],(int*)v[18],(__nv_bfloat16*)(void*)v[19],M,cap,start);};if(first)launch.template operator()<true>();else launch.template operator()<false>();break;}\n switch(which){FT_CASES(GO) default:WX_CHECK(false,"unknown fused configuration");}\n #undef GO\n WX_CUDA_CHECK(cudaGetLastError());\n}\n'
M128_SCALES_SCALE_CPP = '#include <pybind11/pybind11.h>\n#include <pybind11/stl.h>\n#include <vector>\n#include <cstdint>\nint64_t msrec_init();std::vector<int64_t> msrec_info(int64_t,int64_t);void msrec_route(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t);void msrec_run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);'
M128_SCALES_SCALE_FUNCTIONS = ['msrec_init', 'msrec_info', 'msrec_route', 'msrec_run']

@functools.lru_cache(None)
def m128_scales_scale_module():
    arch = '90a' if torch.cuda.get_device_capability() == (9, 0) else '120'
    mod = load_inline(name='m128_scales_' + hashlib.sha256(M128_SCALES_SCALE_CUDA.encode()).hexdigest()[:12], cpp_sources=M128_SCALES_SCALE_CPP, cuda_sources=M128_SCALES_SCALE_CUDA, functions=M128_SCALES_SCALE_FUNCTIONS, no_implicit_headers=True, with_pytorch_error_handling=False, extra_cuda_cflags=['-O3', '-std=c++20', '-lineinfo', '-gencode=arch=compute_' + arch + ',code=sm_' + arch], verbose=False)
    mod.msrec_init()
    return mod

class m128_scales_ScaleABI:

    def __init__(self, name):
        self.mod = m128_scales_scale_module()
        self.name = name

    def ft_route(self, ids, c, off, order, task, slots, pos, state, out, nt, start, clear):
        return getattr(self.mod, self.name + '_route')([v.data_ptr() for v in (ids, c, off, order, task, slots, pos, state, out)], ids.shape[0], nt, start, clear, torch.cuda.current_stream().cuda_stream)

    def ft_run(self, v, first, which, grid, start):
        return getattr(self.mod, self.name + '_run')([t.data_ptr() for t in v], first, which, grid, start, v[11].shape[0], v[9].shape[0], torch.cuda.current_stream().cuda_stream)

def m128_scales_scale_prepare(w):
    bank = {}
    for key in ('w13', 'w2'):
        q, sf = (w[key], w[key + '_sf'])
        e, n, k16 = sf.shape
        planes = []
        scales = []
        for expert in range(e):
            code, neg, zero = m128_scales_encode_scales(sf[expert])
            folded = q[expert].reshape(n, k16, 8) ^ neg.to(torch.uint8)[..., None] * 136
            folded = torch.where(zero[..., None], torch.zeros_like(folded), folded)
            planes.append(m128_scales_mw_repack_planes(folded.reshape(1, n, -1)))
            scales.append(code)
        bank[key] = torch.cat(planes)
        sfbytes = torch.stack(scales)
        bank[key + '_sf'] = sfbytes.reshape(e, n // 16, 2, 8, k16 // 2, 2).permute(0, 1, 4, 3, 5, 2).contiguous()
    bank['w13_gs'] = torch.cat((w.get('w13_gs_gate', w['w13_gs']), w.get('w13_gs_up', w['w13_gs']))).float().contiguous()
    bank['w2_gs'] = w['w2_gs'].float().contiguous()
    bank['pnw'] = w['post_norm_w'].contiguous()
    ng = bank['w13'].shape[1]
    rr = torch.arange(ng // 2, device=w['w13'].device).reshape(-1, 2)
    order = torch.cat((rr, rr + ng // 2), 1).flatten()
    bank['w13'] = bank['w13'].index_select(1, order).contiguous()
    bank['w13_sf'] = bank['w13_sf'].index_select(1, order).contiguous()
    bank['mod'] = m128_scales_ScaleABI('msrec')
    return bank

def m128_scales_scale_call(data, bank, which=3, grid=528):
    for f in m128_scales_repair_plan(data, bank, which, grid)['steps']:
        out = f()
    return out
