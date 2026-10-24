RG_CUDA='\n#include <cuda_runtime.h>\n\n#include <cuda_bf16.h>\n#include <stdint.h>\n#include <math_constants.h>\n\nnamespace {\n\nconstexpr int HD = 4096;\nconstexpr int NLOC = 32;\nconstexpr int TILE_BYTES = 36864;\nconstexpr int TILE_W_BYTES = 32768;\nconstexpr int NT1 = 512;\nconstexpr int NT2 = 256;\nconstexpr int NCW = 8;                 // consumer warps\nconstexpr int NCT = NCW * 32;          // consumer threads\nconstexpr int NTHR = NCT;             // thread 0 also prefetches\nconstexpr uint32_t DQ_MASK = 0x81C081C0u;\nconstexpr uint32_t LA_MASK = 0x60306030u;\n// persistent counters\nenum { C_DONE = 0, C_ZERO = 32, C_BAR = 33, C_GEN = 34 };\n\n__device__ __forceinline__ uint32_t smem_u32(const void* p) {\n  return static_cast<uint32_t>(__cvta_generic_to_shared(p));\n}\n__device__ __forceinline__ void mbar_init(uint64_t* b, int cnt) {\n  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(b)), "r"(cnt));\n}\n__device__ __forceinline__ bool mbar_try(uint64_t* b, uint32_t parity) {\n  uint32_t ok;\n  asm volatile("{\\n .reg .pred p;\\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\\n selp.u32 %0, 1, 0, p;\\n}\\n"\n               : "=r"(ok) : "r"(smem_u32(b)), "r"(parity) : "memory");\n  return ok != 0;\n}\n__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t parity) {\n  while (!mbar_try(b, parity)) {}\n}\n__device__ __forceinline__ void mbar_arrive(uint64_t* b) {\n  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(b)) : "memory");\n}\n__device__ __forceinline__ void mbar_expect_tx(uint64_t* b, uint32_t bytes) {\n  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(b)), "r"(bytes) : "memory");\n}\n__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* b, uint64_t pol) {\n  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"\n               :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(b)), "l"(pol) : "memory");\n}\n__device__ __forceinline__ void bar_consumers() {\n  asm volatile("bar.sync 1, %0;" :: "n"(NCT) : "memory");\n}\n__device__ __forceinline__ int ld_acquire(const int* p) {\n  int v;\n  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");\n  return v;\n}\n__device__ __forceinline__ uint32_t rotr(uint32_t x, int s) { return __funnelshift_r(x, x, s); }\n__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {\n  uint32_t d;\n  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));\n  return d;\n}\n// E4M3 -> two equal BF16 lanes scaled by 2^118. dq8 supplies 2^-126;\n// the existing GEMM epilogue supplies 2^8. Preserve zeros, signs and subnormals.\n__device__ __forceinline__ uint32_t scale_e4m3(uint32_t code){\n const uint32_t b=0x7000u+(code<<4);return b|(b<<16);}\n\n__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,\n                                         uint32_t b0, uint32_t b1) {\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"\n               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])\n               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));\n}\n// 16 E2M1 codes (two words) -> 8 BF16x2 registers holding value * 2^-126 (exact).\n__device__ __forceinline__ void dq8(uint32_t A, uint32_t B, uint32_t* o) {\n  o[0] = A & DQ_MASK;\n  o[1] = rotr(A, 4) & DQ_MASK;\n  o[2] = rotr(A, 10) & DQ_MASK;\n  o[3] = rotr(B, 2) & DQ_MASK;\n  o[4] = rotr(B, 6) & DQ_MASK;\n  o[5] = rotr(B, 12) & DQ_MASK;\n  const uint32_t W = (A & LA_MASK) | (B & ~LA_MASK);\n  o[6] = rotr(W, 24) & DQ_MASK;\n  o[7] = rotr(W, 30) & DQ_MASK;\n}\n\nstruct PrefetchState {\n  volatile int index, stage, phase, pass;\n  volatile int e, nb, m, mb, me;\n  volatile uint64_t pol_first, pol_normal;\n};\n\nstruct Route {        // shared-memory routing summary\n  int n[NLOC];        // pairs per local expert\n  int off[NLOC];      // first sorted position\n  int nblk[NLOC];     // token blocks\n  int us1[NLOC + 1];  // phase-1 unit prefix\n  int us2[NLOC + 1];  // phase-2 unit prefix\n  int c1, c2;         // m-chunk sizes\n  int run[NLOC];\n  int wc[NCW][NLOC];\n  PrefetchState pre;\n};\n\n// unit -> (expert, token block, m tile); order (e, m-chunk, block, m)\n__device__ __forceinline__ void decode_unit(int u, const int* us, const int* nblk, int cc, int ntile,\n                                            int& e, int& nb, int& m) {\n  e = 0;\n#pragma unroll\n  for (int d = 16; d > 0; d >>= 1) {\n    if (us[e + d] <= u) e += d;\n  }\n  const int rel = u - us[e];\n  const int nbk = nblk[e];\n  const int nfull = ntile / cc;\n  const int span = cc * nbk;\n  if (rel < nfull * span) {\n    const int mc = rel / span;\n    const int r2 = rel - mc * span;\n    nb = r2 / cc;\n    m = mc * cc + (r2 - nb * cc);\n  } else {\n    const int L = ntile - nfull * cc;\n    const int r3 = rel - nfull * span;\n    nb = r3 / L;\n    m = nfull * cc + (r3 - nb * L);\n  }\n}\n\n__device__ __forceinline__ void advance_unit(int& e, int& nb, int& m, int& mb, int& me,\n                                               const int* nblk, int cc, int ntile) {\n  if (++m == me) {\n    if (++nb == nblk[e]) {\n      nb = 0;\n      mb += cc;\n      if (mb >= ntile) {\n        ++e;\n        while (e < NLOC && nblk[e] == 0) ++e;\n        mb = 0;\n      }\n      me = min(ntile, mb + cc);\n    }\n    m = mb;\n  }\n}\n\n\n__device__ __forceinline__ void prefetch_one(\n    Route* rt, int total, int first, int a2, int nstage,\n    uint8_t* stages, uint64_t* fullb, uint64_t* emptyb,\n    const uint8_t* w13p, const uint8_t* w2p) {\n  auto& p = rt->pre;\n  int idx = p.index;\n  if (idx >= total) return;\n  int st=p.stage, ph=p.phase, pass=p.pass;\n  int e=p.e, nb=p.nb, m=p.m, mb=p.mb, me=p.me;\n  const int ntile=pass?NT2:NT1;\n  const int cc=pass?rt->c2:rt->c1;\n  const uint8_t* base=pass?w2p:w13p;\n  const uint8_t* src=base+(static_cast<size_t>(e)*ntile+m)*TILE_BYTES;\n  const uint64_t pol=rt->nblk[e]>1?p.pol_normal:p.pol_first;\n  mbar_wait(&emptyb[st], ph^1u);\n  mbar_expect_tx(&fullb[st], TILE_BYTES);\n  bulk_g2s(stages+st*TILE_BYTES, src, TILE_BYTES, &fullb[st], pol);\n  ++idx;\n  if (idx < total) {\n    if (idx == first) {\n      pass=1;\n      decode_unit(a2,rt->us2,rt->nblk,rt->c2,NT2,e,nb,m);\n      mb=(m/rt->c2)*rt->c2;\n      me=min(NT2,mb+rt->c2);\n    } else {\n      advance_unit(e,nb,m,mb,me,rt->nblk,cc,ntile);\n    }\n  }\n  if (++st == nstage) { st=0; ph^=1; }\n  p.index=idx; p.stage=st; p.phase=ph; p.pass=pass;\n  p.e=e; p.nb=nb; p.m=m; p.mb=mb; p.me=me;\n}\n\n\n#include <stdint.h>\n\n// Per-call, input-derived metadata. Every entry is int32, no device pointers.\n// 8 sorted rows per NT1 token block; tiles are W13=512 and W2=256.\nstruct alignas(128) MoERouteMeta {\n  int32_t version, tokens, local_start, pairs;\n  int32_t count[32];\n  int32_t offset[33];\n  int32_t blocks8[32];\n  int32_t units13[33], units2[33];\n  int32_t token_local[32];\n  int32_t pair_of[256]; // sorted local position -> token*8+slot, -1 unused\n  int32_t pos_of[256];  // token*8+slot -> sorted local position, -1 remote\n};\nstatic_assert(sizeof(MoERouteMeta)==2944);\n\nstruct RouteGlue {\n  const float* pr;const float* bias;const float* pre;int splits;\n  int64_t ps,rs;\n  int* out_ids;int* out_lids;float* out_wts;int* local_start;MoERouteMeta* meta;\n};\n\n__device__ __forceinline__ float rg_wsum(float x){\n  #pragma unroll\n  for(int d=16;d;d>>=1)x+=__shfl_xor_sync(~0u,x,d);return x;\n}\n// Each CTA constructs its own summary; no inter-CTA publication or waiting.\n// Only CTA zero exports layer IDs. The MoE body consumes the private ids/weights.\n__device__ __noinline__ void rg_prepare(RouteGlue rg,int M,int* ids,float* weights,float* inv){\n  int lane=threadIdx.x&31,warp=threadIdx.x>>5;\n  for(int tok=warp;tok<M;tok+=8){\n    float score[8],biased[8];\n    #pragma unroll\n    for(int j=0;j<8;++j){\n      int e=j*32+lane;float logit=0;\n      for(int s=0;s<rg.splits;++s){\n        const float* p=rg.pr+s*rg.ps+(int64_t)tok*rg.rs;\n        logit=__fadd_rn(logit,__fadd_rn(p[4096+e],p[4352+e]));\n      }\n      score[j]=1.f/(1.f+__expf(-logit*0.1767766952966369f));\n      biased[j]=score[j]+rg.bias[e];\n    }\n    int pick[8];float prob[8],total=0;\n    #pragma unroll\n    for(int k=0;k<8;++k){\n      float best=-CUDART_INF_F;int id=256;\n      #pragma unroll\n      for(int j=0;j<8;++j){int e=j*32+lane;if(biased[j]>best||(biased[j]==best&&e<id)){best=biased[j];id=e;}}\n      #pragma unroll\n      for(int d=16;d;d>>=1){float b=__shfl_xor_sync(~0u,best,d);int i=__shfl_xor_sync(~0u,id,d);\n        if(b>best||(b==best&&i<id)){best=b;id=i;}}\n      float p=0;\n      #pragma unroll\n      for(int j=0;j<8;++j)if(j*32+lane==id){p=score[j];biased[j]=-CUDART_INF_F;}\n      p=rg_wsum(p);pick[k]=id;prob[k]=p;total+=p;\n    }\n    if(lane<8){int local=(pick[lane]-*rg.local_start+256)%256;\n      ids[tok*8+lane]=local<32?local:-1;\n      weights[tok*8+lane]=prob[lane]/(total+1e-15f);\n      if(blockIdx.x==0){rg.out_ids[tok*8+lane]=pick[lane];rg.out_lids[tok*8+lane]=ids[tok*8+lane];rg.out_wts[tok*8+lane]=weights[tok*8+lane];}}\n  }\n  __syncthreads();\n  if(M==1){\n    __shared__ float sums[8];float ss=0;\n    for(int d=threadIdx.x;d<4096;d+=256){float v=0;\n      for(int s=0;s<rg.splits;++s)v+=rg.pr[(int64_t)s*rg.ps+d];\n      ss=fmaf(v,v,ss);\n    }\n    ss=rg_wsum(ss);if(lane==0)sums[warp]=ss;\n    __syncthreads();ss=lane<8?sums[lane]:0.f;ss=rg_wsum(ss);\n    if(threadIdx.x==0)inv[0]=rsqrtf(ss*(1.f/4096)+1e-5f);\n  }else{\n    for(int tok=warp;tok<M;tok+=8){float ss=0;\n      for(int d=lane;d<4096;d+=32){float v=0;\n        for(int s=0;s<rg.splits;++s)v+=rg.pr[(int64_t)s*rg.ps+(int64_t)tok*rg.rs+d];\n        ss=fmaf(v,v,ss);\n      }\n      ss=rg_wsum(ss);if(lane==0)inv[tok]=rsqrtf(ss*(1.f/4096)+1e-5f);\n    }\n  }\n  __syncthreads();\n}\n\n__device__ __forceinline__ __nv_bfloat16 rg_value(RouteGlue rg,int token,int col,float inv){\n  float value=0;\n  for(int s=0;s<rg.splits;++s)value+=rg.pr[(int64_t)s*rg.ps+(int64_t)token*rg.rs+col];\n  return __float2bfloat16_rn(value*inv*rg.pre[col]);\n}\n\n// MoERouteMeta v1 from nt1_branches/route_meta.h. Inputs are the CTA-private\n// local route IDs; shared scratch reuses the MoE body\'s Route storage before\n// that storage is initialized by the expert scheduler.\n__device__ __forceinline__ void rg_emit_meta(MoERouteMeta* dst,int M,int start,const int* ids,int* scratch){\n  int* n=scratch;int* wc=n+32;int* off=wc+256;\n  int t=threadIdx.x,lane=t&31,warp=t>>5;\n  for(int j=t;j<(int)sizeof(MoERouteMeta)/4;j+=256)((int*)dst)[j]=0;\n  __syncthreads();\n  if(t<32)n[t]=0;\n  dst->pair_of[t]=-1;dst->pos_of[t]=-1;\n  __syncthreads();\n  int e=t<M*8?ids[t]:-1;\n  if(e<0||e>=32)e=-1;\n  if(e>=0)atomicAdd(n+e,1);\n  __syncthreads();\n  if(warp==0){\n    int count=n[lane],nb=(count+7)/8,p=count,u1=nb*512,u2=nb*256;\n    #pragma unroll\n    for(int d=1;d<32;d<<=1){\n      int a=__shfl_up_sync(~0u,p,d),b=__shfl_up_sync(~0u,u1,d),c=__shfl_up_sync(~0u,u2,d);\n      if(lane>=d){p+=a;u1+=b;u2+=c;}\n    }\n    off[lane]=p-count;\n    dst->count[lane]=count;dst->offset[lane]=p-count;dst->blocks8[lane]=nb;\n    dst->units13[lane+1]=u1;dst->units2[lane+1]=u2;\n    if(lane==31){dst->pairs=p;dst->offset[32]=p;}\n    if(lane==0){dst->version=1;dst->tokens=M;dst->local_start=start;}\n  }\n  wc[warp*32+lane]=0;__syncthreads();\n  unsigned mask=__match_any_sync(~0u,e);int rank=__popc(mask&((1u<<lane)-1));\n  if(e>=0&&rank==0)wc[warp*32+e]=__popc(mask);\n  __syncthreads();\n  if(t<32){int sum=off[t];for(int w=0;w<8;++w){int v=wc[w*32+t];wc[w*32+t]=sum;sum+=v;}}\n  __syncthreads();\n  if(e>=0){int pos=wc[warp*32+e]+rank;dst->pair_of[pos]=t;dst->pos_of[t]=pos;}\n  if(t<M){int count=0;\n    #pragma unroll\n    for(int s=0;s<8;++s)count+=(ids[t*8+s]>=0);\n    dst->token_local[t]=count;\n  }\n  __syncthreads();\n}\n\ntemplate <int NT>\n__global__ void __launch_bounds__(NTHR, 2) moe_fused(\n    const __nv_bfloat16* __restrict__ x, const int* __restrict__ ids, const float* __restrict__ tw,\n    const __nv_bfloat16* __restrict__ pnw, const uint8_t* __restrict__ w13p, const uint8_t* __restrict__ w2p,\n    const float* __restrict__ gs13, const float* __restrict__ gs2, __nv_bfloat16* __restrict__ inter,\n    float* __restrict__ y, float* __restrict__ ss, int* __restrict__ cnt, __nv_bfloat16* __restrict__ out,\n    int M, int local_start, int nstage, RouteGlue rg) {\n  constexpr int NB = 8 * NT;\n  constexpr int RED = NT * 4;\n  const int npairs = M * 8;\n  extern __shared__ __align__(128) uint8_t smem[];\n  uint8_t* stages = smem;\n  float* red = reinterpret_cast<float*>(smem + nstage * TILE_BYTES);        // [2][NCW][32][RED]\n  uint64_t* fullb = reinterpret_cast<uint64_t*>(red + 2 * NCW * 32 * RED);\n  uint64_t* emptyb = fullb + nstage;\n  Route* rt = reinterpret_cast<Route*>(emptyb + nstage);\n  int8_t* eflat = reinterpret_cast<int8_t*>(rt + 1);                        // [npairs] local expert or -1\n  int16_t* pos_of = reinterpret_cast<int16_t*>(eflat + ((npairs + 15) & ~15));  // [npairs]\n  int16_t* pair_of = pos_of + npairs;                                        // [npairs] sorted pos -> flat\n\n  int* private_ids=reinterpret_cast<int*>(pair_of+npairs);\n  float* private_tw=reinterpret_cast<float*>(private_ids+npairs);\n  float* private_inv=private_tw+npairs;\n  rg_prepare(rg,M,private_ids,private_tw,private_inv);\n  if(blockIdx.x==0 && M<=32)rg_emit_meta(rg.meta,M,*rg.local_start,private_ids,reinterpret_cast<int*>(rt));\n  __syncthreads();\n  ids=private_ids;tw=private_tw;\n  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;\n  const int G = gridDim.x, c = blockIdx.x;\n  int bar_gen0 = 0;\n\n  // ---------------- routing: counts ----------------\n  if (tid < NLOC) rt->n[tid] = 0;\n  if (tid == 0) {\n    for (int s = 0; s < nstage; ++s) { mbar_init(&fullb[s], 1); mbar_init(&emptyb[s], NCW); }\n    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");\n    bar_gen0 = ld_acquire(cnt + C_GEN);\n  }\n  __syncthreads();\n  for (int i = tid; i < npairs; i += NTHR) {\n    int e = ids[i] - local_start;\n    if (e < 0 || e >= NLOC) e = -1;\n    eflat[i] = static_cast<int8_t>(e);\n    if (e >= 0) atomicAdd(&rt->n[e], 1);\n  }\n  __syncthreads();\n  if (warp == 0) {\n    const int n = rt->n[lane];\n    const int nb = (n + NB - 1) / NB;\n    int off = n, u1 = nb * NT1, u2 = nb * NT2;\n#pragma unroll\n    for (int d = 1; d < 32; d <<= 1) {\n      const int a = __shfl_up_sync(~0u, off, d), b = __shfl_up_sync(~0u, u1, d), e2 = __shfl_up_sync(~0u, u2, d);\n      if (lane >= d) { off += a; u1 += b; u2 += e2; }\n    }\n    rt->off[lane] = off - n;\n    rt->run[lane] = off - n;\n    rt->nblk[lane] = nb;\n    rt->us1[lane + 1] = u1;\n    rt->us2[lane + 1] = u2;\n    if (lane == 0) { rt->us1[0] = 0; rt->us2[0] = 0; }\n    const int U1 = __shfl_sync(~0u, u1, 31), U2 = __shfl_sync(~0u, u2, 31);\n    if (lane == 0) {\n      rt->c1 = min(NT1, max(1, (U1 + G / 2) / G));\n      rt->c2 = min(NT2, max(1, (U2 + G / 2) / G));\n    }\n  }\n  __syncthreads();\n\n  const int U1 = rt->us1[NLOC], U2 = rt->us2[NLOC];\n  const int a1 = static_cast<int>((long long)U1 * c / G), b1 = static_cast<int>((long long)U1 * (c + 1) / G);\n  const int a2 = static_cast<int>((long long)U2 * c / G), b2 = static_cast<int>((long long)U2 * (c + 1) / G);\n\n  const int first_count=b1-a1, total_count=first_count+b2-a2;\n  if (tid == 0) {\n    uint64_t pf,pn;\n    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pf));\n    asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(pn));\n    rt->pre.pol_first=pf; rt->pre.pol_normal=pn;\n    rt->pre.index=0; rt->pre.stage=0; rt->pre.phase=0;\n    if (total_count) {\n      int e,nb,m;\n      const int pass=first_count?0:1;\n      const int cc=pass?rt->c2:rt->c1, ntile=pass?NT2:NT1;\n      decode_unit(pass?a2:a1,pass?rt->us2:rt->us1,rt->nblk,cc,ntile,e,nb,m);\n      rt->pre.pass=pass; rt->pre.e=e; rt->pre.nb=nb; rt->pre.m=m;\n      rt->pre.mb=(m/cc)*cc; rt->pre.me=min(ntile,(m/cc)*cc+cc);\n    }\n    for (int i=0;i<nstage;++i)\n      prefetch_one(rt,total_count,first_count,a2,nstage,stages,fullb,emptyb,w13p,w2p);\n  }\n  __syncthreads();\n\n  // ---------------- consumers: deterministic token lists ----------------\n  for (int base = 0; base < npairs; base += NCT) {\n    rt->wc[warp][lane] = 0;\n    bar_consumers();\n    const int i = base + tid;\n    const int e = i < npairs ? eflat[i] : -1;\n    const unsigned mk = __match_any_sync(~0u, e);\n    const int rank = __popc(mk & ((1u << lane) - 1u));\n    if (e >= 0 && rank == 0) rt->wc[warp][e] = __popc(mk);\n    bar_consumers();\n    if (tid < NLOC) {\n      int s = rt->run[tid];\n      for (int w = 0; w < NCW; ++w) { const int t = rt->wc[w][tid]; rt->wc[w][tid] = s; s += t; }\n      rt->run[tid] = s;\n    }\n    bar_consumers();\n    if (i < npairs) {\n      if (e >= 0) {\n        const int pos = rt->wc[warp][e] + rank;\n        pos_of[i] = static_cast<int16_t>(pos);\n        pair_of[pos] = static_cast<int16_t>(i);\n      } else {\n        pos_of[i] = -1;\n      }\n    }\n  }\n  // zero this CTA\'s slice of the sum-of-squares accumulators, then publish\n  {\n    const int P = rt->us1[NLOC] > 0 ? rt->off[NLOC - 1] + rt->n[NLOC - 1] : 0;\n    const int z0 = static_cast<int>((long long)P * c / G), z1 = static_cast<int>((long long)P * (c + 1) / G);\n    for (int i = z0 + tid; i < z1; i += NCT) ss[i] = 0.f;\n    __threadfence();\n    bar_consumers();\n    if (tid == 0) atomicAdd(cnt + C_ZERO, 1);\n  }\n\n  const int q = lane & 3;\n  uint32_t Bf[NT][8][8];\n  int st = 0;\n  uint32_t ph = 0;\n  int buf = 0;\n  int cur_pass = -1, cur_e = -1, cur_nb = -1, seg_tiles = 0;\n  float fw[2];   // (1 + post_norm_w) for this thread\'s phase-2 epilogue row, refreshed per tile\n\n  for (int pass = 0; pass < 2; ++pass) {\n    const int ua = pass ? a2 : a1, ub = pass ? b2 : b1;\n    const int* us = pass ? rt->us2 : rt->us1;\n    const int cc = pass ? rt->c2 : rt->c1;\n    const int ntile = pass ? NT2 : NT1;\n    int e=0, nb=0, m=0, mb=0, me=0;\n    if (ua < ub) {\n      decode_unit(ua, us, rt->nblk, cc, ntile, e, nb, m);\n      mb = (m / cc) * cc;\n      me = min(ntile, mb + cc);\n    }\n    for (int u = ua; u < ub; ++u) {\n      const int ne = rt->n[e], off = rt->off[e];\n      if (pass != cur_pass || e != cur_e || nb != cur_nb) {\n        if (cur_pass == 0 && seg_tiles > 0) {  // publish the finished phase-1 segment\n          __threadfence();\n          bar_consumers();\n          if (tid == 0) atomicAdd(cnt + C_DONE + cur_e, seg_tiles);\n        }\n        cur_pass = pass; cur_e = e; cur_nb = nb; seg_tiles = 0;\n        if (pass == 1) {\n          const int target = NT1 * rt->nblk[e];\n          while (ld_acquire(cnt + C_DONE + e) < target) __nanosleep(64);\n          while (ld_acquire(cnt + C_ZERO) < G) __nanosleep(64);\n        }\n#pragma unroll\n        for (int j = 0; j < NT; ++j) {\n          const int idx = nb * NB + j * 8 + (lane >> 2);\n          const __nv_bfloat16* row = nullptr;\n          if (idx < ne) {\n            const int pos = off + idx;\n            row = pass ? inter + static_cast<size_t>(pos) * HD : x + static_cast<size_t>(pair_of[pos] >> 3) * HD;\n          }\n#pragma unroll\n          for (int g = 0; g < 8; ++g) {\n            uint4 v0 = make_uint4(0, 0, 0, 0), v1 = v0;\n            if (row) {\n              const uint4* p = reinterpret_cast<const uint4*>(row + warp * 512 + g * 64 + q * 16);\n              if (pass) { v0 = __ldcg(p); v1 = __ldcg(p + 1); }\n              else {\n                const int token=pair_of[off+idx]>>3;\n                __nv_bfloat16 vv[16];\n                #pragma unroll\n                for(int z=0;z<16;++z)vv[z]=rg_value(rg,token,warp*512+g*64+q*16+z,private_inv[token]);\n                v0=*reinterpret_cast<uint4*>(vv);v1=*reinterpret_cast<uint4*>(vv+8);\n              }\n            }\n            Bf[j][g][0] = v0.x; Bf[j][g][1] = v0.y; Bf[j][g][2] = v0.z; Bf[j][g][3] = v0.w;\n            Bf[j][g][4] = v1.x; Bf[j][g][5] = v1.y; Bf[j][g][6] = v1.z; Bf[j][g][7] = v1.w;\n          }\n        }\n      }\n      if (pass == 1 && tid < 16 * NB) fw[0] = 1.f + __bfloat162float(pnw[m * 16 + (tid & 15)]);\n\n      mbar_wait(&fullb[st], ph);\n      const uint8_t* tile = stages + st * TILE_BYTES;\n      float acc[NT][4];\n#pragma unroll\n      for (int j = 0; j < NT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;\n      {\n        const uint4* wq = reinterpret_cast<const uint4*>(tile) + warp * 8 * 32 + lane;\n        const uint4 scv = *(reinterpret_cast<const uint4*>(tile + TILE_W_BYTES) + warp * 32 + lane);\n        const uint32_t scw[4] = {scv.x, scv.y, scv.z, scv.w};\n#pragma unroll\n        for (int g = 0; g < 8; ++g) {\n          const uint4 wv = wq[g * 32];\n          const uint32_t sw = scw[g >> 1];\n          const uint32_t Sr = scale_e4m3((sw >> ((g & 1) * 16)) & 255u);\n          const uint32_t Sr8 = scale_e4m3((sw >> (8 + (g & 1) * 16)) & 255u);\n          uint32_t o[16];\n          dq8(wv.x, wv.y, o);\n          dq8(wv.z, wv.w, o + 8);\n#pragma unroll\n          for (int s = 0; s < 4; ++s) {\n            const uint32_t* os = o + (s >> 1) * 8 + (s & 1) * 4;\n            const uint32_t a0 = hmul2(os[0], Sr), a1 = hmul2(os[1], Sr8);\n            const uint32_t a2 = hmul2(os[2], Sr), a3 = hmul2(os[3], Sr8);\n#pragma unroll\n            for (int j = 0; j < NT; ++j) mma16816(acc[j], a0, a1, a2, a3, Bf[j][g][2 * s], Bf[j][g][2 * s + 1]);\n          }\n        }\n      }\n      __syncwarp();\n      if (lane == 0) mbar_arrive(&emptyb[st]);\n      if (++st == nstage) { st = 0; ph ^= 1u; }\n\n      float* rp = red + ((buf * NCW + warp) * 32 + lane) * RED;\n#pragma unroll\n      for (int j = 0; j < NT; ++j) reinterpret_cast<float4*>(rp)[j] = make_float4(acc[j][0], acc[j][1], acc[j][2], acc[j][3]);\n      bar_consumers();\n      if (tid == 0) prefetch_one(rt,total_count,first_count,a2,nstage,stages,fullb,emptyb,w13p,w2p);\n      const float* rb = red + buf * NCW * 32 * RED;\n      if (pass == 0) {\n        const float sc = gs13[e] * 256.f;\n        if (tid < 8 * NB) {\n          const int i = tid & 7, n = tid >> 3, j = n >> 3, col = n & 7;\n          const int ls = i * 4 + (col >> 1), rg = j * 4 + (col & 1);\n          float G1 = 0.f, U = 0.f;\n#pragma unroll\n          for (int w = 0; w < NCW; ++w) {\n            G1 += rb[(w * 32 + ls) * RED + rg];\n            U += rb[(w * 32 + ls) * RED + rg + 2];\n          }\n          const int idx = nb * NB + n;\n          if (idx < ne) {\n            G1 *= sc;\n            U *= sc;\n            const float h = G1 / (1.f + __expf(-G1)) * U;\n            inter[static_cast<size_t>(off + idx) * HD + m * 8 + i] = __float2bfloat16(h);\n          }\n        }\n      } else {\n        const float sc = gs2[e] * 256.f;\n        if (tid < 16 * NB) {\n          const int i = tid & 15, n = tid >> 4, j = n >> 3, col = n & 7;\n          const int ls = (i & 7) * 4 + (col >> 1), rg = j * 4 + (col & 1) + ((i >> 3) << 1);\n          float v = 0.f;\n#pragma unroll\n          for (int w = 0; w < NCW; ++w) v += rb[(w * 32 + ls) * RED + rg];\n          v *= sc;\n          const int idx = nb * NB + n;\n          const bool ok = idx < ne;\n          if (ok) y[static_cast<size_t>(off + idx) * HD + m * 16 + i] = v;\n          float t2 = ok ? (v * fw[0]) * (v * fw[0]) : 0.f;\n#pragma unroll\n          for (int d = 8; d > 0; d >>= 1) t2 += __shfl_xor_sync(~0u, t2, d);   // 16 rows of one token\n          if (ok && i == 0) atomicAdd(ss + off + idx, t2);\n        }\n      }\n      buf ^= 1;\n      ++seg_tiles;\n      advance_unit(e, nb, m, mb, me, rt->nblk, cc, ntile);\n    }\n  }\n  if (cur_pass == 0 && seg_tiles > 0) {\n    __threadfence();\n    bar_consumers();\n    if (tid == 0) atomicAdd(cnt + C_DONE + cur_e, seg_tiles);\n  }\n\n  // ---------------- grid barrier ----------------\n  __threadfence();\n  bar_consumers();\n  if (tid == 0) {\n    if (atomicAdd(cnt + C_BAR, 1) == G - 1) {\n      for (int i = 0; i < NLOC; ++i) cnt[C_DONE + i] = 0;   // every CTA is past its waits\n      cnt[C_ZERO] = 0;\n      cnt[C_BAR] = 0;\n      __threadfence();\n      atomicAdd(cnt + C_GEN, 1);\n    } else {\n      while (ld_acquire(cnt + C_GEN) == bar_gen0) __nanosleep(64);\n    }\n  }\n  bar_consumers();\n\n  // ---------------- combine: item = (token, half of the columns) ----------------\n  for (int it = c; it < 2 * M; it += G) {\n    const int t = it >> 1;\n    const int c0 = (it & 1) * 2048 + tid * 8;\n    float f[8];\n    {\n      const uint4 wv = *reinterpret_cast<const uint4*>(pnw + c0);\n      const __nv_bfloat16* wb = reinterpret_cast<const __nv_bfloat16*>(&wv);\n#pragma unroll\n      for (int k = 0; k < 8; ++k) f[k] = 1.f + __bfloat162float(wb[k]);\n    }\n    float acc[8];\n#pragma unroll\n    for (int k = 0; k < 8; ++k) acc[k] = 0.f;\n#pragma unroll\n    for (int j = 0; j < 8; ++j) {\n      const int pos = pos_of[t * 8 + j];\n      if (pos >= 0) {\n        const float sc = tw[t * 8 + j] * rsqrtf(__ldcg(ss + pos) * (1.f / HD) + 1e-8f);\n        const float4* p = reinterpret_cast<const float4*>(y + static_cast<size_t>(pos) * HD + c0);\n        const float4 u0 = __ldcg(p), u1 = __ldcg(p + 1);\n        acc[0] += sc * u0.x * f[0]; acc[1] += sc * u0.y * f[1]; acc[2] += sc * u0.z * f[2]; acc[3] += sc * u0.w * f[3];\n        acc[4] += sc * u1.x * f[4]; acc[5] += sc * u1.y * f[5]; acc[6] += sc * u1.z * f[6]; acc[7] += sc * u1.w * f[7];\n      }\n    }\n    uint4 ov;\n    __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(&ov);\n#pragma unroll\n    for (int k = 0; k < 8; ++k) ob[k] = __float2bfloat16(acc[k]);\n    *reinterpret_cast<uint4*>(out + static_cast<size_t>(t) * HD + c0) = ov;\n  }\n}\n\nsize_t smem_bytes(int nt, int nstage, int M) {\n  const int npairs = M * 8;\n  return static_cast<size_t>(nstage) * TILE_BYTES + 2 * NCW * 32 * nt * 4 * sizeof(float) + 2 * nstage * 8 +\n         sizeof(Route) + ((npairs + 15) & ~15) + 4 * npairs + 8*npairs + 4*M;\n}\n\n}  // namespace\n\nextern "C" int rg_max_smem(){int dev=0,n=0;cudaGetDevice(&dev);cudaDeviceGetAttribute(&n,cudaDevAttrMaxSharedMemoryPerBlockOptin,dev);cudaFuncAttributes attr={};auto e=cudaFuncGetAttributes(&attr,moe_fused<1>);if(e!=cudaSuccess)return -(int)e;n-=attr.sharedSizeBytes;e=cudaFuncSetAttribute(moe_fused<1>,cudaFuncAttributeMaxDynamicSharedMemorySize,n);return e==cudaSuccess?n:-(int)e;}\nextern "C" int rg_smem(int stages,int m){return smem_bytes(1,stages,m);}\nextern "C" int rg_resident(int stages,int m){int n=0;cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,moe_fused<1>,NTHR,smem_bytes(1,stages,m));return n;}\nextern "C" int rg_run(const void** p,int M,int S,int stages,int grid,int64_t ps,int64_t rs,void* stream){\n RouteGlue rg={(const float*)p[14],(const float*)p[15],(const float*)p[16],S,ps,rs,(int*)p[17],(int*)p[18],(float*)p[19],(int*)p[20],(MoERouteMeta*)p[21]};\n moe_fused<1><<<grid,NTHR,smem_bytes(1,stages,M),(cudaStream_t)stream>>>((const __nv_bfloat16*)p[0],(const int*)p[1],(const float*)p[2],(const __nv_bfloat16*)p[3],\n (const uint8_t*)p[4],(const uint8_t*)p[5],(const float*)p[6],(const float*)p[7],(__nv_bfloat16*)p[8],(float*)p[9],(float*)p[10],(int*)p[11],(__nv_bfloat16*)p[12],M,0,stages,rg);\n return (int)cudaGetLastError();\n}\n'
RG_CPP=''
FG_CUDA='#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n#include <cuda/atomic>\n#include <stdint.h>\n#include <math_constants.h>\n\nnamespace glue {\nusing B=__nv_bfloat16;\n__device__ __forceinline__ float warp_sum(float x){\n  for(int d=16;d;d>>=1)x+=__shfl_xor_sync(~0u,x,d);return x;\n}\n__device__ __forceinline__ int arrive(int* p){\n  cuda::atomic_ref<int,cuda::thread_scope_device> a(*p);\n  return a.fetch_add(1,cuda::std::memory_order_acq_rel);\n}\n__device__ __forceinline__ float read_branch(const void* p,size_t i,bool bf){\n  return bf?__bfloat162float(((const B*)p)[i]):((const float*)p)[i];\n}\n__device__ __forceinline__ void cp(B* d,const B* s,bool valid){\n  unsigned a=(unsigned)__cvta_generic_to_shared(d);int n=valid?16:0;\n  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"::"r"(a),"l"(s),"r"(n):"memory");\n}\n__device__ __forceinline__ void mma(float* d,unsigned a0,unsigned a1,unsigned a2,unsigned a3,unsigned b0,unsigned b1){\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "\n    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"\n    :"+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]):"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));\n}\nstruct Args{\n  const B *w,*x;const void* branch;int sb;size_t bs,rs;bool branch_bf;\n  const float *stream,*alpha,*beta,*prew,*postw,*bias,*expertw;\n  const int* local;\n  float *parts,*product,*sout,*route_w;\n  B *hidden,*expert_x;int *ids,*lids,*counts;\n  int m,n,k,ks,mode;\n};\n\n// All CTAs independently reconstruct the exact BF16 hidden row for their K slice.\n// Four sufficient statistics avoid keeping an entire 8192-column row in registers.\n// No global rendezvous is used for the prologue. Only output tile zero writes residuals.\ntemplate<int T,int W>\n__device__ void prologue(Args a,B* sx){\n  int lane=threadIdx.x&31,warp=threadIdx.x>>5;\n  for(int t=warp;t<T;t+=W){\n    int tok=blockIdx.z*T+t;\n    if(tok>=a.m)continue;\n    float bb=0,aa=0,ab=0,cc=0;\n    for(int j=lane;j<8192;j+=32){\n      float z=0;\n      for(int s=0;s<a.sb;++s)z+=read_branch(a.branch,s*a.bs+(size_t)tok*a.rs+j,a.branch_bf);\n      float u=a.alpha[j]*a.stream[(size_t)tok*8192+j],v=a.beta[j]*z;\n      bb=fmaf(z,z,bb);aa=fmaf(u,u,aa);ab=fmaf(u,v,ab);cc=fmaf(v,v,cc);\n    }\n    bb=warp_sum(bb);aa=warp_sum(aa);ab=warp_sum(ab);cc=warp_sum(cc);\n    float rb=rsqrtf(bb*(1.f/8192)+1e-8f);\n    // Same two norms as the separate boundary, with the second squared sum expanded.\n    float rh=rsqrtf((aa+2.f*rb*ab+rb*rb*cc)*(1.f/8192)+1e-5f);\n    for(int q=lane;q<a.ks;q+=32){\n      int j=blockIdx.y*a.ks+q;float z=0;\n      for(int s=0;s<a.sb;++s)z+=read_branch(a.branch,s*a.bs+(size_t)tok*a.rs+j,a.branch_bf);\n      float st=a.alpha[j]*a.stream[(size_t)tok*8192+j]+a.beta[j]*z*rb;\n      sx[t*(a.ks+8)+q]=__float2bfloat16_rn(st*rh*a.prew[j]);\n    }\n    if(blockIdx.x==0&&blockIdx.y==0){\n      for(int j=lane;j<8192;j+=32){\n        float z=0;\n        for(int s=0;s<a.sb;++s)z+=read_branch(a.branch,s*a.bs+(size_t)tok*a.rs+j,a.branch_bf);\n        a.sout[(size_t)tok*8192+j]=a.alpha[j]*a.stream[(size_t)tok*8192+j]+a.beta[j]*z*rb;\n      }\n    }\n  }\n  // Predicated MMA token columns still require initialized shared input.\n  for(int e=threadIdx.x;e<T*a.ks;e+=W*32){int t=e/a.ks,q=e%a.ks;\n    if(blockIdx.z*T+t>=a.m)sx[t*(a.ks+8)+q]=__float2bfloat16(0.f);}\n  __syncthreads();\n}\n\n// One warp per token performs FP32 hi+lo routing and the pre-expert BF16 norm.\ntemplate<int T,int W>\n__device__ void route(Args a){\n  int lane=threadIdx.x&31,warp=threadIdx.x>>5;\n  for(int t=warp;t<T;t+=W){\n    int tok=blockIdx.z*T+t;if(tok>=a.m)continue;\n    const float* p=a.product+(size_t)tok*a.n;\n    float ss=0;\n    for(int j=lane;j<4096;j+=32){float v=p[j];ss=fmaf(v,v,ss);}\n    float inv=rsqrtf(warp_sum(ss)*(1.f/4096)+1e-5f);\n    for(int j=lane;j<4096;j+=32)a.expert_x[(size_t)tok*4096+j]=__float2bfloat16_rn(p[j]*inv*a.expertw[j]);\n    float score[8],biased[8];\n    #pragma unroll\n    for(int j=0;j<8;++j){int e=j*32+lane;score[j]=1.f/(1.f+__expf(-(p[4096+e]+p[4352+e])*0.1767766952966369f));biased[j]=score[j]+a.bias[e];}\n    int picks[8];float probs[8],total=0;\n    #pragma unroll\n    for(int k=0;k<8;++k){\n      float best=-CUDART_INF_F;int id=256;\n      #pragma unroll\n      for(int j=0;j<8;++j){int e=j*32+lane;if(biased[j]>best||(biased[j]==best&&e<id)){best=biased[j];id=e;}}\n      #pragma unroll\n      for(int d=16;d;d>>=1){float other=__shfl_xor_sync(~0u,best,d);int oi=__shfl_xor_sync(~0u,id,d);\n        if(other>best||(other==best&&oi<id)){best=other;id=oi;}}\n      float prob=0;\n      #pragma unroll\n      for(int j=0;j<8;++j){if(j*32+lane==id){prob=score[j];biased[j]=-CUDART_INF_F;}}\n      prob=warp_sum(prob);picks[k]=id;probs[k]=prob;total+=prob;\n    }\n    if(lane<8){int id=picks[lane],local=(id-*a.local+256)%256;\n      a.ids[tok*8+lane]=id;a.route_w[tok*8+lane]=probs[lane]/(total+1e-15f);a.lids[tok*8+lane]=local<32?local:-1;}\n  }\n}\n\ntemplate<int W>\n__device__ float cta_sum(float x,float* tmp){\n  x=warp_sum(x);if((threadIdx.x&31)==0)tmp[threadIdx.x>>5]=x;\n  __syncthreads();float y=threadIdx.x<W?tmp[threadIdx.x]:0.f;\n  y=warp_sum(y);if(threadIdx.x==0)tmp[W]=y;\n  __syncthreads();return tmp[W];\n}\n\ntemplate<int T,int W>\n__device__ void boundary2(Args a){\n  int lane=threadIdx.x&31,warp=threadIdx.x>>5;\n  if(a.m==1){\n    __shared__ float tmp[4][W+1];float bb=0,aa=0,ab=0,cc=0;\n    for(int j=threadIdx.x;j<8192;j+=W*32){\n      float z=j<a.n?a.product[j]*(a.mode==3?1.f:a.postw[j]):0.f;\n      float u=a.alpha[j]*a.stream[j],v=a.beta[j]*z;\n      bb=fmaf(z,z,bb);aa=fmaf(u,u,aa);ab=fmaf(u,v,ab);cc=fmaf(v,v,cc);\n    }\n    bb=cta_sum<W>(bb,tmp[0]);aa=cta_sum<W>(aa,tmp[1]);ab=cta_sum<W>(ab,tmp[2]);cc=cta_sum<W>(cc,tmp[3]);\n    float rb=rsqrtf(bb*(1.f/8192)+1e-8f),rh=rsqrtf((aa+2*rb*ab+rb*rb*cc)*(1.f/8192)+1e-5f);\n    for(int j=threadIdx.x;j<8192;j+=W*32){\n      float z=j<a.n?a.product[j]*(a.mode==3?1.f:a.postw[j]):0.f;\n      float st=a.alpha[j]*a.stream[j]+a.beta[j]*z*rb;\n      a.sout[j]=st;a.hidden[j]=__float2bfloat16_rn(st*rh*a.prew[j]);\n    }\n    return;\n  }\n  for(int t=warp;t<T;t+=W){\n    int tok=blockIdx.z*T+t;if(tok>=a.m)continue;\n    float bb=0,aa=0,ab=0,cc=0;\n    for(int j=lane;j<8192;j+=32){\n      float z=j<a.n?a.product[(size_t)tok*a.n+j]*(a.mode==3?1.f:a.postw[j]):0.f;\n      float u=a.alpha[j]*a.stream[(size_t)tok*8192+j],v=a.beta[j]*z;\n      bb=fmaf(z,z,bb);aa=fmaf(u,u,aa);ab=fmaf(u,v,ab);cc=fmaf(v,v,cc);\n    }\n    bb=warp_sum(bb);aa=warp_sum(aa);ab=warp_sum(ab);cc=warp_sum(cc);\n    float rb=rsqrtf(bb*(1.f/8192)+1e-8f);\n    float rh=rsqrtf((aa+2.f*rb*ab+rb*rb*cc)*(1.f/8192)+1e-5f);\n    for(int j=lane;j<8192;j+=32){\n      float z=j<a.n?a.product[(size_t)tok*a.n+j]*(a.mode==3?1.f:a.postw[j]):0.f;\n      float st=a.alpha[j]*a.stream[(size_t)tok*8192+j]+a.beta[j]*z*rb;\n      a.sout[(size_t)tok*8192+j]=st;\n      a.hidden[(size_t)tok*8192+j]=__float2bfloat16_rn(st*rh*a.prew[j]);\n    }\n  }\n}\n\ntemplate<int R,int T,int W,int MODE>\n__global__ __launch_bounds__(W*32) void kernel(Args a,bool pdl){\n  constexpr bool POOL=MODE==1;\n  constexpr int BK=128,STR=136,STAGES=2,CELLS=(R/16)*(T/8)/W;\n  extern __shared__ __align__(16) B mem[];\n  B* sx=mem;B* sw=mem+(POOL?T*(a.ks+8):0);\n  int tid=threadIdx.x,lane=tid&31,warp=tid>>5,g=lane>>2,l=lane&3;\n  int row0=blockIdx.x*R,tok0=blockIdx.z*T;\n  int begin=blockIdx.y*a.ks,steps=a.ks/BK;\n  if(pdl){cudaTriggerProgrammaticLaunchCompletion();cudaGridDependencySynchronize();}\n  if(POOL)prologue<T,W>(a,sx);\n  auto load=[&](int step){\n    B* s=sw+(step%STAGES)*(R+(POOL?0:T))*STR;\n    for(int v=tid;v<(R+(POOL?0:T))*BK/8;v+=W*32){\n      int r=v/(BK/8),k=(v%(BK/8))*8;\n      bool weight=r<R;int global=weight?row0+r:tok0+r-R;\n      bool valid=step<steps&&global<(weight?a.n:a.m);\n      const B* src=valid?(weight?a.w:a.x)+(size_t)global*a.k+begin+step*BK+k:a.w;\n      cp(s+r*STR+k,src,valid);\n    }\n    asm volatile("cp.async.commit_group;":::"memory");\n  };\n  float accum[CELLS][4]={};load(0);\n  for(int step=0;step<steps;++step){\n    load(step+1);asm volatile("cp.async.wait_group 1;":::"memory");__syncthreads();\n    B* s=sw+(step%STAGES)*(R+(POOL?0:T))*STR;\n    #pragma unroll\n    for(int k=0;k<BK;k+=16){\n      #pragma unroll\n      for(int c=0;c<CELLS;++c){\n        int cell=warp+c*W,rn=(cell/(T/8))*16,tm=(cell%(T/8))*8;\n        const B* wa=s+(rn+g)*STR+k+l*2;\n        const B* xb=POOL?sx+(tm+g)*(a.ks+8)+step*BK+k+l*2:s+(R+tm+g)*STR+k+l*2;\n        mma(accum[c],*(const unsigned*)wa,*(const unsigned*)(wa+8*STR),*(const unsigned*)(wa+8),*(const unsigned*)(wa+8*STR+8),\n          *(const unsigned*)xb,*(const unsigned*)(xb+(POOL?8:8)));\n      }\n    }\n    __syncthreads();\n  }\n  asm volatile("cp.async.wait_group 0;":::"memory");\n  int nt=(a.n+R-1)/R,tiles=nt*((a.m+T-1)/T),tile=blockIdx.z*nt+blockIdx.x,splits=a.k/a.ks;\n  #pragma unroll\n  for(int c=0;c<CELLS;++c){\n    int cell=warp+c*W,rn=(cell/(T/8))*16+g,tm=(cell%(T/8))*8+l*2;\n    #pragma unroll\n    for(int i=0;i<4;++i){int r=rn+(i/2)*8,t=tm+(i&1);\n      a.parts[((size_t)tile*splits+blockIdx.y)*R*T+t*R+r]=accum[c][i];}\n  }\n  // Every writer publishes before thread zero arrives. Acq/rel arrivals form a\n  // transitive happens-before chain; non-final CTAs leave rather than spin.\n  __threadfence();__syncthreads();__shared__ int last;\n  if(tid==0)last=arrive(a.counts+tile)==splits-1;\n  __syncthreads();if(!last)return;\n  for(int e=tid;e<R*T;e+=W*32){int r=e%R,t=e/R;\n    if(row0+r<a.n&&tok0+t<a.m){float v=0;\n      for(int s=0;s<splits;++s)v+=((volatile float*)a.parts)[((size_t)tile*splits+s)*R*T+e];\n      a.product[(size_t)(tok0+t)*a.n+row0+r]=v;}}\n  __threadfence();__syncthreads();\n  if(tid==0){atomicExch(a.counts+tile,0);last=arrive(a.counts+tiles+blockIdx.z)==nt-1;}\n  __syncthreads();if(!last)return;\n  if(MODE==1||MODE==2)route<T,W>(a);else boundary2<T,W>(a);\n  __syncthreads();if(tid==0)atomicExch(a.counts+tiles+blockIdx.z,0);\n}\n\ntemplate<int R,int T,int W,int MODE>\nint run(Args a,bool pdl,cudaStream_t stream){\n  constexpr bool P=MODE==1;\n  int bytes=((P?T*(a.ks+8):0)+2*(R+(P?0:T))*136)*2;\n  auto f=kernel<R,T,W,MODE>;\n  auto err=cudaFuncSetAttribute(f,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);\n  if(err!=cudaSuccess)return (int)err;\n  dim3 grid((a.n+R-1)/R,a.k/a.ks,(a.m+T-1)/T);\n  if(pdl){cudaLaunchConfig_t c={};c.gridDim=grid;c.blockDim=dim3(W*32);c.dynamicSmemBytes=bytes;c.stream=stream;\n    cudaLaunchAttribute attr={};attr.id=cudaLaunchAttributeProgrammaticStreamSerialization;attr.val.programmaticStreamSerializationAllowed=1;\n    c.attrs=&attr;c.numAttrs=1;err=cudaLaunchKernelEx(&c,f,a,pdl);if(err!=cudaSuccess)return (int)err;\n  }else f<<<grid,W*32,bytes,stream>>>(a,pdl);\n  return (int)cudaGetLastError();\n}\n}\nextern "C" int glue_launch(const void** p,int m,int n,int k,int ks,int sb,size_t bs,size_t rs,int branch_bf,int pool,int recipe,int pdl,void* stream){\n  using namespace glue;Args a={};\n  a.w=(const B*)p[0];a.x=(const B*)p[1];a.branch=p[2];a.stream=(const float*)p[3];\n  a.alpha=(const float*)p[4];a.beta=(const float*)p[5];a.prew=(const float*)p[6];a.postw=(const float*)p[7];\n  a.bias=(const float*)p[8];a.expertw=(const float*)p[9];a.local=(const int*)p[10];\n  a.parts=(float*)p[11];a.product=(float*)p[12];a.sout=(float*)p[13];a.hidden=(B*)p[14];\n  a.expert_x=(B*)p[15];a.ids=(int*)p[16];a.route_w=(float*)p[17];a.lids=(int*)p[18];a.counts=(int*)p[19];\n  a.m=m;a.n=n;a.k=k;a.ks=ks;a.mode=pool;a.sb=sb;a.bs=bs;a.rs=rs;a.branch_bf=branch_bf;\n  if(k%ks||ks%128||m>256||m<1||n%64)return (int)cudaErrorInvalidValue;\n  #define CALL(R,T,W,P) return run<R,T,W,P>(a,pdl,(cudaStream_t)stream)\n  #define CHOOSE(MODE) if(recipe==0){CALL(64,8,4,MODE);}else{CALL(64,16,8,MODE);}\n  if(pool==1){CHOOSE(1)}else if(pool==2){CHOOSE(2)}else if(pool==3){CHOOSE(3)}else{CHOOSE(0)}\n  #undef CHOOSE\n  #undef CALL\n}\n'
import ctypes as _fgct
import hashlib as _fghash
import os as _fgos
import torch
from torch.utils.cpp_extension import load_inline as _fgload

_FG_LIB=None

def fg_library():
    global _FG_LIB
    if _FG_LIB is None:
        arch='9.0a' if torch.cuda.get_device_capability()[0]==9 else '12.0'
        old=_fgos.environ.get('TORCH_CUDA_ARCH_LIST');_fgos.environ['TORCH_CUDA_ARCH_LIST']=arch
        try:
            path=_fgload(name='muse_fused_glue_'+_fghash.sha256(FG_CUDA.encode()).hexdigest()[:16],
                cpp_sources='',cuda_sources=FG_CUDA,functions=None,is_python_module=False,
                no_implicit_headers=True,extra_cuda_cflags=['-O3','-std=c++17','-lineinfo'],verbose=False)
        finally:
            if old is None:_fgos.environ.pop('TORCH_CUDA_ARCH_LIST',None)
            else:_fgos.environ['TORCH_CUDA_ARCH_LIST']=old
        _FG_LIB=_fgct.CDLL(path)
        _FG_LIB.glue_launch.argtypes=[_fgct.c_void_p]+[_fgct.c_int]*5+[_fgct.c_size_t]*2+[_fgct.c_int]*4+[_fgct.c_void_p]
        _FG_LIB.glue_launch.restype=_fgct.c_int
    return _FG_LIB


class FusedGlue:
    """Private per-call-site scratch. All arrivals return to zero on each replay."""
    def __init__(self,state,tokens,pool=True,recipe=0,ks=1024):
        self.lib=fg_library();self.state=state;self.m=tokens;self.pool=pool;self.recipe=recipe
        self.mode=1 if pool is True else 0 if pool is False else pool
        self.ks=ks;self.t=8 if recipe==0 else 16;self.r=64
        self.name='o' if self.mode==3 else 'pool_router' if self.mode in (1,2) else ('unpooler_slice' if tokens<=8 else 'unpooler')
        w=state[self.name];self.n,self.k=w.shape;dev=w.device
        nt=self.n//64;mt=(tokens+self.t-1)//self.t;splits=self.k//ks
        self.parts=torch.empty((nt*mt,splits,64*self.t),device=dev,dtype=torch.float32)
        self.counts=torch.zeros(nt*mt+mt,device=dev,dtype=torch.int32)
        self.product=torch.empty((tokens,self.n),device=dev,dtype=torch.float32)
        self.sout=torch.empty((tokens,8192),device=dev,dtype=torch.float32)
        self.hidden=torch.empty((tokens,8192),device=dev,dtype=torch.bfloat16)
        self.x=torch.empty((tokens,4096),device=dev,dtype=torch.bfloat16)
        self.ids=torch.empty((tokens,8),device=dev,dtype=torch.int32)
        self.wts=torch.empty((tokens,8),device=dev,dtype=torch.float32)
        self.lids=torch.empty_like(self.ids)

    def __call__(self,value,stream,local=None,pdl=True):
        st=self.state;pool=self.mode==1;front=self.mode in (1,3)
        if pool:
            p,s=value;x=p;bs=p.stride(0);rs=p.stride(1)
            if p.shape[1:]!=(self.m,8192) or p.dtype not in (torch.float32,torch.bfloat16):
                raise ValueError('boundary-1 producer ABI')
        else:
            x=value;p=x;s=1;bs=0;rs=0
            if x.shape!=(self.m,self.k) or x.dtype!=torch.bfloat16 or not x.is_contiguous():
                raise ValueError('unpooler BF16 input ABI')
        tensors=[st[self.name],x,p,stream,st['alpha1' if front else 'alpha2'],st['beta1' if front else 'beta2'],
            st['pre_ffn_w' if front else 'next_pre_attn_w'],st['post_ffn_w'],st['router_bias'],st['pre_expert_w'],
            local if local is not None else self.ids,self.parts,self.product,self.sout,self.hidden,
            self.x,self.ids,self.wts,self.lids,self.counts]
        args=(_fgct.c_void_p*20)(*(v.data_ptr() for v in tensors))
        err=self.lib.glue_launch(args,self.m,self.n,self.k,self.ks,s,bs,rs,int(p.dtype==torch.bfloat16),
            self.mode,self.recipe,int(pdl),torch.cuda.current_stream(stream.device).cuda_stream)
        if err:raise RuntimeError(f'fused glue CUDA={err} M={self.m} pool={pool}')
        if self.mode in (1,2):return (self.product.unsqueeze(0),1),self.sout,(self.x,self.ids,self.wts,self.lids)
        return self.hidden,self.sout


def fg_prepare(state,tokens=(1,8,16,32),recipe=0):
    state['fused_glue_ops']={(t,p):FusedGlue(state,t,p,recipe) for t in tokens for p in (True,False,2,3)}


def fg_run(m,data,state,plan,cell='all'):
    """Table-selected pool, tail, all, or epilogue-only fusion."""
    if cell not in ('pool','tail','all','epi'):raise ValueError('unknown glue cell')
    t=data['config']['tokens'];sync=[None,None,None]
    if plan['glue']=='best5':
        raw=(torch.empty if m._best7_old_chunks(t)==1 else torch.zeros)((15*t,),device=data['stream'].device,dtype=torch.int32)
        si=raw[:3*t].view(3,t);sf=raw[3*t:].view(torch.float32).view(3,t,4)
        sync=[(sf[i],si[i]) for i in range(3)]
    qkv=m._best7_project(data['attn_input'],'qkv_gate',state,plan)
    attn=m._best7_attention(qkv,data,state,plan)
    if cell=='epi':
        h,s1=state['fused_glue_ops'][t,3](attn,data['stream'])
        pr,_,(x,ids,wts,lids)=state['fused_glue_ops'][t,2](h,s1,data['local_start'])
    else:a=m._best7_project(attn,'o',state,plan)
    if cell in ('pool','all'):
        pr,s1,(x,ids,wts,lids)=state['fused_glue_ops'][t,True](a,data['stream'],data['local_start'])
    elif cell!='epi':
        h,s1=m._best7_boundary(a,data['stream'],state,plan,'b1',sync[0])
        pr=m._best7_project(h,'pool_router',state,plan,out_dtype=torch.float32)
        x,ids,wts,lids=m._best7_route(pr,state,data,plan,sync[1])
    moe=m._best7_expert(pr,x,wts,lids,state,plan)
    if cell in ('tail','all','epi'):
        hidden,s2=state['fused_glue_ops'][t,False](moe,s1)
    else:
        name='unpooler_slice' if t<=8 else 'unpooler'
        f=m._best7_project(moe,name,state,plan)
        hidden,s2=m._best7_boundary(f,s1,state,plan,'b2',sync[2])
    return hidden,s2,ids

import hashlib as _rghash
import torch
from torch.utils.cpp_extension import load_inline as _rgload
import triton
import triton.language as tl

import ctypes as _rgct
import os as _rgos
_RG_MOD=None

class _RGNative:
    def __init__(self,path):
        self.lib=_rgct.CDLL(path)
        self.lib.rg_run.argtypes=[_rgct.c_void_p]+[_rgct.c_int]*4+[_rgct.c_int64]*2+[_rgct.c_void_p]
        self.lib.rg_run.restype=_rgct.c_int
        self.lib.rg_smem.argtypes=[_rgct.c_int]*2;self.lib.rg_smem.restype=_rgct.c_int
        self.lib.rg_resident.argtypes=[_rgct.c_int]*2;self.lib.rg_resident.restype=_rgct.c_int
        self.lib.rg_max_smem.restype=_rgct.c_int
        self.smem_optin=self.lib.rg_max_smem()
        if self.smem_optin<=0:raise RuntimeError(f'route MoE smem opt-in CUDA={-self.smem_optin}')
    def smem_needed(self,nt,stages,m):return self.lib.rg_smem(stages,m)
    def resident_blocks(self,nt,stages,m):return self.lib.rg_resident(stages,m)
    def run(self,x,lids,wts,pnw,w13,w2,gs13,gs2,inter,y,ss,cnt,out,
            local_start,nt,stages,grid,p,bias,pre,ids,out_lids,out_wts,local,s,meta):
        if local_start!=0 or nt!=1:raise ValueError('NT1 private local routes only')
        values=[x,lids,wts,pnw,w13,w2,gs13,gs2,inter,y,ss,cnt,out,x,p,bias,pre,ids,out_lids,out_wts,local,meta]
        ptrs=(_rgct.c_void_p*22)(*(t.data_ptr() for t in values))
        err=self.lib.rg_run(ptrs,x.shape[0],s,stages,grid,p.stride(0),p.stride(1),torch.cuda.current_stream(p.device).cuda_stream)
        if err:raise RuntimeError(f'route MoE launch CUDA={err}')


def rg_module():
    global _RG_MOD
    if _RG_MOD is None:
        arch='9.0a' if torch.cuda.get_device_capability()[0]==9 else '12.0'
        old=_rgos.environ.get('TORCH_CUDA_ARCH_LIST');_rgos.environ['TORCH_CUDA_ARCH_LIST']=arch
        try:
            path=_rgload(name='muse_route_moe_'+_rghash.sha256(RG_CUDA.encode()).hexdigest()[:16],
                cpp_sources='',cuda_sources=RG_CUDA,functions=None,is_python_module=False,no_implicit_headers=True,
                extra_cuda_cflags=['-O3','-std=c++20','-lineinfo'],verbose=False)
        finally:
            if old is None:_rgos.environ.pop('TORCH_CUDA_ARCH_LIST',None)
            else:_rgos.environ['TORCH_CUDA_ARCH_LIST']=old
        _RG_MOD=_RGNative(path)
    return _RG_MOD


def rg_expert(pr,state,data):
    """NT1 prologue derives routes and BF16 expert inputs from original FP32 PR."""
    p,s=pr;m=p.shape[1];device=p.device;bank=state['moe_bank'];mod=state['route_moe_mod']
    stages=2
    while stages>1 and mod.smem_needed(1,stages,m)>mod.smem_optin:stages-=1
    if mod.smem_needed(1,stages,m)>mod.smem_optin:raise ValueError('route MoE shared-memory limit')
    if p.dtype!=torch.float32:raise ValueError('FP32 pool/router required')
    ids=torch.empty((m,8),device=device,dtype=torch.int32);lids=torch.empty_like(ids)
    wts=torch.empty((m,8),device=device,dtype=torch.float32)
    placeholder=torch.empty((m,4096),device=device,dtype=torch.bfloat16)
    inter=torch.empty((8*m,4096),device=device,dtype=torch.bfloat16)
    y=torch.empty((8*m,4096),device=device,dtype=torch.float32)
    ss=torch.empty(8*m,device=device,dtype=torch.float32)
    out=torch.empty_like(placeholder)
    meta=torch.empty(736 if m<=32 else 0,device=device,dtype=torch.int32)
    mod.run(placeholder,lids,wts,bank['pnw'],bank['w13p'],bank['w2p'],bank['gs13'],bank['gs2'],
        inter,y,ss,state['route_moe_counts'],out,0,1,stages,
        bank['grid']*mod.resident_blocks(1,stages,m),p,state['router_bias'],state['pre_expert_w'],
        ids,lids,wts,data['local_start'],s,meta)
    return out,ids,wts,lids,meta


@triton.jit
def _rg_humming_prologue(PR,S:tl.constexpr,PS:tl.constexpr,RS:tl.constexpr,BIAS,LSTART,
                         POOLED,IDS,WTS,LIDS,POS,COUNTS,ARRIVE,M:tl.constexpr,B:tl.constexpr):
    t=tl.program_id(0);d=tl.arange(0,4096);e=tl.arange(0,256)
    p=tl.full((4096,),0.,tl.float32);logit=tl.full((256,),0.,tl.float32)
    bias=tl.load(BIAS+e);start=tl.load(LSTART)
    for s in tl.static_range(S):
        row=PR+s*PS+t*RS
        p+=tl.load(row+d).to(tl.float32)
        logit+=tl.load(row+4096+e).to(tl.float32)+tl.load(row+4352+e).to(tl.float32)
    tl.store(POOLED+t*4096+d,p.to(tl.bfloat16))
    scores=tl.sigmoid(logit*0.1767766952966369);biased=scores+bias
    bits=biased.to(tl.int32,bitcast=True)
    ordered=tl.where(bits<0,~bits,bits^(-2147483648)).to(tl.uint32)
    keys=(ordered.to(tl.uint64)<<32)|(255-e).to(tl.uint64)
    winners=tl.topk(keys,8);ids=255-(winners&0xffffffff).to(tl.int32)
    scores=tl.gather(scores,ids,0);k=tl.arange(0,8)
    tl.store(IDS+t*8+k,ids);tl.store(WTS+t*8+k,scores/(tl.sum(scores,0)+1.e-15))
    local=(ids-start+256)%256;tl.store(LIDS+t*8+k,tl.where(local<32,local,-1))
    # Every producer thread publishes before the CTA leader's arrival. Exactly
    # the final token CTA computes deterministic token-major positions/counts.
    tl.inline_asm_elementwise("fence.acq_rel.gpu; mov.u32 $0,$1;", "=r,r",
        [tl.arange(0,256)],dtype=tl.int32,is_pure=False,pack=1)
    tl.debug_barrier()
    previous=tl.atomic_add(ARRIVE,1,sem='acq_rel')
    if previous==M-1:
        i=tl.arange(0,B);tokens=i//8;slots=i%8
        local_ids=tl.load(LIDS+i,i<M*8,-1)
        # One scalar expert at a time keeps the live vector at M*8 elements.
        for expert in range(32):
            match=(i<M*8)&(local_ids==expert)
            rank=tl.cumsum(match.to(tl.int32),0)-1
            tl.store(POS+i,rank,match)
            tl.store(COUNTS+expert,tl.sum(match.to(tl.int32),0))
        tl.inline_asm_elementwise("fence.acq_rel.gpu; mov.u32 $0,$1;", "=r,r",
            [tl.arange(0,256)],dtype=tl.int32,is_pure=False,pack=1)
        tl.debug_barrier()
        tl.atomic_xchg(ARRIVE,0,sem='release')


def rg_humming(pr,state,data):
    p,s=pr;m=p.shape[1];dev=p.device
    pooled=torch.empty((m,4096),device=dev,dtype=torch.bfloat16)
    ids=torch.empty((m,8),device=dev,dtype=torch.int32);lids=torch.empty_like(ids);pos=torch.empty_like(ids)
    wts=torch.empty((m,8),device=dev,dtype=torch.float32);counts=torch.empty(32,device=dev,dtype=torch.int32)
    _rg_humming_prologue[(m,)](p,s,p.stride(0),p.stride(1),state['router_bias'],data['local_start'],
        pooled,ids,wts,lids,pos,counts,state['route_humming_count'],M=m,B=triton.next_power_of_2(m*8),num_warps=8)
    return pooled,ids,wts,lids,pos,counts


def rg_run(m,data,state,plan,cell='route'):
    t=data['config']['tokens'];sync=[None,None,None]
    if cell not in ('route','tail','both','pool_epilogue'):raise ValueError('unknown route glue cell')
    if plan['glue']=='best5':
        raw=(torch.empty if m._best7_old_chunks(t)==1 else torch.zeros)((15*t,),device=data['stream'].device,dtype=torch.int32)
        si=raw[:3*t].view(3,t);sf=raw[3*t:].view(torch.float32).view(3,t,4);sync=[(sf[i],si[i]) for i in range(3)]
    qkv=m._best7_project(data['attn_input'],'qkv_gate',state,plan)
    attn=m._best7_attention(qkv,data,state,plan)
    a=m._best7_project(attn,'o',state,plan)
    h,s1=m._best7_boundary(a,data['stream'],state,plan,'b1',sync[0])
    if cell=='pool_epilogue':
        pr,_,(x,ids,wts,lids)=state['route_pool_ops'][t](h,s1,data['local_start'])
        moe=m._best7_expert(pr,x,wts,lids,state,plan)
    else:
        pr=m._best7_project(h,'pool_router',state,plan,out_dtype=torch.float32)
        if cell in ('route','both'):
            if plan['moe']=='v9nt1':moe,ids,_,_,_=rg_expert(pr,state,data)
            elif plan['moe']=='humming_masked':
                pooled,ids,wts,lids,pos,counts=rg_humming(pr,state,data)
                moe=m._pf_experts_humming_masked(pooled,wts,lids,pos,counts,state['moe6_adapters'])
            else:raise ValueError('unsupported expert for route fusion')
        else:
            x,ids,wts,lids=m._best7_route(pr,state,data,plan,sync[1])
            moe=m._best7_expert(pr,x,wts,lids,state,plan)
    if cell in ('tail','both'):
        hidden,s2=state['route_tail_ops'][t](moe,s1)
    else:
        name='unpooler_slice' if t<=8 else 'unpooler'
        f=m._best7_project(moe,name,state,plan)
        hidden,s2=m._best7_boundary(f,s1,state,plan,'b2',sync[2])
    return hidden,s2,ids

# Populate only from a passing H200 whole-layer comparison.
BEST7_ROUTE_GLUE_CELLS={}
_rg_base_prepare=prepare
_rg_base_kernel=kernel_fn

class _RGNamespace:
    def __init__(self,ns):self.ns=ns
    def __getattr__(self,name):return self.ns[name]


def prepare(weights):
    state=_rg_base_prepare(weights)
    tokens={b*q for b,q,w in BEST7_ROUTE_GLUE_CELLS}
    if not tokens:return state
    if any(t not in (1,8,16,32,128,256) for t in tokens):raise ValueError('unadmitted glue M')
    state['route_moe_mod']=rg_module()
    state['route_moe_counts']=torch.zeros_like(state['moe_bank']['cnt'])
    state['route_humming_count']=torch.zeros(1,device=state['o'].device,dtype=torch.int32)
    state['route_pool_ops']={t:FusedGlue(state,t,2,1 if t>=128 else 0,1024) for t in tokens}
    state['route_tail_ops']={t:FusedGlue(state,t,False,1 if t>=128 else 0,1024) for t in tokens}
    return state


def kernel_fn(data,state):
    c=data['config'];key=c['batch'],c['q_len'],c['window'];cell=BEST7_ROUTE_GLUE_CELLS.get(key)
    if cell is None:return _rg_base_kernel(data,state)
    return rg_run(_RGNamespace(globals()),data,state,state['best7_plans'][key],cell)
