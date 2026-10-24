"""Generated streaming projections with in-launch boundary continuations."""
SP_CUDA = '#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n#include <stdint.h>\n\n// Standalone ABI: no PyTorch headers, no architecture-specific host branch.\n// The exact cp.async + mma.sync path also executes on the local sm_120.\nnamespace stream_proj {\nusing B = __nv_bfloat16;\n__device__ __forceinline__ uint4 load16(const B* p) {\n  uint4 v;\n  asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"\n      : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));\n  return v;\n}\n__device__ __forceinline__ void cp16(B* dst, const B* src, bool valid) {\n  unsigned a = (unsigned)__cvta_generic_to_shared(dst);\n  int bytes = valid ? 16 : 0;\n  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"\n      :: "r"(a), "l"(src), "r"(bytes) : "memory");\n}\n__device__ __forceinline__ void mma(float* d, unsigned a0, unsigned a1,\n                                   unsigned a2, unsigned a3, unsigned b0, unsigned b1) {\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "\n      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"\n      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])\n      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));\n}\n__device__ __forceinline__ void output(void* y, size_t i, float v, bool fp32) {\n  if (fp32) ((float*)y)[i] = v;\n  else ((B*)y)[i] = __float2bfloat16_rn(v);\n}\n\n} // namespace stream_proj\n#include <cuda/atomic>\n\nnamespace stream_proj {\n// All scratch is private to one call site. Publication is a last-arrival\n// continuation: a CTA never waits for another CTA to become resident.\nstruct Glue {\n  int phase, prologue;\n  const float *stream, *alpha, *beta, *pre, *post, *bias, *input_pre;\n  float* stream_out;\n  B* norm;\n  int *ids, *local_ids;\n  float* weights;\n  const int* start;\n  unsigned* ready;\n  float* moments;\n};\n\n__device__ __forceinline__ float warp_sum(float v) {\n#pragma unroll\n  for (int d=16; d; d>>=1) v += __shfl_xor_sync(0xffffffff, v, d);\n  return v;\n}\n__device__ __forceinline__ unsigned long long warp_max(unsigned long long v) {\n#pragma unroll\n  for (int d=16; d; d>>=1) {\n    auto b=__shfl_xor_sync(0xffffffff, v, d); v=v>b?v:b;\n  }\n  return v;\n}\n__device__ __forceinline__ float group_sum(float v, int threads, float* tmp) {\n  v=warp_sum(v);\n  if(threads==32) return v;\n  if ((threadIdx.x&31)==0) tmp[threadIdx.x/32]=v;\n  __syncthreads();\n  const int first=(threadIdx.x/threads)*(threads/32);\n  float s=0;\n  for(int i=0;i<threads/32;++i) s+=tmp[first+i];\n  __syncthreads();\n  return s;\n}\n__device__ __forceinline__ unsigned long long group_max(unsigned long long v, int threads,\n                                                       unsigned long long* tmp) {\n  v=warp_max(v);\n  if(threads==32) return v;\n  if ((threadIdx.x&31)==0) tmp[threadIdx.x/32]=v;\n  __syncthreads();\n  const int first=(threadIdx.x/threads)*(threads/32);\n  unsigned long long s=0;\n  for(int i=0;i<threads/32;++i) s=s>tmp[first+i]?s:tmp[first+i];\n  __syncthreads();\n  return s;\n}\n\ntemplate<int T, int THREADS>\n__device__ __forceinline__ void init_input(Glue gl, const B* x, int M, int K, float* inv) {\n  const int tok0=blockIdx.z*T;\n  // One warp per input row; this pass is needed only when the producer leaves\n  // its FP32 residual stream for the pool/router prologue to normalize.\n  for (int t=threadIdx.x/32;t<T;t+=THREADS/32) {\n    float s=0;\n    if(tok0+t<M) {\n      const float* row=(const float*)x+(size_t)(tok0+t)*K;\n      for(int k=(threadIdx.x&31)*4;k<K;k+=32*4) {\n        float4 v=*(const float4*)(row+k);\n        s+=v.x*v.x+v.y*v.y+v.z*v.z+v.w*v.w;\n      }\n    }\n    s=warp_sum(s);\n    if((threadIdx.x&31)==0) inv[t]=rsqrtf(s/K+1.e-5f);\n  }\n  __syncthreads();\n}\n\n__device__ __forceinline__ uint4 input16(Glue gl, const B* x, int tok, int k, int K, float inv) {\n  if(!gl.prologue) return *(const uint4*)(x+(size_t)tok*K+k);\n  const float* row=(const float*)x+(size_t)tok*K+k;\n  float4 a=*(const float4*)row, b=*(const float4*)(row+4);\n  float4 p=*(const float4*)(gl.input_pre+k), q=*(const float4*)(gl.input_pre+k+4);\n  uint4 v;\n  ((B*)&v)[0]=__float2bfloat16_rn((a.x*inv)*p.x);\n  ((B*)&v)[1]=__float2bfloat16_rn((a.y*inv)*p.y);\n  ((B*)&v)[2]=__float2bfloat16_rn((a.z*inv)*p.z);\n  ((B*)&v)[3]=__float2bfloat16_rn((a.w*inv)*p.w);\n  ((B*)&v)[4]=__float2bfloat16_rn((b.x*inv)*q.x);\n  ((B*)&v)[5]=__float2bfloat16_rn((b.y*inv)*q.y);\n  ((B*)&v)[6]=__float2bfloat16_rn((b.z*inv)*q.z);\n  ((B*)&v)[7]=__float2bfloat16_rn((b.w*inv)*q.w);\n  return v;\n}\n\ntemplate<int R, int T, int THREADS>\n__device__ __noinline__ void publish(Glue gl, float* raw, int M, int N, int nt) {\n  if(!gl.phase) return;\n  __shared__ int done[T];\n  __shared__ float tmp[THREADS/32];\n  __shared__ unsigned long long keys_tmp[THREADS/32];\n  const int tok0=blockIdx.z*T;\n  const int valid=M-tok0<T?M-tok0:T;\n  int groups=1;\n  while(groups*2<=THREADS/32 && groups*2<=valid) groups*=2;\n  const int threads=THREADS/groups, g=threadIdx.x/threads, lane=threadIdx.x%threads;\n  if(gl.moments) {\n    // Reduce local moments while all output-row CTAs are still available.\n    // The final CTA then reads nt float4s instead of making multiple passes\n    // over a complete row. The residual square is a quadratic in the branch\n    // inverse RMS; all moments and accumulation remain FP32.\n    __syncthreads();\n    for(int base=0;base<T;base+=groups) {\n      int t=base+g,tok=tok0+t;\n      bool active=t<T && tok<M;\n      float sb=0,sa=0,sab=0,sbb=0;\n      if(active) for(int r=lane;r<R;r+=threads) {\n        int j=blockIdx.x*R+r;\n        if(gl.phase==2) {\n          float p=j<4096?raw[(size_t)tok*N+j]:0.f;\n          sb+=p*p;\n        } else {\n          for(;j<8192;j+=nt*R) {\n            float p=j<N?raw[(size_t)tok*N+j]:0.f;\n            if(gl.phase==3)p*=gl.post[j];\n            float a=gl.alpha[j]*gl.stream[(size_t)tok*8192+j],b=gl.beta[j]*p;\n            sb+=p*p;sa+=a*a;sab+=a*b;sbb+=b*b;\n          }\n        }\n      }\n      sb=group_sum(sb,threads,tmp);sa=group_sum(sa,threads,tmp);\n      sab=group_sum(sab,threads,tmp);sbb=group_sum(sbb,threads,tmp);\n      if(active && lane==0)\n        ((float4*)gl.moments)[(size_t)tok*nt+blockIdx.x]=make_float4(sb,sa,sab,sbb);\n    }\n  }\n  // Every thread publishes its own output writes before any thread announces\n  // tile completion. The last acq_rel RMW acquires all earlier publications.\n  __threadfence();\n  __syncthreads();\n  for(int t=threadIdx.x;t<T;t+=THREADS) {\n    if(tok0+t<M) {\n      cuda::atomic_ref<unsigned,cuda::thread_scope_device> a(gl.ready[tok0+t]);\n      done[t]=(a.fetch_add(1,cuda::memory_order_acq_rel)==(unsigned)(nt-1));\n    } else done[t]=0;\n  }\n  __syncthreads();\n  for(int base=0;base<T;base+=groups) {\n    const int t=base+g, tok=tok0+t;\n    const bool active=t<T && tok<M && done[t];\n    float sum=0;\n    float sa=0,sab=0,sbb=0;\n    if(gl.moments) {\n      if(active) for(int j=lane;j<nt;j+=threads) {\n        const volatile float* p=gl.moments+((size_t)tok*nt+j)*4;\n        sum+=p[0];sa+=p[1];sab+=p[2];sbb+=p[3];\n      }\n      sum=group_sum(sum,threads,tmp);sa=group_sum(sa,threads,tmp);\n      sab=group_sum(sab,threads,tmp);sbb=group_sum(sbb,threads,tmp);\n    }\n    if(gl.phase==2) {\n      // Pool outputs and hi/lo router logits stay FP32 through the epilogue.\n      if(active && !gl.moments) for(int j=lane;j<4096;j+=threads) {\n        float p=((volatile float*)raw)[(size_t)tok*N+j]; sum+=p*p;\n      }\n      float inv=rsqrtf((gl.moments?sum:group_sum(sum,threads,tmp))/4096.f+1.e-5f);\n      if(active) for(int j=lane;j<4096;j+=threads) {\n        float p=((volatile float*)raw)[(size_t)tok*N+j];\n        gl.norm[(size_t)tok*4096+j]=__float2bfloat16_rn((p*inv)*gl.pre[j]);\n      }\n      // Deterministic top eight, using the FP32 score and expert-index tie key.\n      unsigned long long key[8]={};\n#pragma unroll\n      for(int i=0;i<8;++i) {\n        int e=lane+i*threads;\n        if(active && e<256) {\n          float logit=((volatile float*)raw)[(size_t)tok*N+4096+e]\n                     +((volatile float*)raw)[(size_t)tok*N+4352+e];\n          float score=1.f/(1.f+__expf(-logit*.1767766952966369f));\n          int bits=__float_as_int(score+gl.bias[e]);\n          unsigned ordered=bits<0?(unsigned)~bits:((unsigned)bits^0x80000000u);\n          key[i]=((unsigned long long)ordered<<32)|(unsigned)(255-e);\n        }\n      }\n      float picked[8],total=0;\n#pragma unroll\n      for(int slot=0;slot<8;++slot) {\n        unsigned long long best=0;\n#pragma unroll\n        for(int i=0;i<8;++i) best=best>key[i]?best:key[i];\n        best=group_max(best,threads,keys_tmp);\n#pragma unroll\n        for(int i=0;i<8;++i) if(key[i]==best) key[i]=0;\n        if(active && lane==0) {\n          int e=255-(unsigned)best;\n          float logit=((volatile float*)raw)[(size_t)tok*N+4096+e]\n                     +((volatile float*)raw)[(size_t)tok*N+4352+e];\n          picked[slot]=1.f/(1.f+__expf(-logit*.1767766952966369f));\n          total+=picked[slot];\n          gl.ids[tok*8+slot]=e;\n          int local=(e-*gl.start+256)%256;\n          gl.local_ids[tok*8+slot]=local<32?local:-1;\n        }\n      }\n      if(active && lane==0) {\n#pragma unroll\n        for(int slot=0;slot<8;++slot)\n          gl.weights[tok*8+slot]=picked[slot]/(total+1.e-15f);\n      }\n    } else {\n      if(active && !gl.moments) for(int j=lane;j<8192;j+=threads) {\n        float p=j<N?((volatile float*)raw)[(size_t)tok*N+j]:0.f;\n        if(gl.phase==3) p*=gl.post[j];\n        sum+=p*p;\n      }\n      float inv=rsqrtf((gl.moments?sum:group_sum(sum,threads,tmp))/8192.f+1.e-8f);\n      float out_inv=gl.moments?rsqrtf(fmaxf(0.f,sa+2.f*inv*sab+inv*inv*sbb)/8192.f+1.e-5f):0.f;\n      sum=0;\n      if(active) for(int j=lane;j<8192;j+=threads) {\n        float p=j<N?((volatile float*)raw)[(size_t)tok*N+j]:0.f;\n        if(gl.phase==3) p*=gl.post[j];\n        float s=gl.alpha[j]*gl.stream[(size_t)tok*8192+j]+gl.beta[j]*(p*inv);\n        gl.stream_out[(size_t)tok*8192+j]=s;\n        sum+=s*s;\n        if(gl.moments && gl.norm)\n          gl.norm[(size_t)tok*8192+j]=__float2bfloat16_rn((s*out_inv)*gl.pre[j]);\n      }\n      inv=gl.moments?out_inv:rsqrtf(group_sum(sum,threads,tmp)/8192.f+1.e-5f);\n      if(active && gl.norm && !gl.moments) for(int j=lane;j<8192;j+=threads) {\n        float s=gl.stream_out[(size_t)tok*8192+j];\n        gl.norm[(size_t)tok*8192+j]=__float2bfloat16_rn((s*inv)*gl.pre[j]);\n      }\n    }\n    __syncthreads();\n    if(active && lane==0) {\n      cuda::atomic_ref<unsigned,cuda::thread_scope_device> a(gl.ready[tok]);\n      a.store(0,cuda::memory_order_relaxed);\n    }\n    __syncthreads();\n  }\n}\n}  // namespace stream_proj\n\nnamespace stream_proj {\n// No spinning, no extra launch: every split publishes its own disjoint tile;\n// the last CTA reduces in FP32. The private counter returns to zero before exit.\n// scratch/counters belong to this call site and cannot be shared concurrently.\ntemplate<int R, int T, int THREADS>\n__device__ __forceinline__ bool finish(float* partial, unsigned* counters, void* y,\n    int M, int N, int splits, int nt, int mt, bool fp32) {\n  __shared__ int last;\n  const int tile = blockIdx.z * nt + blockIdx.x;\n  __threadfence();\n  __syncthreads();\n  if (threadIdx.x == 0) last = atomicAdd(counters + tile, 1u) == (unsigned)(splits - 1);\n  __syncthreads();\n  if (last) {\n    for (int e = threadIdx.x; e < R * T; e += THREADS) {\n      const int r = e % R, t = e / R;\n      if (blockIdx.x * R + r < N && blockIdx.z * T + t < M) {\n        float sum = 0.f;\n        for (int s = 0; s < splits; ++s)\n          sum += ((volatile float*)partial)[((size_t)tile * splits + s) * R * T + e];\n        output(y, (size_t)(blockIdx.z * T + t) * N + blockIdx.x * R + r, sum, fp32);\n      }\n    }\n    __syncthreads();\n    if (threadIdx.x == 0) atomicExch(counters + tile, 0u);\n  }\n  return last;\n}\n\n// A single lane-preserving shuffle creates a true dependency on every\n// outstanding weight vector. The returned FP32 value is exactly +0, but\n// ptxas cannot sink the vector loads to the first FMA that uses each one.\n// Unlike a bare barrier, this survives instruction scheduling in SASS.\ntemplate<int U, int ROWS>\n__device__ __forceinline__ float ring_dependency(uint4 (&w)[ROWS][U]) {\n  unsigned hash = 0;\n#pragma unroll\n  for (int r = 0; r < ROWS; ++r)\n#pragma unroll\n    for (int u = 0; u < U; ++u) hash ^= w[r][u].x;\n  unsigned same;\n  asm volatile("shfl.sync.idx.b32 %0, %1, %2, 31, -1;"\n      : "=r"(same) : "r"(hash), "r"(threadIdx.x & 31));\n  return __uint_as_float(same ^ hash);\n}\n\ntemplate<int U, int WARPS, int ROWS>\n__global__ __launch_bounds__(WARPS * 32,1)\nvoid scalar(const B* __restrict__ w, const B* __restrict__ x, void* y,\n            float* partial, unsigned* counters, int M, int N, int K, int splits, bool fp32, Glue gl) {\n  constexpr int R = WARPS * ROWS;\n  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;\n  const int row = blockIdx.x * R + warp * ROWS;\n  const int ks = K / splits, begin = blockIdx.y * ks, end = begin + ks;\n  const int tok = blockIdx.z;\n  __shared__ float input_inv[1];\n  if(gl.prologue) init_input<1,WARPS*32>(gl,x,M,K,input_inv);\n  float a[ROWS][8] = {};\n  for (int base = begin; base < end; base += U * 256) {\n    // Issue the entire register ring before consuming it. U=8 is 4 KiB of\n    // outstanding weight requests per warp, before adding another resident CTA.\n    uint4 xv[U], wv[ROWS][U];\n#pragma unroll\n    for (int u = 0; u < U; ++u) {\n      const int k = base + u * 256 + lane * 8;\n      xv[u] = k < end ? input16(gl,x,tok,k,K,gl.prologue?input_inv[0]:1.f) : uint4{};\n#pragma unroll\n      for (int r = 0; r < ROWS; ++r)\n        wv[r][u] = k < end && row + r < N ? load16(w + (size_t)(row + r) * K + k) : uint4{};\n    }\n    const float dependency = ring_dependency(wv);\n#pragma unroll\n    for (int r = 0; r < ROWS; ++r)\n#pragma unroll\n      for (int j = 0; j < 8; ++j) a[r][j] += dependency;\n#pragma unroll\n    for (int u = 0; u < U; ++u) {\n      const B* xx = reinterpret_cast<const B*>(&xv[u]);\n#pragma unroll\n      for (int r = 0; r < ROWS; ++r) {\n        const B* ww = reinterpret_cast<const B*>(&wv[r][u]);\n#pragma unroll\n        for (int j = 0; j < 8; ++j)\n          a[r][j] = fmaf(__bfloat162float(ww[j]), __bfloat162float(xx[j]), a[r][j]);\n      }\n    }\n  }\n  const int nt = (N + R - 1) / R, tile = tok * nt + blockIdx.x;\n#pragma unroll\n  for (int r = 0; r < ROWS; ++r) {\n    float v = 0.f;\n#pragma unroll\n    for (int j = 0; j < 8; ++j) v += a[r][j];\n#pragma unroll\n    for (int d = 16; d; d >>= 1) v += __shfl_down_sync(0xffffffff, v, d);\n    if (lane == 0 && row + r < N) {\n      if (splits == 1) output(y, (size_t)tok * N + row + r, v, fp32);\n      else partial[((size_t)tile * splits + blockIdx.y) * R + warp * ROWS + r] = v;\n    }\n  }\n  bool complete=splits==1 || finish<R,1,WARPS*32>(partial,counters,y,M,N,splits,nt,M,fp32);\n  if(complete) publish<R,1,WARPS*32>(gl,(float*)y,M,N,nt);\n}\n\ntemplate<int R, int T, int BK, int STAGES, int WARPS>\n__global__ __launch_bounds__(WARPS * 32,1)\nvoid tensor(const B* __restrict__ w, const B* __restrict__ x, void* y,\n            float* partial, unsigned* counters, int M, int N, int K, int splits, bool fp32, Glue gl) {\n  constexpr int STRIDE = BK + 8, STAGE = (R + T) * STRIDE;\n  constexpr int CELLS = (R / 16) * (T / 8) / WARPS;\n  extern __shared__ __align__(16) B smem[];\n  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;\n  const int g = lane >> 2, l = lane & 3;\n  const int row0 = blockIdx.x * R, tok0 = blockIdx.z * T;\n  const int ks = K / splits, begin = blockIdx.y * ks, end = begin + ks;\n  const int tiles = (ks + BK - 1) / BK;\n  __shared__ float input_inv[T];\n  if(gl.prologue) init_input<T,WARPS*32>(gl,x,M,K,input_inv);\n  auto load = [&](int tile) {\n    B* s = smem + (tile % STAGES) * STAGE;\n    const int kb = begin + tile * BK;\n    // 16-byte cp.async; padded shared rows avoid bank conflicts for the MMA\n    // fragment\'s eight independent rows (four lanes per row).\n    for (int v = tid; v < (R + T) * BK / 8; v += WARPS * 32) {\n      const int r = v / (BK / 8), c = (v % (BK / 8)) * 8;\n      const bool is_w = r < R;\n      const int global_r = is_w ? row0 + r : tok0 + r - R;\n      const bool valid = kb + c < end && global_r < (is_w ? N : M);\n      const B* src = valid ? (is_w ? w : x) + (size_t)global_r * K + kb + c : x;\n      if(!is_w && gl.prologue) {\n        uint4 v=valid?input16(gl,x,global_r,kb+c,K,input_inv[r-R]):uint4{};\n        *(uint4*)(s+r*STRIDE+c)=v;\n      } else cp16(s + r * STRIDE + c, src, valid);\n    }\n    asm volatile("cp.async.commit_group;" ::: "memory");\n  };\n  float a[CELLS][4] = {};\n#pragma unroll\n  for (int st = 0; st < STAGES - 1; ++st) load(st);\n  for (int tile = 0; tile < tiles; ++tile) {\n    load(tile + STAGES - 1);\n    asm volatile("cp.async.wait_group %0;" :: "n"(STAGES - 1) : "memory");\n    __syncthreads();\n    B* s = smem + (tile % STAGES) * STAGE;\n#pragma unroll\n    for (int k = 0; k < BK; k += 16) {\n#pragma unroll\n      for (int c = 0; c < CELLS; ++c) {\n        const int cell = warp + c * WARPS;\n        const int rn = (cell / (T / 8)) * 16, tm = (cell % (T / 8)) * 8;\n        const B* wa = s + (rn + g) * STRIDE + k + l * 2;\n        const B* xb = s + (R + tm + g) * STRIDE + k + l * 2;\n        unsigned a0 = *reinterpret_cast<const unsigned*>(wa);\n        unsigned a1 = *reinterpret_cast<const unsigned*>(wa + 8 * STRIDE);\n        unsigned a2 = *reinterpret_cast<const unsigned*>(wa + 8);\n        unsigned a3 = *reinterpret_cast<const unsigned*>(wa + 8 * STRIDE + 8);\n        unsigned b0 = *reinterpret_cast<const unsigned*>(xb);\n        unsigned b1 = *reinterpret_cast<const unsigned*>(xb + 8);\n        mma(a[c], a0, a1, a2, a3, b0, b1);\n      }\n    }\n    __syncthreads();\n  }\n  asm volatile("cp.async.wait_group 0;" ::: "memory");\n  const int nt = (N + R - 1) / R, mt = (M + T - 1) / T;\n  const int tile_id = blockIdx.z * nt + blockIdx.x;\n#pragma unroll\n  for (int c = 0; c < CELLS; ++c) {\n    const int cell = warp + c * WARPS;\n    const int rn = (cell / (T / 8)) * 16 + g, tm = (cell % (T / 8)) * 8 + l * 2;\n#pragma unroll\n    for (int i = 0; i < 4; ++i) {\n      const int r = rn + (i / 2) * 8, t = tm + (i & 1);\n      if (row0 + r < N && tok0 + t < M) {\n        if (splits == 1) output(y, (size_t)(tok0 + t) * N + row0 + r, a[c][i], fp32);\n        else partial[((size_t)tile_id * splits + blockIdx.y) * R * T + t * R + r] = a[c][i];\n      }\n    }\n  }\n  bool complete=splits==1 || finish<R,T,WARPS*32>(partial,counters,y,M,N,splits,nt,mt,fp32);\n  if(complete) publish<R,T,WARPS*32>(gl,(float*)y,M,N,nt);\n}\n\ntemplate<int R, int T, int BK, int STAGES, int WARPS>\n__global__ __launch_bounds__(WARPS * 32,1)\nvoid tensor_vector(const B* __restrict__ w, const B* __restrict__ x, void* y,\n            float* partial, unsigned* counters, int M, int N, int K, int splits, bool fp32, Glue gl) {\n  constexpr int STRIDE = BK + 32, STAGE = (R + T) * STRIDE;\n  constexpr int CELLS = (R / 16) * (T / 8) / WARPS;\n  extern __shared__ __align__(16) B smem[];\n  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;\n  const int g = lane >> 2, l = lane & 3;\n  const int row0 = blockIdx.x * R, tok0 = blockIdx.z * T;\n  const int ks = K / splits, begin = blockIdx.y * ks, end = begin + ks;\n  const int tiles = (ks + BK - 1) / BK;\n  __shared__ float input_inv[T];\n  if(gl.prologue) init_input<T,WARPS*32>(gl,x,M,K,input_inv);\n  auto load = [&](int tile) {\n    B* s = smem + (tile % STAGES) * STAGE;\n    const int kb = begin + tile * BK;\n    // 16-byte cp.async; padded shared rows avoid bank conflicts for the MMA\n    // fragment\'s eight independent rows (four lanes per row).\n    for (int v = tid; v < (R + T) * BK / 8; v += WARPS * 32) {\n      const int r = v / (BK / 8), c = (v % (BK / 8)) * 8;\n      const bool is_w = r < R;\n      const int global_r = is_w ? row0 + r : tok0 + r - R;\n      const bool valid = kb + c < end && global_r < (is_w ? N : M);\n      const B* src = valid ? (is_w ? w : x) + (size_t)global_r * K + kb + c : x;\n      if(!is_w && gl.prologue) {\n        uint4 v=valid?input16(gl,x,global_r,kb+c,K,input_inv[r-R]):uint4{};\n        *(uint4*)(s+r*STRIDE+c)=v;\n      } else cp16(s + r * STRIDE + c, src, valid);\n    }\n    asm volatile("cp.async.commit_group;" ::: "memory");\n  };\n  float a[CELLS][4] = {};\n#pragma unroll\n  for (int st = 0; st < STAGES - 1; ++st) load(st);\n  for (int tile = 0; tile < tiles; ++tile) {\n    load(tile + STAGES - 1);\n    asm volatile("cp.async.wait_group %0;" :: "n"(STAGES - 1) : "memory");\n    __syncthreads();\n    B* s = smem + (tile % STAGES) * STAGE;\n    // Two MMA steps consume one 16-byte shared load per operand row. The\n    // builder uses BK+32 shared rows: each eight-lane transaction visits the\n    // first and second rows in opposite halves of the bank array.\n    // The K permutation (eight adjacent BF16s per lane) is identical for W\n    // and X, so this changes summation order, not values or precision.\n#pragma unroll\n    for (int k = 0; k < BK; k += 32) {\n#pragma unroll\n      for (int c = 0; c < CELLS; ++c) {\n        const int cell = warp + c * WARPS;\n        const int rn = (cell / (T / 8)) * 16, tm = (cell % (T / 8)) * 8;\n        const B* wa = s + (rn + g) * STRIDE + k + l * 8;\n        const B* xb = s + (R + tm + g) * STRIDE + k + l * 8;\n        uint4 a0 = *reinterpret_cast<const uint4*>(wa);\n        uint4 a1 = *reinterpret_cast<const uint4*>(wa + 8 * STRIDE);\n        uint4 b = *reinterpret_cast<const uint4*>(xb);\n        mma(a[c], a0.x, a1.x, a0.y, a1.y, b.x, b.y);\n        mma(a[c], a0.z, a1.z, a0.w, a1.w, b.z, b.w);\n      }\n    }\n    __syncthreads();\n  }\n  asm volatile("cp.async.wait_group 0;" ::: "memory");\n  const int nt = (N + R - 1) / R, mt = (M + T - 1) / T;\n  const int tile_id = blockIdx.z * nt + blockIdx.x;\n#pragma unroll\n  for (int c = 0; c < CELLS; ++c) {\n    const int cell = warp + c * WARPS;\n    const int rn = (cell / (T / 8)) * 16 + g, tm = (cell % (T / 8)) * 8 + l * 2;\n#pragma unroll\n    for (int i = 0; i < 4; ++i) {\n      const int r = rn + (i / 2) * 8, t = tm + (i & 1);\n      if (row0 + r < N && tok0 + t < M) {\n        if (splits == 1) output(y, (size_t)(tok0 + t) * N + row0 + r, a[c][i], fp32);\n        else partial[((size_t)tile_id * splits + blockIdx.y) * R * T + t * R + r] = a[c][i];\n      }\n    }\n  }\n  bool complete=splits==1 || finish<R,T,WARPS*32>(partial,counters,y,M,N,splits,nt,mt,fp32);\n  if(complete) publish<R,T,WARPS*32>(gl,(float*)y,M,N,nt);\n}\n\ntemplate<int U, int W, int R>\nint launch_scalar(const B* w, const B* x, void* y, float* p, unsigned* c,\n                  int M, int N, int K, int splits, bool fp32, Glue gl, cudaStream_t stream) {\n  scalar<U,W,R><<<dim3((N + W*R - 1)/(W*R), splits, M), W*32, 0, stream>>>\n      (w,x,y,p,c,M,N,K,splits,fp32,gl);\n  return (int)cudaGetLastError();\n}\ntemplate<int R, int T, int BK, int S, int W>\nint launch_tensor(const B* w, const B* x, void* y, float* p, unsigned* c,\n                  int M, int N, int K, int splits, bool fp32, Glue gl, cudaStream_t stream) {\n#ifdef SP_LOCAL_SHARED_CAP\n  // Local Blackwell has 99 KiB/CTA; H200 has 227 KiB. Execute the identical\n  // fragments/reducer/glue with fewer stages locally instead of skipping it.\n  constexpr int fit = SP_LOCAL_SHARED_CAP / ((R + T) * (BK + 8) * sizeof(B));\n  constexpr int stages = S < fit ? S : fit;\n#else\n  constexpr int stages = S;\n#endif\n  static_assert(stages > 0);\n  constexpr int bytes = stages * (R + T) * (BK + 8) * sizeof(B);\n  auto kernel = tensor<R,T,BK,stages,W>;\n  auto e = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);\n  if (e != cudaSuccess) return (int)e;\n  kernel<<<dim3((N + R - 1)/R, splits, (M + T - 1)/T), W*32, bytes, stream>>>\n      (w,x,y,p,c,M,N,K,splits,fp32,gl);\n  return (int)cudaGetLastError();\n}\ntemplate<int R, int T, int BK, int S, int W>\nint launch_tensor_vector(const B* w, const B* x, void* y, float* p, unsigned* c,\n                  int M, int N, int K, int splits, bool fp32, Glue gl, cudaStream_t stream) {\n#ifdef SP_LOCAL_SHARED_CAP\n  // Local Blackwell has 99 KiB/CTA; H200 has 227 KiB. Execute the identical\n  // fragments/reducer/glue with fewer stages locally instead of skipping it.\n  constexpr int fit = SP_LOCAL_SHARED_CAP / ((R + T) * (BK + 32) * sizeof(B));\n  constexpr int stages = S < fit ? S : fit;\n#else\n  constexpr int stages = S;\n#endif\n  static_assert(stages > 0);\n  constexpr int bytes = stages * (R + T) * (BK + 32) * sizeof(B);\n  auto kernel = tensor_vector<R,T,BK,stages,W>;\n  auto e = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);\n  if (e != cudaSuccess) return (int)e;\n  kernel<<<dim3((N + R - 1)/R, splits, (M + T - 1)/T), W*32, bytes, stream>>>\n      (w,x,y,p,c,M,N,K,splits,fp32,gl);\n  return (int)cudaGetLastError();\n}\n} // namespace stream_proj\n\n// Same-source read control: isolate request throughput from arithmetic and\n// output/reduction costs, with the identical cold-L2 graph/event protocol.\n__global__ void sp_read_kernel(const uint4* w, unsigned* sink, size_t vectors) {\n  const size_t first = (size_t)blockIdx.x * blockDim.x + threadIdx.x;\n  const size_t step = (size_t)gridDim.x * blockDim.x;\n  unsigned sum = 0;\n  for (size_t b = first; b < vectors; b += step * 8) {\n    uint4 v[8];\n#pragma unroll\n    for (int u = 0; u < 8; ++u) {\n      const size_t i = b + u * step;\n      v[u] = i < vectors ? stream_proj::load16((const stream_proj::B*)(w + i)) : uint4{};\n    }\n#pragma unroll\n    for (int u = 0; u < 8; ++u) sum ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;\n  }\n  // Every vector contributes; each thread has a private sink element.\n  sink[(size_t)blockIdx.x * blockDim.x + threadIdx.x] = sum;\n}\nextern "C" int sp_read(const void* w, void* sink, size_t bytes, int sms, void* stream) {\n  sp_read_kernel<<<sms, 512, 0, (cudaStream_t)stream>>>((const uint4*)w, (unsigned*)sink, bytes/16);\n  return (int)cudaGetLastError();\n}\n\nextern "C" int sp_launch(const void* w, const void* x, void* y, void* p, void* c,\n                         int M, int N, int K, int config, int splits, int fp32, const void* glue, void* stream) {\n  using namespace stream_proj;\n  Glue gl=glue?*(const Glue*)glue:Glue{};\n  if (K % (splits * 8)) return (int)cudaErrorInvalidValue;\n#define ARG (const B*)w,(const B*)x,y,(float*)p,(unsigned*)c,M,N,K,splits,(bool)fp32,gl,(cudaStream_t)stream\n  switch(config) {\n    case 0: return launch_scalar<4,4,1>(ARG);\n    case 1: return launch_scalar<8,4,1>(ARG);\n    case 2: return launch_scalar<16,4,1>(ARG);\n    case 3: return launch_scalar<8,8,1>(ARG);\n    case 4: return launch_scalar<4,4,2>(ARG);\n    case 5: return launch_scalar<8,4,2>(ARG);\n    case 10: return launch_tensor<32,16,128,3,4>(ARG);\n    case 11: return launch_tensor<32,16,256,3,4>(ARG);\n    case 12: return launch_tensor<64,8,256,3,4>(ARG);\n    case 13: return launch_tensor<64,16,256,3,8>(ARG);\n    case 14: return launch_tensor<32,32,256,3,8>(ARG);\n    case 15: return launch_tensor<32,16,256,4,4>(ARG);\n    case 16: return launch_tensor<64,16,128,4,8>(ARG);\n    case 20: return launch_tensor<32,64,128,3,8>(ARG);\n    case 21: return launch_tensor<64,64,128,2,8>(ARG);\n    case 22: return launch_tensor<32,128,128,3,8>(ARG);\n    case 23: return launch_tensor<64,128,128,2,8>(ARG);\n    case 24: return launch_tensor<32,64,256,3,8>(ARG);\n    case 25: return launch_tensor<32,128,256,2,8>(ARG);\n    case 110: return launch_tensor_vector<32,16,128,3,4>(ARG);\n    case 111: return launch_tensor_vector<32,16,256,3,4>(ARG);\n    case 112: return launch_tensor_vector<64,8,256,3,4>(ARG);\n    case 113: return launch_tensor_vector<64,16,256,3,8>(ARG);\n    case 114: return launch_tensor_vector<32,32,256,3,8>(ARG);\n    case 115: return launch_tensor_vector<32,16,256,4,4>(ARG);\n    case 116: return launch_tensor_vector<64,16,128,4,8>(ARG);\n    case 120: return launch_tensor_vector<32,64,128,3,8>(ARG);\n    case 121: return launch_tensor_vector<64,64,128,2,8>(ARG);\n    case 122: return launch_tensor_vector<32,128,128,3,8>(ARG);\n    case 123: return launch_tensor_vector<64,128,128,2,8>(ARG);\n    case 124: return launch_tensor_vector<32,64,256,3,8>(ARG);\n    case 125: return launch_tensor_vector<32,128,256,2,8>(ARG);\n    default: return (int)cudaErrorInvalidValue;\n  }\n#undef ARG\n}\n'

import ctypes as _sp_ct
import hashlib as _sp_hash
import os as _sp_os

import torch
from torch.utils.cpp_extension import load_inline as _sp_load_inline


# id -> weight rows, token rows, K stage, stages, warps, scalar prefetch depth
SP_CONFIGS = {
    0: (4, 1, 0, 0, 4, 4), 1: (4, 1, 0, 0, 4, 8),
    2: (4, 1, 0, 0, 4, 16), 3: (8, 1, 0, 0, 8, 8),
    4: (8, 1, 0, 0, 4, 4), 5: (8, 1, 0, 0, 4, 8),
    10: (32, 16, 128, 3, 4, 0), 11: (32, 16, 256, 3, 4, 0),
    12: (64, 8, 256, 3, 4, 0), 13: (64, 16, 256, 3, 8, 0),
    14: (32, 32, 256, 3, 8, 0), 15: (32, 16, 256, 4, 4, 0),
    16: (64, 16, 128, 4, 8, 0),
    20: (32, 64, 128, 3, 8, 0), 21: (64, 64, 128, 2, 8, 0),
    22: (32, 128, 128, 3, 8, 0), 23: (64, 128, 128, 2, 8, 0),
    24: (32, 64, 256, 3, 8, 0), 25: (32, 128, 256, 2, 8, 0),
}
_SP_LIB = None


def sp_library():
    global _SP_LIB
    if _SP_LIB is None:
        arch = "9.0a" if torch.cuda.get_device_capability()[0] == 9 else "12.0"
        old = _sp_os.environ.get("TORCH_CUDA_ARCH_LIST")
        _sp_os.environ["TORCH_CUDA_ARCH_LIST"] = arch
        try:
            path = _sp_load_inline(
                name="muse_stream_proj_" + _sp_hash.sha256(SP_CUDA.encode()).hexdigest()[:16],
                cpp_sources="", cuda_sources=SP_CUDA, functions=None,
                is_python_module=False, no_implicit_headers=True,
                extra_cuda_cflags=["-O3", "-std=c++17", "-lineinfo", "--ptxas-options=-v"]
                    + (["-DSP_LOCAL_SHARED_CAP=98304"] if arch == "12.0" else []),
                verbose=False,
            )
        finally:
            if old is None:
                _sp_os.environ.pop("TORCH_CUDA_ARCH_LIST", None)
            else:
                _sp_os.environ["TORCH_CUDA_ARCH_LIST"] = old
        _SP_LIB = _sp_ct.CDLL(path)
        _SP_LIB.sp_launch.argtypes = [_sp_ct.c_void_p] * 5 + [_sp_ct.c_int] * 6 + [_sp_ct.c_void_p] * 2
        _SP_LIB.sp_launch.restype = _sp_ct.c_int
        _SP_LIB.sp_read.argtypes = [_sp_ct.c_void_p] * 2 + [_sp_ct.c_size_t, _sp_ct.c_int, _sp_ct.c_void_p]
        _SP_LIB.sp_read.restype = _sp_ct.c_int
    return _SP_LIB


class StreamingProjection:
    """One-launch projection with private, replay-safe last-CTA scratch.

    Construct outside capture (prepare). Instances must not run concurrently;
    callers on separate streams use separate instances. Weights are BF16 views
    of the original bank. FP32 pool/router keeps all hi and lo rows separately,
    matching the decode lane's boundary consumer ABI.
    """

    def __init__(self, w, m, config, splits=1, fp32=False):
        if w.dtype != torch.bfloat16 or not w.is_contiguous():
            raise ValueError("expected contiguous original BF16 weights")
        if w.shape[1] % (8 * splits):
            raise ValueError("split must preserve an eight-BF16 vector")
        self.lib, self.w, self.m = sp_library(), w, m
        self.config, self.splits, self.fp32 = config, splits, fp32
        r, t, *_ = SP_CONFIGS[config]
        n, k = w.shape
        self.tiles = ((n + r - 1) // r) * ((m + t - 1) // t)
        self.ctas = self.tiles * splits
        self.partial = torch.empty((self.tiles, splits, r * t), dtype=torch.float32, device=w.device)
        self.counters = torch.zeros(self.tiles, dtype=torch.int32, device=w.device)
        self.out = torch.empty((m, n), dtype=torch.float32 if fp32 else torch.bfloat16, device=w.device)

    def __call__(self, x):
        if x.shape != (self.m, self.w.shape[1]) or x.dtype != torch.bfloat16 or not x.is_contiguous():
            raise ValueError("input shape, precision or stride changed")
        e = self.lib.sp_launch(
            self.w.data_ptr(), x.data_ptr(), self.out.data_ptr(),
            self.partial.data_ptr(), self.counters.data_ptr(), self.m,
            self.w.shape[0], self.w.shape[1], self.config, self.splits, int(self.fp32),
            None, torch.cuda.current_stream(x.device).cuda_stream,
        )
        if e:
            raise RuntimeError(f"streaming projection launch failed CUDA={e} config={self.config}")
        return self.out


import statistics as _sp_stats

import time as _sp_time

def sp_inputs(m, k, seed, device):
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn((m, k), generator=gen, device=device, dtype=torch.bfloat16)
    x[:, :max(1, k // 100)] *= 30
    # Every eighth row is an exact column probe, including M=1.
    rows = torch.arange(0, m if m > 1 or seed % 2 else 0, 8, device=device)
    cols = (rows * 977 + seed) % k
    x[rows] = 0
    x[rows, cols] = 32
    return x

def sp_check(out, reference, x, router=None, pool_rows=4096):
    """FP32 gate, per-row gate and exact columns; router checked at its full precision."""
    delta = (out.float() - reference).norm(dim=-1)
    denom = reference.norm(dim=-1).clamp_min(reference.norm(dim=-1).mean() * .1).clamp_min(1e-20)
    rel = float((out.float() - reference).norm() / reference.norm().clamp_min(1e-20))
    tol = 1e-4 if out.dtype == torch.float32 else 1e-2
    if not bool(torch.isfinite(out).all()) or rel > tol or float((delta / denom).max()) > 5 * tol:
        raise RuntimeError(f"FP32 gate failed L2={rel:.3e} row={float((delta/denom).max()):.3e}")
    probe = torch.nonzero(torch.count_nonzero(x, dim=-1) == 1).flatten()
    if probe.numel():
        p = float((delta[probe] / reference[probe].norm(dim=-1).clamp_min(1e-20)).max())
        if p > 1e-3:
            raise RuntimeError(f"column probe failed {p:.3e}")
    if router is not None:
        count = router.shape[0]
        actual = out[:, pool_rows:pool_rows+count] + out[:, pool_rows+count:pool_rows+2*count]
        expected = x.float() @ router.t()
        err = float((actual - expected).norm() / expected.norm().clamp_min(1e-20))
        rowerr = (actual - expected).norm(dim=-1) / expected.norm(dim=-1).clamp_min(expected.norm(dim=-1).mean() * .1)
        if err > 1e-4 or float(rowerr.max()) > 5e-4:
            raise RuntimeError(f"router hi+lo failed {err:.3e}/{float(rowerr.max()):.3e}")
    return rel

import triton
import triton.language as tl


SP_CONFIGS.update({100+k:v for k,v in list(SP_CONFIGS.items()) if k>=10})
SP_CONFIGS.update({200+k:v for k,v in list(SP_CONFIGS.items())})


class _SgParams(_sp_ct.Structure):
    _fields_=[('phase',_sp_ct.c_int),('prologue',_sp_ct.c_int)]+[
        (name,_sp_ct.c_void_p) for name in ('stream','alpha','beta','pre','post','bias','input_pre',
                                          'stream_out','norm','ids','local_ids','weights','start','ready','moments')]


class FusedStreamingProjection(StreamingProjection):
    """One launch: projection, split-K completion and boundary/routing continuation.

    params contains FP32 alpha/beta and 1+norm vectors. Pool prologue accepts an
    FP32 stream and rounds its normalized activation to BF16 before every MMA.
    Returned tensors and private scratch have stable addresses across replays.
    """
    def __init__(self,w,m,config,splits,phase,params,prologue=False,normalize=True):
        super().__init__(w,m,config,splits,fp32=True)
        self.phase,self.params,self.prologue,self.normalize=phase,params,prologue,normalize
        dev=w.device
        self.ready=torch.zeros(m,device=dev,dtype=torch.int32)
        width=4096 if phase=='pool' else 8192
        self.norm=torch.empty((m,width),device=dev,dtype=torch.bfloat16) if normalize else None
        self.stream_out=torch.empty((m,8192),device=dev,dtype=torch.float32) if phase!='pool' else None
        self.ids=torch.empty((m,8),device=dev,dtype=torch.int32) if phase=='pool' else None
        self.local_ids=torch.empty_like(self.ids) if phase=='pool' else None
        self.weights=torch.empty((m,8),device=dev,dtype=torch.float32) if phase=='pool' else None
        self.moments=torch.empty((m,triton.cdiv(w.shape[0],SP_CONFIGS[config][0]),4),
                                 device=dev,dtype=torch.float32) if config>=200 else None

    def __call__(self,x,stream=None,start=None):
        dtype=torch.float32 if self.prologue else torch.bfloat16
        if x.shape!=(self.m,self.w.shape[1]) or x.dtype!=dtype or not x.is_contiguous():
            raise ValueError('fused input shape, precision or stride changed')
        p=self.params
        def ptr(x):return x.data_ptr() if x is not None else None
        phase={'o':1,'pool':2,'unpool':3}[self.phase]
        args=_SgParams(phase,int(self.prologue),ptr(stream),ptr(p.get('alpha')),ptr(p.get('beta')),
            ptr(p.get('pre')),ptr(p.get('post')),ptr(p.get('bias')),ptr(p.get('input_pre')),
            ptr(self.stream_out),ptr(self.norm),ptr(self.ids),ptr(self.local_ids),ptr(self.weights),
            ptr(start),ptr(self.ready),ptr(self.moments))
        e=self.lib.sp_launch(self.w.data_ptr(),x.data_ptr(),self.out.data_ptr(),self.partial.data_ptr(),
            self.counters.data_ptr(),self.m,self.w.shape[0],self.w.shape[1],self.config%200,self.splits,1,
            _sp_ct.byref(args),torch.cuda.current_stream(x.device).cuda_stream)
        if e:raise RuntimeError(f'fused projection launch CUDA={e} config={self.config}')
        return (self.norm,self.ids,self.weights,self.local_ids) if phase==2 else (self.norm,self.stream_out)


@triton.jit
def _sg_boundary(P,STREAM,ALPHA,BETA,PRE,POST,Y,SOUT,N:tl.constexpr,POST_NORM:tl.constexpr):
    t=tl.program_id(0);j=tl.arange(0,8192)
    p=tl.load(P+t*N+j,j<N,0).to(tl.float32)
    if POST_NORM:p*=tl.load(POST+j)
    p*=tl.rsqrt(tl.sum(p*p,0)/8192+1.e-8)
    s=tl.load(ALPHA+j)*tl.load(STREAM+t*8192+j)+tl.load(BETA+j)*p
    inv=tl.rsqrt(tl.sum(s*s,0)/8192+1.e-5)
    tl.store(SOUT+t*8192+j,s)
    tl.store(Y+t*8192+j,(s*inv*tl.load(PRE+j)).to(tl.bfloat16))


@triton.jit
def _sg_norm(S,W,Y):
    t=tl.program_id(0);j=tl.arange(0,8192)
    s=tl.load(S+t*8192+j)
    v=s*tl.rsqrt(tl.sum(s*s,0)/8192+1.e-5)*tl.load(W+j)
    tl.store(Y+t*8192+j,v.to(tl.bfloat16))


@triton.jit
def _sg_route(P,PRE,BIAS,START,X,IDS,WTS,LIDS):
    t=tl.program_id(0);d=tl.arange(0,4096);e=tl.arange(0,256)
    pool=tl.load(P+t*4608+d)
    norm=pool*tl.rsqrt(tl.sum(pool*pool,0)/4096+1.e-5)*tl.load(PRE+d)
    tl.store(X+t*4096+d,norm.to(tl.bfloat16))
    logits=tl.load(P+t*4608+4096+e)+tl.load(P+t*4608+4352+e)
    scores=tl.sigmoid(logits*.1767766952966369)
    biased=scores+tl.load(BIAS+e)
    bits=biased.to(tl.int32,bitcast=True)
    ordered=tl.where(bits<0,~bits,bits^(-2147483648)).to(tl.uint32)
    keys=(ordered.to(tl.uint64)<<32)|(255-e).to(tl.uint64)
    ids=255-(tl.topk(keys,8)&0xffffffff).to(tl.int32)
    picked=tl.gather(scores,ids,0);j=tl.arange(0,8)
    tl.store(IDS+t*8+j,ids)
    tl.store(WTS+t*8+j,picked/(tl.sum(picked,0)+1.e-15))
    local=(ids-tl.load(START)+256)%256
    tl.store(LIDS+t*8+j,tl.where(local<32,local,-1))


class CublasGlue:
    """Paired stock arithmetic: cuBLAS plus the existing boundary semantics."""
    def __init__(self,w,m,phase,params,prologue=False):
        self.w,self.m,self.phase,self.params,self.prologue=w,m,phase,params,prologue
        dev=w.device
        self.raw=torch.empty((m,w.shape[0]),device=dev,dtype=torch.float32 if phase=='pool' else torch.bfloat16)
        self.x=torch.empty((m,w.shape[1]),device=dev,dtype=torch.bfloat16) if prologue else None
        self.norm=torch.empty((m,4096 if phase=='pool' else 8192),device=dev,dtype=torch.bfloat16)
        self.stream_out=torch.empty((m,8192),device=dev,dtype=torch.float32)
        self.ids=torch.empty((m,8),device=dev,dtype=torch.int32)
        self.local_ids=torch.empty_like(self.ids)
        self.weights=torch.empty((m,8),device=dev,dtype=torch.float32)

    def __call__(self,x,stream=None,start=None):
        p=self.params
        if self.prologue:
            _sg_norm[(self.m,)](x,p['input_pre'],self.x,num_warps=8)
            x=self.x
        if self.phase=='pool':
            torch.mm(x,self.w.t(),out_dtype=torch.float32,out=self.raw)
            _sg_route[(self.m,)](self.raw,p['pre'],p['bias'],start,self.norm,self.ids,self.weights,
                                self.local_ids,num_warps=4)
            return self.norm,self.ids,self.weights,self.local_ids
        torch.mm(x,self.w.t(),out=self.raw)
        _sg_boundary[(self.m,)](self.raw,stream,p['alpha'],p['beta'],p['pre'],p.get('post',p['pre']),
            self.norm,self.stream_out,self.w.shape[0],self.phase=='unpool',num_warps=8)
        return self.norm,self.stream_out


class StreamingGlueControl(CublasGlue):
    """The identical streaming geometry with separate normalization/routing."""
    def __init__(self,w,m,phase,params,config,splits,prologue=False):
        super().__init__(w,m,phase,params,prologue)
        self.projection=StreamingProjection(w,m,config%200,splits,True)

    def __call__(self,x,stream=None,start=None):
        p=self.params
        if self.prologue:
            _sg_norm[(self.m,)](x,p['input_pre'],self.x,num_warps=8)
            x=self.x
        raw=self.projection(x)
        if self.phase=='pool':
            _sg_route[(self.m,)](raw,p['pre'],p['bias'],start,self.norm,self.ids,self.weights,
                                self.local_ids,num_warps=4)
            return self.norm,self.ids,self.weights,self.local_ids
        _sg_boundary[(self.m,)](raw,stream,p['alpha'],p['beta'],p['pre'],p.get('post',p['pre']),
            self.norm,self.stream_out,self.w.shape[0],self.phase=='unpool',num_warps=8)
        return self.norm,self.stream_out


def sg_params(weights):
    def coeff(name):
        g=weights[name].float()/.3
        beta=g.sigmoid();alpha=(torch.sigmoid(-g)*(1.+beta)).clamp_min(1.e-3).sqrt()
        return alpha.contiguous(),beta.contiguous()
    a,b=coeff('attn_residual_gate');c,d=coeff('ffn_residual_gate')
    pre=1.+weights['pre_ffn_norm'].float()
    return {'o':dict(alpha=a,beta=b,pre=pre),
            'pool':dict(pre=1.+weights['pre_expert_norm'].float(),bias=weights['router_bias'].float(),input_pre=pre),
            'unpool':dict(alpha=c,beta=d,pre=1.+weights['next_pre_attn_norm'].float(),post=1.+weights['post_ffn_norm'].float())}


def sg_norm(x,w,eps=1.e-5):
    x=x.float()
    return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+eps)*w


def sg_oracle(phase,x,stream,start,w,params,prologue=False):
    effective=sg_norm(x,params['input_pre']).bfloat16() if prologue else x
    raw=torch.mm(effective,w.t(),out_dtype=torch.float32)
    if phase=='qkv':return raw,effective,raw
    if phase=='pool':
        logits=raw[:,4096:4352]+raw[:,4352:4608]
        scores=(logits*.1767766952966369).sigmoid()
        # stable sorting breaks ties by increasing expert index, as the kernels do.
        ids=torch.argsort(scores+params['bias'],dim=-1,descending=True,stable=True)[:,:8].int()
        picked=scores.gather(1,ids.long());wts=picked/(picked.sum(-1,keepdim=True)+1.e-15)
        local=(ids-start+256)%256;local=torch.where(local<32,local,-1).int()
        out=(sg_norm(raw[:,:4096],params['pre']).bfloat16(),ids,wts,local)
    else:
        br=torch.nn.functional.pad(raw,(0,8192-w.shape[0]))
        if phase=='unpool':br=br*params['post']
        s=params['alpha']*stream+params['beta']*sg_norm(br,1.,1.e-8)
        out=(sg_norm(s,params['pre']).bfloat16(),s)
    return out,effective,raw


def sg_error(a,b,tol):
    a,b=a.float(),b.float()
    d=(a-b).norm(dim=-1);n=b.norm(dim=-1);den=n.clamp_min(n.mean()*.1).clamp_min(1.e-20)
    rel=float(d.norm()/n.norm().clamp_min(1.e-20));row=float((d/den).max())
    if not bool(torch.isfinite(a).all()) or rel>tol or row>5*tol:
        raise RuntimeError(f'gate failed L2={rel:.3e} row={row:.3e} tol={tol}')
    return rel


def sg_check(phase,out,ref,raw,rawref,effective,router=None):
    # Projection error is tested before the nonlinear epilogue, at 1e-4.
    if raw is not None:sp_check(raw,rawref,effective,router)
    if phase=='qkv':return sg_error(out,ref,1.e-2 if out.dtype==torch.bfloat16 else 1.e-4)
    error=sg_error(out[0],ref[0],1.e-2)
    if phase=='pool':
        if not torch.equal(out[1],ref[1]):raise RuntimeError('router expert IDs differ from FP32 oracle')
        if not torch.equal(out[3],ref[3]):raise RuntimeError('local expert IDs differ from FP32 oracle')
        sg_error(out[2],ref[2],1.e-4)
    else:sg_error(out[1],ref[1],1.e-4 if raw is not None else 1.e-2)
    return error


def sg_graph(fn):
    fn();torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):out=fn()
    graph.replay();torch.cuda.synchronize()
    return graph,out


def sg_pair(base,new,inputs,banks,flush,repeats):
    times=[[],[]];a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    for rep in range(repeats):
        for target,fresh in zip(inputs,banks[rep%len(banks)]):target.copy_(fresh)
        for side in ((0,1) if rep%2==0 else (1,0)):
            flush.add_(1);a.record();(base if side==0 else new).replay();b.record();b.synchronize()
            times[side].append(a.elapsed_time(b)*1000.)
    paired=[((times[1][i]/times[0][i])*(times[1][i+1]/times[0][i+1]))**.5 for i in range(0,repeats,2)]
    return _sp_stats.median(times[0]),_sp_stats.median(times[1]),_sp_stats.median(paired)


def sg_choices(m,k):
    if m==1:cfgs=(1,2,3,5,10,12,110,112)
    elif m<=32:cfgs=(10,11,12,13,14,15,16,110,111,112,113,114,115,116)
    else:cfgs=(13,14,20,21,22,23,24,25,113,114,120,121,122,123,124,125)
    # Interleave configurations and split counts so every family gets a trial
    # if a host's prepare leaves less than the entire budget.
    return [(mode+c,s) for s in (1,2,4) for c in cfgs for mode in (0,200) if k%(8*s)==0]


def sg_inputs(m,k,seed,device,prologue=False):
    x=sp_inputs(m,k,seed,device)
    if prologue:x=x.float()
    gen=torch.Generator(device=device).manual_seed(seed+450)
    stream=torch.randn((m,8192),device=device,generator=gen)
    if seed%2:stream[:,::97]*=20
    start=torch.tensor((seed*37)%256,device=device,dtype=torch.int32)
    return x,stream,start


def sg_trial(deadline,w,m,phase,params,router,flush,prologue=False):
    dev=w.device;banks=[sg_inputs(m,w.shape[1],seed+m,dev,prologue) for seed in (40,93)]
    inputs=tuple(t.clone() for t in banks[0]);x,stream,start=inputs
    refs=[sg_oracle(phase,*bank,w,params,prologue) for bank in banks]
    oldop=CublasGlue(w,m,phase,params,prologue) if phase!='qkv' else None
    oldfn=(lambda:oldop(x,stream,start)) if oldop else (lambda:torch.mm(x,w.t()))
    old,oldout=sg_graph(oldfn)
    trials=[];rejections=[]
    try:
        for fresh,(ref,effective,rawref) in zip(banks,refs):
            for a,b in zip(inputs,fresh):a.copy_(b)
            old.replay();sg_check(phase,oldout,ref,None,rawref,effective,router)
        coarse=_sp_time.monotonic()+.64*max(0,deadline-_sp_time.monotonic())
        for cfg,split in sg_choices(m,w.shape[1]):
            if _sp_time.monotonic()>coarse:break
            if phase=='qkv' and cfg>=200:continue
            graph=None
            try:
                if phase=='qkv':
                    op=StreamingProjection(w,m,cfg,split,fp32=True)
                    fn=lambda:op(x)
                else:
                    op=FusedStreamingProjection(w,m,cfg,split,phase,params,prologue)
                    fn=lambda:op(x,stream,start)
                graph,out=sg_graph(fn)
                err=0.
                for fresh,(ref,effective,rawref) in zip(banks,refs):
                    for a,b in zip(inputs,fresh):a.copy_(b)
                    graph.replay();err=max(err,sg_check(phase,out,ref,op.out,rawref,effective,router))
                    if bool(op.counters.any()) or (phase!='qkv' and bool(op.ready.any())):
                        raise RuntimeError('completion counter did not reset')
                b,c,ratio=sg_pair(old,graph,inputs,banks,flush,4)
                trials.append((ratio,cfg,split,b,c,err))
            except Exception as exc:rejections.append(f'{cfg}/{split}:{type(exc).__name__}:{str(exc)[:85]}')
            finally:
                if graph is not None:graph.reset()
        finals=[]
        for _,cfg,split,*_ in sorted(trials)[:2]:
            if _sp_time.monotonic()>deadline-.25:break
            op=(StreamingProjection(w,m,cfg,split,True) if phase=='qkv' else
                FusedStreamingProjection(w,m,cfg,split,phase,params,prologue))
            graph,out=sg_graph((lambda:op(x)) if phase=='qkv' else (lambda:op(x,stream,start)))
            try:
                b,c,ratio=sg_pair(old,graph,inputs,banks,flush,20)
                for fresh,(ref,effective,rawref) in zip(banks,refs):
                    for a,bb in zip(inputs,fresh):a.copy_(bb)
                    graph.replay();sg_check(phase,out,ref,op.out,rawref,effective,router)
                if bool(op.counters.any()) or (phase!='qkv' and bool(op.ready.any())):
                    raise RuntimeError('final completion counter did not reset')
                stream_us=stream_ratio=0.
                if phase!='qkv':
                    separate=StreamingGlueControl(w,m,phase,params,cfg,split,prologue)
                    stream_graph,stream_out=sg_graph(lambda:separate(x,stream,start))
                    try:
                        stream_us,_,stream_ratio=sg_pair(stream_graph,graph,inputs,banks,flush,20)
                        for fresh,(ref,effective,rawref) in zip(banks,refs):
                            for a,bb in zip(inputs,fresh):a.copy_(bb)
                            stream_graph.replay()
                            sg_check(phase,stream_out,ref,separate.projection.out,rawref,effective,router)
                    finally:stream_graph.reset()
                finals.append((ratio,cfg,split,b,c,stream_us,stream_ratio))
            finally:graph.reset()
        return sorted(finals),trials,rejections
    finally:old.reset()


def sg_chain(deadline,weights,params,pool,m,picks,flush):
    """Paired dependent o->pool/router and unpool tail, excluding attention/MoE."""
    dev=pool.device;uw=weights['unpooler'][:1024].contiguous() if m<=8 else weights['unpooler']
    inputs=sg_inputs(m,1024,718,dev)+(sp_inputs(m,4096,77,dev),)
    banks=[sg_inputs(m,1024,seed,dev)+(sp_inputs(m,4096,seed+41,dev),) for seed in (43,99)]
    x,stream,start,moe=inputs
    old_o=CublasGlue(weights['o'],m,'o',params['o'])
    old_p=CublasGlue(pool,m,'pool',params['pool'])
    old_u=CublasGlue(uw,m,'unpool',params['unpool'])
    def control():
        h,s=old_o(x,stream,start);front=old_p(h,s,start);tail=old_u(moe,s,start)
        return front+tail+(s,)
    old,outold=sg_graph(control);lines=[]
    try:
        for prologue in (False,True):
            if _sp_time.monotonic()>deadline-1:break
            names=('o','pool_norm' if prologue else 'pool','unpool')
            if any((m,n) not in picks for n in names):
                lines.append(f'SG_CHAIN m={m} prologue={int(prologue)} missing_winner');continue
            cfg=[picks[m,n][1:3] for n in names]
            a=FusedStreamingProjection(weights['o'],m,*cfg[0],'o',params['o'],normalize=not prologue)
            b=FusedStreamingProjection(pool,m,*cfg[1],'pool',params['pool'],prologue)
            c=FusedStreamingProjection(uw,m,*cfg[2],'unpool',params['unpool'])
            def candidate():
                h,s=a(x,stream,start);front=b(s if prologue else h,s,start);tail=c(moe,s,start)
                return front+tail+(s,)
            new,out=sg_graph(candidate)
            try:
                for fresh in banks:
                    for target,value in zip(inputs,fresh):target.copy_(value)
                    new.replay();old.replay()
                    # Slot weights name different experts if BF16 cuBLAS o
                    # crosses a top-eight boundary. Gate IDs AND weights against
                    # the independent FP32 prefix below, rather than matching
                    # unnamed slots to that different prefix's route.
                    for i in (0,4,5,6):
                        try:sg_error(out[i],outold[i],1.e-2)
                        except RuntimeError as exc:raise RuntimeError(f'cuBLAS chain output {i}: {exc}') from exc
                    # Route agreement is against an independent FP32 prefix,
                    # not cuBLAS's rounded o result near a selection boundary.
                    (h,s),_,_=sg_oracle('o',x,stream,start,weights['o'],params['o'])
                    ref,eff,raw=sg_oracle('pool',s if prologue else h,s,start,pool,params['pool'],prologue)
                    sg_check('pool',out[:4],ref,b.out,raw,eff,weights['router'].float())
                    tail,eff,raw=sg_oracle('unpool',moe,s,start,uw,params['unpool'])
                    sg_check('unpool',out[4:6],tail,c.out,raw,eff)
                aa,bb,ratio=sg_pair(old,new,inputs,banks,flush,20)
                lines.append(f'SG_CHAIN m={m} prologue={int(prologue)} cublas_us={aa:.3f} us={bb:.3f} ratio={ratio:.5f} configs={cfg}')
            except Exception as exc:lines.append(f'SG_CHAIN_REJECT m={m} prologue={int(prologue)} {type(exc).__name__}:{str(exc)[:130]}')
            finally:new.reset()
    finally:old.reset()
    return lines


def lab(deadline,weights,state=None):
    old_tf32=torch.backends.cuda.matmul.allow_tf32
    old_bf=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    lines=[];picks={};begin=_sp_time.monotonic()
    try:
        sp_library();dev=weights['o'].device;params=sg_params(weights)
        router=weights['router'].float();hi=router.bfloat16();lo=(router-hi.float()).bfloat16()
        pool=torch.cat((weights['pooler'],hi,lo)).contiguous()
        flush=torch.zeros(256<<20,device=dev,dtype=torch.uint8)
        lines.append('SG_META cold=256MiB pairs=20 raw_us=1 FP32_gate=1e-4 router=hi+lo one_launch=1 variants=scalar_shared+vector_shared')
        # Reserve a fifth of the remaining budget for dependent chain pairs.
        stage_end=_sp_time.monotonic()+.8*max(0,deadline-_sp_time.monotonic()-2)
        work=[(m,n) for m in (1,8,16,32,128,256) for n in ('qkv','o','pool','pool_norm','unpool')]
        for i,(m,name) in enumerate(work):
            stop=_sp_time.monotonic()+max(0,stage_end-_sp_time.monotonic())/(len(work)-i)
            if stop<=_sp_time.monotonic()+.4:
                lines.append(f'SG_BUDGET m={m} phase={name}');continue
            phase='pool' if name=='pool_norm' else name
            w=weights['qkv_gate'] if phase=='qkv' else weights['o'] if phase=='o' else pool if phase=='pool' else weights['unpooler'][:1024].contiguous() if m<=8 else weights['unpooler']
            try:
                finals,trials,rejects=sg_trial(stop,w,m,phase,params.get(phase,{}),router if phase=='pool' else None,flush,name=='pool_norm')
                if finals:
                    best=finals[0];picks[m,name]=best;ratio,cfg,split,b,c,stream_us,stream_ratio=best
                    lines.append(f'SG_PICK {name} m={m} cfg={cfg} split={split} cublas_us={b:.3f} us={c:.3f} ratio={ratio:.5f} separate_us={stream_us:.3f} fuse_ratio={stream_ratio:.5f} raw_TBs={w.numel()*2/c/1e6:.3f} tried={len(trials)} rejected={len(rejects)}')
                else:lines.append(f'SG_NO_PICK {name} m={m} tried={len(trials)} rejected={len(rejects)}')
                if rejects:lines.append(f'SG_REJECT {name} m={m} '+' | '.join(rejects[:1]))
            except Exception as exc:lines.append(f'SG_STAGE_ERROR {name} m={m} {type(exc).__name__}:{str(exc)[:110]}')
        for i,m in enumerate((1,8,16,32,128,256)):
            stop=_sp_time.monotonic()+max(0,deadline-_sp_time.monotonic()-1)/(6-i)
            if _sp_time.monotonic()>deadline-1:lines.append(f'SG_BUDGET chain m={m}');continue
            try:lines+=sg_chain(stop,weights,params,pool,m,picks,flush)
            except Exception as exc:lines.append(f'SG_CHAIN_ERROR m={m} {type(exc).__name__}:{str(exc)[:110]}')
        lines.append(f'SG_DONE elapsed_s={_sp_time.monotonic()-begin:.3f} left_s={deadline-_sp_time.monotonic():.3f} picks={len(picks)}')
        return lines
    finally:
        torch.backends.cuda.matmul.allow_tf32=old_tf32
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=old_bf
