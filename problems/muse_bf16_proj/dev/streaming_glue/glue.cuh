#include <cuda/atomic>

namespace stream_proj {
// All scratch is private to one call site. Publication is a last-arrival
// continuation: a CTA never waits for another CTA to become resident.
struct Glue {
  int phase, prologue;
  const float *stream, *alpha, *beta, *pre, *post, *bias, *input_pre;
  float* stream_out;
  B* norm;
  int *ids, *local_ids;
  float* weights;
  const int* start;
  unsigned* ready;
  float* moments;
};

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int d=16; d; d>>=1) v += __shfl_xor_sync(0xffffffff, v, d);
  return v;
}
__device__ __forceinline__ unsigned long long warp_max(unsigned long long v) {
#pragma unroll
  for (int d=16; d; d>>=1) {
    auto b=__shfl_xor_sync(0xffffffff, v, d); v=v>b?v:b;
  }
  return v;
}
__device__ __forceinline__ float group_sum(float v, int threads, float* tmp) {
  v=warp_sum(v);
  if(threads==32) return v;
  if ((threadIdx.x&31)==0) tmp[threadIdx.x/32]=v;
  __syncthreads();
  const int first=(threadIdx.x/threads)*(threads/32);
  float s=0;
  for(int i=0;i<threads/32;++i) s+=tmp[first+i];
  __syncthreads();
  return s;
}
__device__ __forceinline__ unsigned long long group_max(unsigned long long v, int threads,
                                                       unsigned long long* tmp) {
  v=warp_max(v);
  if(threads==32) return v;
  if ((threadIdx.x&31)==0) tmp[threadIdx.x/32]=v;
  __syncthreads();
  const int first=(threadIdx.x/threads)*(threads/32);
  unsigned long long s=0;
  for(int i=0;i<threads/32;++i) s=s>tmp[first+i]?s:tmp[first+i];
  __syncthreads();
  return s;
}

template<int T, int THREADS>
__device__ __forceinline__ void init_input(Glue gl, const B* x, int M, int K, float* inv) {
  const int tok0=blockIdx.z*T;
  // One warp per input row; this pass is needed only when the producer leaves
  // its FP32 residual stream for the pool/router prologue to normalize.
  for (int t=threadIdx.x/32;t<T;t+=THREADS/32) {
    float s=0;
    if(tok0+t<M) {
      const float* row=(const float*)x+(size_t)(tok0+t)*K;
      for(int k=(threadIdx.x&31)*4;k<K;k+=32*4) {
        float4 v=*(const float4*)(row+k);
        s+=v.x*v.x+v.y*v.y+v.z*v.z+v.w*v.w;
      }
    }
    s=warp_sum(s);
    if((threadIdx.x&31)==0) inv[t]=rsqrtf(s/K+1.e-5f);
  }
  __syncthreads();
}

__device__ __forceinline__ uint4 input16(Glue gl, const B* x, int tok, int k, int K, float inv) {
  if(!gl.prologue) return *(const uint4*)(x+(size_t)tok*K+k);
  const float* row=(const float*)x+(size_t)tok*K+k;
  float4 a=*(const float4*)row, b=*(const float4*)(row+4);
  float4 p=*(const float4*)(gl.input_pre+k), q=*(const float4*)(gl.input_pre+k+4);
  uint4 v;
  ((B*)&v)[0]=__float2bfloat16_rn((a.x*inv)*p.x);
  ((B*)&v)[1]=__float2bfloat16_rn((a.y*inv)*p.y);
  ((B*)&v)[2]=__float2bfloat16_rn((a.z*inv)*p.z);
  ((B*)&v)[3]=__float2bfloat16_rn((a.w*inv)*p.w);
  ((B*)&v)[4]=__float2bfloat16_rn((b.x*inv)*q.x);
  ((B*)&v)[5]=__float2bfloat16_rn((b.y*inv)*q.y);
  ((B*)&v)[6]=__float2bfloat16_rn((b.z*inv)*q.z);
  ((B*)&v)[7]=__float2bfloat16_rn((b.w*inv)*q.w);
  return v;
}

template<int R, int T, int THREADS>
__device__ __noinline__ void publish(Glue gl, float* raw, int M, int N, int nt) {
  if(!gl.phase) return;
  __shared__ int done[T];
  __shared__ float tmp[THREADS/32];
  __shared__ unsigned long long keys_tmp[THREADS/32];
  const int tok0=blockIdx.z*T;
  const int valid=M-tok0<T?M-tok0:T;
  int groups=1;
  while(groups*2<=THREADS/32 && groups*2<=valid) groups*=2;
  const int threads=THREADS/groups, g=threadIdx.x/threads, lane=threadIdx.x%threads;
  if(gl.moments) {
    // Reduce local moments while all output-row CTAs are still available.
    // The final CTA then reads nt float4s instead of making multiple passes
    // over a complete row. The residual square is a quadratic in the branch
    // inverse RMS; all moments and accumulation remain FP32.
    __syncthreads();
    for(int base=0;base<T;base+=groups) {
      int t=base+g,tok=tok0+t;
      bool active=t<T && tok<M;
      float sb=0,sa=0,sab=0,sbb=0;
      if(active) for(int r=lane;r<R;r+=threads) {
        int j=blockIdx.x*R+r;
        if(gl.phase==2) {
          float p=j<4096?raw[(size_t)tok*N+j]:0.f;
          sb+=p*p;
        } else {
          for(;j<8192;j+=nt*R) {
            float p=j<N?raw[(size_t)tok*N+j]:0.f;
            if(gl.phase==3)p*=gl.post[j];
            float a=gl.alpha[j]*gl.stream[(size_t)tok*8192+j],b=gl.beta[j]*p;
            sb+=p*p;sa+=a*a;sab+=a*b;sbb+=b*b;
          }
        }
      }
      sb=group_sum(sb,threads,tmp);sa=group_sum(sa,threads,tmp);
      sab=group_sum(sab,threads,tmp);sbb=group_sum(sbb,threads,tmp);
      if(active && lane==0)
        ((float4*)gl.moments)[(size_t)tok*nt+blockIdx.x]=make_float4(sb,sa,sab,sbb);
    }
  }
  // Every thread publishes its own output writes before any thread announces
  // tile completion. The last acq_rel RMW acquires all earlier publications.
  __threadfence();
  __syncthreads();
  for(int t=threadIdx.x;t<T;t+=THREADS) {
    if(tok0+t<M) {
      cuda::atomic_ref<unsigned,cuda::thread_scope_device> a(gl.ready[tok0+t]);
      done[t]=(a.fetch_add(1,cuda::memory_order_acq_rel)==(unsigned)(nt-1));
    } else done[t]=0;
  }
  __syncthreads();
  for(int base=0;base<T;base+=groups) {
    const int t=base+g, tok=tok0+t;
    const bool active=t<T && tok<M && done[t];
    float sum=0;
    float sa=0,sab=0,sbb=0;
    if(gl.moments) {
      if(active) for(int j=lane;j<nt;j+=threads) {
        const volatile float* p=gl.moments+((size_t)tok*nt+j)*4;
        sum+=p[0];sa+=p[1];sab+=p[2];sbb+=p[3];
      }
      sum=group_sum(sum,threads,tmp);sa=group_sum(sa,threads,tmp);
      sab=group_sum(sab,threads,tmp);sbb=group_sum(sbb,threads,tmp);
    }
    if(gl.phase==2) {
      // Pool outputs and hi/lo router logits stay FP32 through the epilogue.
      if(active && !gl.moments) for(int j=lane;j<4096;j+=threads) {
        float p=((volatile float*)raw)[(size_t)tok*N+j]; sum+=p*p;
      }
      float inv=rsqrtf((gl.moments?sum:group_sum(sum,threads,tmp))/4096.f+1.e-5f);
      if(active) for(int j=lane;j<4096;j+=threads) {
        float p=((volatile float*)raw)[(size_t)tok*N+j];
        gl.norm[(size_t)tok*4096+j]=__float2bfloat16_rn((p*inv)*gl.pre[j]);
      }
      // Deterministic top eight, using the FP32 score and expert-index tie key.
      unsigned long long key[8]={};
#pragma unroll
      for(int i=0;i<8;++i) {
        int e=lane+i*threads;
        if(active && e<256) {
          float logit=((volatile float*)raw)[(size_t)tok*N+4096+e]
                     +((volatile float*)raw)[(size_t)tok*N+4352+e];
          float score=1.f/(1.f+__expf(-logit*.1767766952966369f));
          int bits=__float_as_int(score+gl.bias[e]);
          unsigned ordered=bits<0?(unsigned)~bits:((unsigned)bits^0x80000000u);
          key[i]=((unsigned long long)ordered<<32)|(unsigned)(255-e);
        }
      }
      float picked[8],total=0;
#pragma unroll
      for(int slot=0;slot<8;++slot) {
        unsigned long long best=0;
#pragma unroll
        for(int i=0;i<8;++i) best=best>key[i]?best:key[i];
        best=group_max(best,threads,keys_tmp);
#pragma unroll
        for(int i=0;i<8;++i) if(key[i]==best) key[i]=0;
        if(active && lane==0) {
          int e=255-(unsigned)best;
          float logit=((volatile float*)raw)[(size_t)tok*N+4096+e]
                     +((volatile float*)raw)[(size_t)tok*N+4352+e];
          picked[slot]=1.f/(1.f+__expf(-logit*.1767766952966369f));
          total+=picked[slot];
          gl.ids[tok*8+slot]=e;
          int local=(e-*gl.start+256)%256;
          gl.local_ids[tok*8+slot]=local<32?local:-1;
        }
      }
      if(active && lane==0) {
#pragma unroll
        for(int slot=0;slot<8;++slot)
          gl.weights[tok*8+slot]=picked[slot]/(total+1.e-15f);
      }
    } else {
      if(active && !gl.moments) for(int j=lane;j<8192;j+=threads) {
        float p=j<N?((volatile float*)raw)[(size_t)tok*N+j]:0.f;
        if(gl.phase==3) p*=gl.post[j];
        sum+=p*p;
      }
      float inv=rsqrtf((gl.moments?sum:group_sum(sum,threads,tmp))/8192.f+1.e-8f);
      float out_inv=gl.moments?rsqrtf(fmaxf(0.f,sa+2.f*inv*sab+inv*inv*sbb)/8192.f+1.e-5f):0.f;
      sum=0;
      if(active) for(int j=lane;j<8192;j+=threads) {
        float p=j<N?((volatile float*)raw)[(size_t)tok*N+j]:0.f;
        if(gl.phase==3) p*=gl.post[j];
        float s=gl.alpha[j]*gl.stream[(size_t)tok*8192+j]+gl.beta[j]*(p*inv);
        gl.stream_out[(size_t)tok*8192+j]=s;
        sum+=s*s;
        if(gl.moments && gl.norm)
          gl.norm[(size_t)tok*8192+j]=__float2bfloat16_rn((s*out_inv)*gl.pre[j]);
      }
      inv=gl.moments?out_inv:rsqrtf(group_sum(sum,threads,tmp)/8192.f+1.e-5f);
      if(active && gl.norm && !gl.moments) for(int j=lane;j<8192;j+=threads) {
        float s=gl.stream_out[(size_t)tok*8192+j];
        gl.norm[(size_t)tok*8192+j]=__float2bfloat16_rn((s*inv)*gl.pre[j]);
      }
    }
    __syncthreads();
    if(active && lane==0) {
      cuda::atomic_ref<unsigned,cuda::thread_scope_device> a(gl.ready[tok]);
      a.store(0,cuda::memory_order_relaxed);
    }
    __syncthreads();
  }
}
}  // namespace stream_proj
