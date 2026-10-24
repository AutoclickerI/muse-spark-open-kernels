struct RouteGlue {
  const float* pr;const float* bias;const float* pre;int splits;
  int64_t ps,rs;
  int* out_ids;int* out_lids;float* out_wts;int* local_start;MoERouteMeta* meta;
};

__device__ __forceinline__ float rg_wsum(float x){
  #pragma unroll
  for(int d=16;d;d>>=1)x+=__shfl_xor_sync(~0u,x,d);return x;
}
// Each CTA constructs its own summary; no inter-CTA publication or waiting.
// Only CTA zero exports layer IDs. The MoE body consumes the private ids/weights.
__device__ __noinline__ void rg_prepare(RouteGlue rg,int M,int* ids,float* weights,float* inv){
  int lane=threadIdx.x&31,warp=threadIdx.x>>5;
  for(int tok=warp;tok<M;tok+=8){
    float score[8],biased[8];
    #pragma unroll
    for(int j=0;j<8;++j){
      int e=j*32+lane;float logit=0;
      for(int s=0;s<rg.splits;++s){
        const float* p=rg.pr+s*rg.ps+(int64_t)tok*rg.rs;
        logit=__fadd_rn(logit,__fadd_rn(p[4096+e],p[4352+e]));
      }
      score[j]=1.f/(1.f+__expf(-logit*0.1767766952966369f));
      biased[j]=score[j]+rg.bias[e];
    }
    int pick[8];float prob[8],total=0;
    #pragma unroll
    for(int k=0;k<8;++k){
      float best=-CUDART_INF_F;int id=256;
      #pragma unroll
      for(int j=0;j<8;++j){int e=j*32+lane;if(biased[j]>best||(biased[j]==best&&e<id)){best=biased[j];id=e;}}
      #pragma unroll
      for(int d=16;d;d>>=1){float b=__shfl_xor_sync(~0u,best,d);int i=__shfl_xor_sync(~0u,id,d);
        if(b>best||(b==best&&i<id)){best=b;id=i;}}
      float p=0;
      #pragma unroll
      for(int j=0;j<8;++j)if(j*32+lane==id){p=score[j];biased[j]=-CUDART_INF_F;}
      p=rg_wsum(p);pick[k]=id;prob[k]=p;total+=p;
    }
    if(lane<8){int local=(pick[lane]-*rg.local_start+256)%256;
      ids[tok*8+lane]=local<32?local:-1;
      weights[tok*8+lane]=prob[lane]/(total+1e-15f);
      if(blockIdx.x==0){rg.out_ids[tok*8+lane]=pick[lane];rg.out_lids[tok*8+lane]=ids[tok*8+lane];rg.out_wts[tok*8+lane]=weights[tok*8+lane];}}
  }
  __syncthreads();
  if(M==1){
    __shared__ float sums[8];float ss=0;
    for(int d=threadIdx.x;d<4096;d+=256){float v=0;
      for(int s=0;s<rg.splits;++s)v+=rg.pr[(int64_t)s*rg.ps+d];
      ss=fmaf(v,v,ss);
    }
    ss=rg_wsum(ss);if(lane==0)sums[warp]=ss;
    __syncthreads();ss=lane<8?sums[lane]:0.f;ss=rg_wsum(ss);
    if(threadIdx.x==0)inv[0]=rsqrtf(ss*(1.f/4096)+1e-5f);
  }else{
    for(int tok=warp;tok<M;tok+=8){float ss=0;
      for(int d=lane;d<4096;d+=32){float v=0;
        for(int s=0;s<rg.splits;++s)v+=rg.pr[(int64_t)s*rg.ps+(int64_t)tok*rg.rs+d];
        ss=fmaf(v,v,ss);
      }
      ss=rg_wsum(ss);if(lane==0)inv[tok]=rsqrtf(ss*(1.f/4096)+1e-5f);
    }
  }
  __syncthreads();
}

__device__ __forceinline__ __nv_bfloat16 rg_value(RouteGlue rg,int token,int col,float inv){
  float value=0;
  for(int s=0;s<rg.splits;++s)value+=rg.pr[(int64_t)s*rg.ps+(int64_t)token*rg.rs+col];
  return __float2bfloat16_rn(value*inv*rg.pre[col]);
}
