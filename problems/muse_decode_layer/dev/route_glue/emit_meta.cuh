// MoERouteMeta v1 from nt1_branches/route_meta.h. Inputs are the CTA-private
// local route IDs; shared scratch reuses the MoE body's Route storage before
// that storage is initialized by the expert scheduler.
__device__ __forceinline__ void rg_emit_meta(MoERouteMeta* dst,int M,int start,const int* ids,int* scratch){
  int* n=scratch;int* wc=n+32;int* off=wc+256;
  int t=threadIdx.x,lane=t&31,warp=t>>5;
  for(int j=t;j<(int)sizeof(MoERouteMeta)/4;j+=256)((int*)dst)[j]=0;
  __syncthreads();
  if(t<32)n[t]=0;
  dst->pair_of[t]=-1;dst->pos_of[t]=-1;
  __syncthreads();
  int e=t<M*8?ids[t]:-1;
  if(e<0||e>=32)e=-1;
  if(e>=0)atomicAdd(n+e,1);
  __syncthreads();
  if(warp==0){
    int count=n[lane],nb=(count+7)/8,p=count,u1=nb*512,u2=nb*256;
    #pragma unroll
    for(int d=1;d<32;d<<=1){
      int a=__shfl_up_sync(~0u,p,d),b=__shfl_up_sync(~0u,u1,d),c=__shfl_up_sync(~0u,u2,d);
      if(lane>=d){p+=a;u1+=b;u2+=c;}
    }
    off[lane]=p-count;
    dst->count[lane]=count;dst->offset[lane]=p-count;dst->blocks8[lane]=nb;
    dst->units13[lane+1]=u1;dst->units2[lane+1]=u2;
    if(lane==31){dst->pairs=p;dst->offset[32]=p;}
    if(lane==0){dst->version=1;dst->tokens=M;dst->local_start=start;}
  }
  wc[warp*32+lane]=0;__syncthreads();
  unsigned mask=__match_any_sync(~0u,e);int rank=__popc(mask&((1u<<lane)-1));
  if(e>=0&&rank==0)wc[warp*32+e]=__popc(mask);
  __syncthreads();
  if(t<32){int sum=off[t];for(int w=0;w<8;++w){int v=wc[w*32+t];wc[w*32+t]=sum;sum+=v;}}
  __syncthreads();
  if(e>=0){int pos=wc[warp*32+e]+rank;dst->pair_of[pos]=t;dst->pos_of[t]=pos;}
  if(t<M){int count=0;
    #pragma unroll
    for(int s=0;s<8;++s)count+=(ids[t*8+s]>=0);
    dst->token_local[t]=count;
  }
  __syncthreads();
}
