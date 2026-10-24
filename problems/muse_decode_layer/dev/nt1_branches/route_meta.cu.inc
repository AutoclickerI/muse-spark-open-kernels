__global__ void make_route_meta(const int* ids,int M,int start,MoERouteMeta* dst){
  __shared__ int n[32],wc[8][32],off[32],run[32];
  int t=threadIdx.x,lane=t&31,warp=t>>5;
  if(t<32)n[t]=0;
  dst->pair_of[t]=-1;dst->pos_of[t]=-1;
  if(t<32)dst->token_local[t]=0;
  __syncthreads();
  int e=t<M*8?ids[t]-start:-1;
  if(e<0||e>=32)e=-1;
  if(e>=0)atomicAdd(n+e,1);
  __syncthreads();
  if(warp==0){
    int count=n[lane],nb=(count+7)/8;
    int p=count,u1=nb*512,u2=nb*256;
    #pragma unroll
    for(int d=1;d<32;d<<=1){
      int a=__shfl_up_sync(~0u,p,d),b=__shfl_up_sync(~0u,u1,d),c=__shfl_up_sync(~0u,u2,d);
      if(lane>=d){p+=a;u1+=b;u2+=c;}
    }
    off[lane]=run[lane]=p-count;
    dst->count[lane]=count;dst->offset[lane]=p-count;dst->blocks8[lane]=nb;
    dst->units13[lane+1]=u1;dst->units2[lane+1]=u2;
    if(lane==31){dst->pairs=p;dst->offset[32]=p;}
    if(lane==0){dst->version=1;dst->tokens=M;dst->local_start=start;dst->units13[0]=dst->units2[0]=0;}
  }
  wc[warp][lane]=0;__syncthreads();
  unsigned mk=__match_any_sync(~0u,e);int rank=__popc(mk&((1u<<lane)-1));
  if(e>=0&&rank==0)wc[warp][e]=__popc(mk);
  __syncthreads();
  if(t<32){int sum=off[t];for(int w=0;w<8;++w){int v=wc[w][t];wc[w][t]=sum;sum+=v;}}
  __syncthreads();
  if(e>=0){int pos=wc[warp][e]+rank;dst->pair_of[pos]=t;dst->pos_of[t]=pos;}
  if(t<M){int cnt=0;
    #pragma unroll
    for(int s=0;s<8;++s){int x=ids[t*8+s]-start;cnt+=(x>=0&&x<32);}
    dst->token_local[t]=cnt;
  }
}
