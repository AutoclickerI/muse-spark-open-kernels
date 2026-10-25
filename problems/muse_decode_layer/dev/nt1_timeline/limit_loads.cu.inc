// Timing-only controls: no expert output, no dequant/MMA. Same S11 tile addresses, order and two-stage TMA ring.
__global__ void __launch_bounds__(256,2) loads_pipeline(const int* ids,const uint8_t* w13p,const uint8_t* w2p,
    uint32_t* sums,int M,int start,int nstage){
 extern __shared__ __align__(128) uint8_t smem[];
 auto* fullb=reinterpret_cast<uint64_t*>(smem+nstage*TILE_BYTES);auto* emptyb=fullb+nstage;
 auto* rt=reinterpret_cast<Route*>(emptyb+nstage);
 int tid=threadIdx.x,lane=tid&31,warp=tid>>5,c=blockIdx.x,G=gridDim.x;
 if(tid<32)rt->n[tid]=0;
 if(tid==0){for(int s=0;s<nstage;++s){mbar_init(fullb+s,1);mbar_init(emptyb+s,8);}
  asm volatile("fence.mbarrier_init.release.cluster;":::"memory");
  uint64_t pf,pn;asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;":"=l"(pf));
  asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;":"=l"(pn));
  rt->pre.pol_first=pf;rt->pre.pol_normal=pn;rt->pre.index=0;rt->pre.stage=0;rt->pre.phase=0;}
 __syncthreads();
 for(int i=tid;i<M*8;i+=256){int e=__ldg(ids+i)-start;if(e>=0&&e<32)atomicAdd(rt->n+e,1);}
 __syncthreads();
 if(warp==0){int nb=(rt->n[lane]+7)/8,u1=nb*NT1,u2=nb*NT2;rt->nblk[lane]=nb;
#pragma unroll
  for(int d=1;d<32;d<<=1){int a=__shfl_up_sync(~0u,u1,d),b=__shfl_up_sync(~0u,u2,d);if(lane>=d){u1+=a;u2+=b;}}
  rt->us1[lane+1]=u1;rt->us2[lane+1]=u2;if(lane==0){rt->us1[0]=0;rt->us2[0]=0;}
  int U1=__shfl_sync(~0u,u1,31),U2=__shfl_sync(~0u,u2,31);
  if(lane==0){rt->c1=min(NT1,max(1,(U1+G/2)/G));rt->c2=min(NT2,max(1,(U2+G/2)/G));}}
 __syncthreads();
 int U1=rt->us1[32],U2=rt->us2[32],a1=(long long)U1*c/G,b1=(long long)U1*(c+1)/G,
     a2=(long long)U2*c/G,b2=(long long)U2*(c+1)/G,first=b1-a1,total=first+b2-a2;
 if(tid==0){if(total){int e,nb,m;int pass=first?0:1,cc=pass?rt->c2:rt->c1,ntile=pass?NT2:NT1;
   decode_unit(pass?a2:a1,pass?rt->us2:rt->us1,rt->nblk,cc,ntile,e,nb,m);
   rt->pre.pass=pass;rt->pre.e=e;rt->pre.nb=nb;rt->pre.m=m;rt->pre.mb=m/cc*cc;rt->pre.me=min(ntile,m/cc*cc+cc);}
  for(int s=0;s<nstage;++s)prefetch_one(rt,total,first,a2,nstage,smem,fullb,emptyb,w13p,w2p);}
 __syncthreads();
 int st=0,ph=0;uint32_t acc=0;
 for(int u=0;u<total;++u){mbar_wait(fullb+st,ph);
  const uint8_t* tile=smem+st*TILE_BYTES;const uint4* p=reinterpret_cast<const uint4*>(tile)+warp*8*32+lane;
  uint4 scv=*(reinterpret_cast<const uint4*>(tile+TILE_W_BYTES)+warp*32+lane);acc+=scv.x+scv.y+scv.z+scv.w;
#pragma unroll
  for(int g=0;g<8;++g){uint4 v=p[g*32];acc+=v.x+v.y+v.z+v.w;}
  __syncwarp();if(lane==0)mbar_arrive(emptyb+st);if(++st==nstage){st=0;ph^=1;}
  bar_consumers();if(tid==0)prefetch_one(rt,total,first,a2,nstage,smem,fullb,emptyb,w13p,w2p);
 }
 sums[c*256+tid]=acc;
}
// Unconstrained read stream. Each block handles 128 KiB contiguous chunks, no shared-memory residency tax.
__global__ void stream_max(const uint8_t* w13p,const uint8_t* w2p,const int* experts,int count,uint32_t* sums){
 int tid=threadIdx.x;uint32_t acc=0;const int chunks=216;
 for(int chunk=blockIdx.x;chunk<count*chunks;chunk+=gridDim.x){
  int rank=chunk/chunks,sub=chunk-rank*chunks,e=experts[rank],bank=sub>=144;
  const uint8_t* src=(bank?w2p:w13p)+(size_t)e*(bank?9437184:18874368)+(size_t)(bank?sub-144:sub)*131072;
  uint4 v[8];
#pragma unroll
  for(int j=0;j<8;++j){const uint4* p=reinterpret_cast<const uint4*>(src)+tid+j*256;
   asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];":"=r"(v[j].x),"=r"(v[j].y),"=r"(v[j].z),"=r"(v[j].w):"l"(p));}
#pragma unroll
  for(int j=0;j<8;++j)acc+=v[j].x+v[j].y+v[j].z+v[j].w;
  // 256 threads x 8 loads = 32 KiB; four subsegments cover the whole 128 KiB chunk.
#pragma unroll
  for(int k=1;k<4;++k){
#pragma unroll
   for(int j=0;j<8;++j){uint4 a;const uint4* p=reinterpret_cast<const uint4*>(src)+k*2048+tid+j*256;
    asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];":"=r"(a.x),"=r"(a.y),"=r"(a.z),"=r"(a.w):"l"(p));acc+=a.x+a.y+a.z+a.w;}}
 }
 sums[blockIdx.x*256+tid]=acc;
}
} // namespace
int64_t limit_init(){int d,opt;C10_CUDA_CHECK(cudaGetDevice(&d));C10_CUDA_CHECK(cudaDeviceGetAttribute(&opt,cudaDevAttrMaxSharedMemoryPerBlockOptin,d));
 cudaFuncAttributes a{};C10_CUDA_CHECK(cudaFuncGetAttributes(&a,loads_pipeline));C10_CUDA_CHECK(cudaFuncSetAttribute(loads_pipeline,cudaFuncAttributeMaxDynamicSharedMemorySize,opt-a.sharedSizeBytes));return opt;}
int64_t limit_smem(int64_t stages){return stages*TILE_BYTES+2*stages*8+sizeof(Route);}
void limit_loads(std::vector<int64_t>v,int64_t M,int64_t start,int64_t stages,int64_t grid,int64_t smem,int64_t stream){
 loads_pipeline<<<grid,256,smem,(cudaStream_t)stream>>>((const int*)v[0],(const uint8_t*)v[1],(const uint8_t*)v[2],(uint32_t*)v[3],M,start,stages);C10_CUDA_CHECK(cudaGetLastError());}
void limit_stream(std::vector<int64_t>v,int64_t count,int64_t grid,int64_t stream){
 stream_max<<<grid,256,0,(cudaStream_t)stream>>>((const uint8_t*)v[0],(const uint8_t*)v[1],(const int*)v[2],count,(uint32_t*)v[3]);C10_CUDA_CHECK(cudaGetLastError());}
