// Unique (expert, weight tile) ownership. Each K group's decoded operands are reused over every token block.
// S11 packed bank, exact BF16 intermediates. REDO is a fixed-work diagnostic, never an omitted-compute arm.
constexpr int MAX_BLOCKS = 4;
struct ReuseRoute {
  Route base;
  int uniq[33];
  int a1,b1,a2,b2,first,total;
};
__device__ __forceinline__ int unique_boundary(int u, const int* us, const int* nblk, const int* uniq, int ntile) {
  if (u >= us[NLOC]) return uniq[NLOC] * ntile;
  int e=0;
#pragma unroll
  for (int d=16;d;d>>=1) if(us[e+d]<=u)e+=d;
  return uniq[e]*ntile + (u-us[e]+nblk[e]-1)/nblk[e];
}
__device__ __forceinline__ void unique_decode(int u,const int* uniq,int ntile,int& e,int& m) {
  const int rank=u/ntile;
  e=0;
#pragma unroll
  for(int d=16;d;d>>=1)if(uniq[e+d]<=rank)e+=d;
  m=u-rank*ntile;
}
__device__ __forceinline__ void reuse_prefetch(ReuseRoute* rr,int nstage,uint8_t* stages,uint64_t* fullb,
    uint64_t* emptyb,const uint8_t* w13p,const uint8_t* w2p) {
  auto& rt=rr->base;auto& p=rt.pre;
  int idx=p.index;
  if(idx>=rr->total)return;
  const bool pass=idx>=rr->first;
  const int u=pass?rr->a2+idx-rr->first:rr->a1+idx;
  int e,m;unique_decode(u,rr->uniq,pass?NT2:NT1,e,m);
  int st=p.stage,ph=p.phase;
  const uint8_t* src=(pass?w2p:w13p)+(static_cast<size_t>(e)*(pass?NT2:NT1)+m)*TILE_BYTES;
  mbar_wait(&emptyb[st],ph^1u);mbar_expect_tx(&fullb[st],TILE_BYTES);
  bulk_g2s(stages+st*TILE_BYTES,src,TILE_BYTES,&fullb[st],p.pol_first);
  p.index=idx+1;
  if(++st==nstage){st=0;ph^=1;}p.stage=st;p.phase=ph;
}
__device__ __forceinline__ uint64_t reuse_clock(){uint64_t t;asm volatile("mov.u64 %0, %%clock64;":"=l"(t)::"memory");return t;}

// Trace per CTA: clocks[entry,route,copy-wait,decode,MMA+activation,reduce+epi,publish,ready,barrier,end],
// units13,units2,blocks13,blocks2. Clocks inside the math are deliberately in a separate instantiation.
template<bool REDO,bool TRACE>
__global__ void __launch_bounds__(256,2) moe_reuse(
    const __nv_bfloat16* __restrict__ x,const int* __restrict__ ids,const float* __restrict__ tw,
    const __nv_bfloat16* __restrict__ pnw,const uint8_t* __restrict__ w13p,const uint8_t* __restrict__ w2p,
    const float* __restrict__ gs13,const float* __restrict__ gs2,__nv_bfloat16* __restrict__ inter,
    float* __restrict__ y,float* __restrict__ ss,int* __restrict__ cnt,__nv_bfloat16* __restrict__ out,
    int M,int local_start,int nstage,uint64_t* trace) {
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t* stages=smem;
  float* red=reinterpret_cast<float*>(stages+nstage*TILE_BYTES);
  uint32_t* cached_last=reinterpret_cast<uint32_t*>(red+2*NCW*32*4);
  uint64_t* fullb=reinterpret_cast<uint64_t*>(cached_last+8*256);uint64_t* emptyb=fullb+nstage;
  ReuseRoute* rr=reinterpret_cast<ReuseRoute*>(emptyb+nstage);Route* rt=&rr->base;
  const int tid=threadIdx.x,lane=tid&31,warp=tid>>5,c=blockIdx.x,G=gridDim.x,npairs=M*8;
  int8_t* eflat=reinterpret_cast<int8_t*>(rr+1);
  int16_t* pos_of=reinterpret_cast<int16_t*>(eflat+((npairs+15)&~15));int16_t* pair_of=pos_of+npairs;
  uint64_t clocks[14]={};if constexpr(TRACE){if(tid==0)clocks[0]=reuse_clock();}
  int gen0=0;
  if(tid<NLOC)rt->n[tid]=0;
  if(tid==0){
    for(int s=0;s<nstage;++s){mbar_init(&fullb[s],1);mbar_init(&emptyb[s],NCW);}
    asm volatile("fence.mbarrier_init.release.cluster;":::"memory");
    uint64_t pf;asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;":"=l"(pf));
    rt->pre.pol_first=pf;rt->pre.index=0;rt->pre.stage=0;rt->pre.phase=0;
  }
  asm volatile("griddepcontrol.wait;":::"memory");
  if(tid==0)gen0=ld_acquire(cnt+C_GEN);
  __syncthreads();
  for(int i=tid;i<npairs;i+=256){int e=__ldg(ids+i)-local_start;if(e<0||e>=NLOC)e=-1;
    eflat[i]=e;if(e>=0)atomicAdd(&rt->n[e],1);}
  __syncthreads();
  if(warp==0){
    const int n=rt->n[lane],nb=(n+7)/8;
    int off=n,work=nb,un=n>0;
#pragma unroll
    for(int d=1;d<32;d<<=1){int a=__shfl_up_sync(~0u,off,d),b=__shfl_up_sync(~0u,work,d),v=__shfl_up_sync(~0u,un,d);
      if(lane>=d){off+=a;work+=b;un+=v;}}
    rt->off[lane]=off-n;rt->run[lane]=off-n;rt->nblk[lane]=nb;
    rt->us1[lane+1]=work*NT1;rt->us2[lane+1]=work*NT2;rr->uniq[lane+1]=un;
    if(lane==0){rt->us1[0]=0;rt->us2[0]=0;rr->uniq[0]=0;}
  }
  __syncthreads();
  if(tid==0){
    int U1=rt->us1[32],U2=rt->us2[32];
    rr->a1=unique_boundary((long long)U1*c/G,rt->us1,rt->nblk,rr->uniq,NT1);
    rr->b1=unique_boundary((long long)U1*(c+1)/G,rt->us1,rt->nblk,rr->uniq,NT1);
    rr->a2=unique_boundary((long long)U2*c/G,rt->us2,rt->nblk,rr->uniq,NT2);
    rr->b2=unique_boundary((long long)U2*(c+1)/G,rt->us2,rt->nblk,rr->uniq,NT2);
    rr->first=rr->b1-rr->a1;rr->total=rr->first+rr->b2-rr->a2;
    for(int s=0;s<nstage;++s)reuse_prefetch(rr,nstage,stages,fullb,emptyb,w13p,w2p);
  }
  __syncthreads();
  for(int base=0;base<npairs;base+=256){
    rt->wc[warp][lane]=0;bar_consumers();int i=base+tid,e=i<npairs?eflat[i]:-1;
    unsigned mk=__match_any_sync(~0u,e);int rank=__popc(mk&((1u<<lane)-1u));
    if(e>=0&&rank==0)rt->wc[warp][e]=__popc(mk);bar_consumers();
    if(tid<NLOC){int s=rt->run[tid];for(int w=0;w<NCW;++w){int t=rt->wc[w][tid];rt->wc[w][tid]=s;s+=t;}rt->run[tid]=s;}
    bar_consumers();if(i<npairs){if(e>=0){int pos=rt->wc[warp][e]+rank;pos_of[i]=pos;pair_of[pos]=i;}else pos_of[i]=-1;}
  }
  bar_consumers();
  if constexpr(TRACE){if(tid==0)clocks[1]=reuse_clock()-clocks[0];}
  uint32_t cached_b[7][8];
  int st=0,ph=0,buf=0,cur_e=-1,cur_pass=-1,seg_tiles=0;
  auto publish=[&](){
    if(cur_pass==0&&seg_tiles){uint64_t t=0;if constexpr(TRACE){if(tid==0)t=reuse_clock();}
      __threadfence();bar_consumers();if(tid==0){atomicAdd(cnt+C_DONE+cur_e,seg_tiles);if constexpr(TRACE)clocks[6]+=reuse_clock()-t;}}
  };
  for(int pass=0;pass<2;++pass){int ua=pass?rr->a2:rr->a1,ub=pass?rr->b2:rr->b1;
    int ntile=pass?NT2:NT1;
    for(int u=ua;u<ub;++u){
      int e,m;unique_decode(u,rr->uniq,ntile,e,m);const int nb=rt->nblk[e],ne=rt->n[e],off=rt->off[e];
      if(e!=cur_e||pass!=cur_pass){publish();cur_e=e;cur_pass=pass;seg_tiles=0;
        if(pass){uint64_t t=0;if constexpr(TRACE){if(tid==0)t=reuse_clock();}
          while(ld_acquire(cnt+C_DONE+e)<NT1)__nanosleep(64);
          if constexpr(TRACE){if(tid==0)clocks[7]+=reuse_clock()-t;}}
        const int idx=lane>>2;const __nv_bfloat16* row=nullptr;
        if(idx<ne)row=pass?inter+(size_t)(off+idx)*HD:x+(size_t)(pair_of[off+idx]>>3)*HD;
#pragma unroll
        for(int g=0;g<8;++g){uint4 a=make_uint4(0,0,0,0),b=a;
          if(row){const uint4* p=reinterpret_cast<const uint4*>(row+warp*512+g*64+(lane&3)*16);if(pass){a=__ldcg(p);b=__ldcg(p+1);}else{a=__ldg(p);b=__ldg(p+1);}}
          if(g<7){cached_b[g][0]=a.x;cached_b[g][1]=a.y;cached_b[g][2]=a.z;cached_b[g][3]=a.w;
            cached_b[g][4]=b.x;cached_b[g][5]=b.y;cached_b[g][6]=b.z;cached_b[g][7]=b.w;
          }else{const uint32_t v[8]={a.x,a.y,a.z,a.w,b.x,b.y,b.z,b.w};
#pragma unroll
            for(int k=0;k<8;++k)cached_last[k*256+tid]=v[k];}

        }
      }
      uint64_t t=0;if constexpr(TRACE){if(tid==0)t=reuse_clock();}
      mbar_wait(&fullb[st],ph);
      if constexpr(TRACE){if(tid==0)clocks[2]+=reuse_clock()-t;}
      float acc[MAX_BLOCKS][4];
#pragma unroll
      for(int j=0;j<MAX_BLOCKS;++j)for(int k=0;k<4;++k)acc[j][k]=0.f;
      const uint8_t* tile=stages+st*TILE_BYTES;
      // REUSE_MATH
      __syncwarp();if(lane==0)mbar_arrive(&emptyb[st]);if(++st==nstage){st=0;ph^=1;}
      if constexpr(TRACE){if(tid==0)t=reuse_clock();}
#pragma unroll
      for(int j=0;j<MAX_BLOCKS;++j)if(j<nb){
        *reinterpret_cast<float4*>(red+((buf*NCW+warp)*32+lane)*4)=make_float4(acc[j][0],acc[j][1],acc[j][2],acc[j][3]);
        bar_consumers();
        if(j==0&&tid==0)reuse_prefetch(rr,nstage,stages,fullb,emptyb,w13p,w2p);
        const float* rb=red+buf*NCW*32*4;
        if(!pass){if(tid<64){int i=tid&7,n=tid>>3,ls=i*4+(n>>1),rg=n&1;float g=0.f,up=0.f;
#pragma unroll
          for(int w=0;w<NCW;++w){g+=rb[(w*32+ls)*4+rg];up+=rb[(w*32+ls)*4+rg+2];}
          int idx=j*8+n;if(idx<ne){if(m==0&&i==0)ss[off+idx]=0.f;g*=gs13[e]*256.f;up*=gs13[e]*256.f;
            inter[(size_t)(off+idx)*HD+m*8+i]=__float2bfloat16(g/(1.f+__expf(-g))*up);}}}
        else if(tid<128){int i=tid&15,n=tid>>4,ls=(i&7)*4+(n>>1),rg=(n&1)+((i>>3)<<1);float v=0.f;
#pragma unroll
          for(int w=0;w<NCW;++w)v+=rb[(w*32+ls)*4+rg];v*=gs2[e]*256.f;
          int idx=j*8+n;bool ok=idx<ne;float z=v*(1.f+__bfloat162float(pnw[m*16+i]));
          if(ok)y[(size_t)(off+idx)*HD+m*16+i]=z;
          float t2=ok?z*z:0.f;
#pragma unroll
          for(int d=8;d;d>>=1)t2+=__shfl_xor_sync(~0u,t2,d);if(ok&&i==0)atomicAdd(ss+off+idx,t2);
        }
        buf^=1;
      }
      if constexpr(TRACE){if(tid==0){clocks[5]+=reuse_clock()-t;clocks[10+pass]++;clocks[12+pass]+=nb;}}
      ++seg_tiles;
    }
  }
  publish();
  uint64_t tb=0;if constexpr(TRACE){if(tid==0)tb=reuse_clock();}
  __threadfence();bar_consumers();
  if(tid==0){if(atomicAdd(cnt+C_BAR,1)==G-1){for(int i=0;i<NLOC;++i)cnt[C_DONE+i]=0;cnt[C_BAR]=0;
      __threadfence();atomicAdd(cnt+C_GEN,1);}else while(ld_acquire(cnt+C_GEN)==gen0)__nanosleep(64);}
  bar_consumers();if constexpr(TRACE){if(tid==0)clocks[8]=reuse_clock()-tb;}
  for(int it=c;it<2*M;it+=G){int t=it>>1,c0=(it&1)*2048+tid*8;float acc[8]={};
#pragma unroll
    for(int j=0;j<8;++j){int pos=pos_of[t*8+j];if(pos>=0){float sc=tw[t*8+j]*rsqrtf(__ldcg(ss+pos)*(1.f/HD)+1e-8f);
      const float4* p=reinterpret_cast<const float4*>(y+(size_t)pos*HD+c0);float4 a=__ldcg(p),b=__ldcg(p+1);
      acc[0]+=sc*a.x;acc[1]+=sc*a.y;acc[2]+=sc*a.z;acc[3]+=sc*a.w;acc[4]+=sc*b.x;acc[5]+=sc*b.y;acc[6]+=sc*b.z;acc[7]+=sc*b.w;}}
    uint4 ov;auto* ob=reinterpret_cast<__nv_bfloat16*>(&ov);
#pragma unroll
    for(int k=0;k<8;++k)ob[k]=__float2bfloat16(acc[k]);*reinterpret_cast<uint4*>(out+(size_t)t*HD+c0)=ov;
  }
  if constexpr(TRACE){if(tid==0){clocks[9]=reuse_clock();for(int k=0;k<14;++k)trace[c*16+k]=clocks[k];}}
}
size_t reuse_smem(int nstage,int M){return (size_t)nstage*TILE_BYTES+(2*NCW*32*4+8*256)*sizeof(float)+2*nstage*8+sizeof(ReuseRoute)+((M*8+15)&~15)+4*M*8;}
#define REUSE_INST(X) X(false,false) X(true,false) X(false,true) X(true,true)
} // namespace
int64_t reuse_init(){int dev,opt;C10_CUDA_CHECK(cudaGetDevice(&dev));C10_CUDA_CHECK(cudaDeviceGetAttribute(&opt,cudaDevAttrMaxSharedMemoryPerBlockOptin,dev));
#define ATTR(R,T) {cudaFuncAttributes a{};C10_CUDA_CHECK(cudaFuncGetAttributes(&a,moe_reuse<R,T>));C10_CUDA_CHECK(cudaFuncSetAttribute(moe_reuse<R,T>,cudaFuncAttributeMaxDynamicSharedMemorySize,opt-a.sharedSizeBytes));}
 REUSE_INST(ATTR)
#undef ATTR
 return opt;}
std::vector<int64_t> reuse_info(int64_t redo,int64_t trace,int64_t stages,int64_t M){int resident=0;cudaFuncAttributes a{};
#define INFO(R,T) if(redo==R&&trace==T){C10_CUDA_CHECK(cudaFuncGetAttributes(&a,moe_reuse<R,T>));C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,moe_reuse<R,T>,256,reuse_smem(stages,M)));}
 REUSE_INST(INFO)
#undef INFO
 return {resident,a.numRegs,(int64_t)a.localSizeBytes,(int64_t)reuse_smem(stages,M)};}
void reuse_run(std::vector<int64_t>v,int64_t M,int64_t start,int64_t stages,int64_t grid,int64_t stream,int64_t redo,int64_t trace){
 if(M>32)throw std::runtime_error("reuse: at most 32 tokens");
 cudaLaunchConfig_t cfg{};cfg.gridDim=dim3(grid);cfg.blockDim=dim3(256);cfg.dynamicSmemBytes=reuse_smem(stages,M);cfg.stream=(cudaStream_t)stream;
 cudaLaunchAttribute attr{};attr.id=cudaLaunchAttributeProgrammaticStreamSerialization;attr.val.programmaticStreamSerializationAllowed=1;cfg.attrs=&attr;cfg.numAttrs=1;
#define LAUNCH(R,T) if(redo==R&&trace==T){C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg,moe_reuse<R,T>,(const __nv_bfloat16*)v[0],(const int*)v[1],(const float*)v[2],(const __nv_bfloat16*)v[3],(const uint8_t*)v[4],(const uint8_t*)v[5],(const float*)v[6],(const float*)v[7],(__nv_bfloat16*)v[8],(float*)v[9],(float*)v[10],(int*)v[11],(__nv_bfloat16*)v[12],(int)M,(int)start,(int)stages,(uint64_t*)v[13]));return;}
 REUSE_INST(LAUNCH)
#undef LAUNCH
 throw std::runtime_error("reuse flags");}
