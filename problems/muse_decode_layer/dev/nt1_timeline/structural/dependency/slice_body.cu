// Producer-owned intermediate slices. MODE0=atomic, MODE1=separate partials,
// MODE2=timing-only no-reduction checksum (never a candidate).
template<int S>constexpr int slice_smem(){return 2*TILE_BYTES+8192+32+sizeof(Route)+256+512+8*S*2;}
template<int S,int MODE>
__global__ void __launch_bounds__(256,2) slice_main(
 const __nv_bfloat16* x,const int* ids,const uint8_t* w13p,const uint8_t* w2slice,
 const float* gs13,float* partial,float* checksum,int M,int start){
 constexpr int NS=4096/S,GP=S/64,W2W=8*GP*512,W2B=W2W+8*32*GP*2;
 const int tid=threadIdx.x,lane=tid&31,warp=tid>>5,q=lane&3,npairs=M*8,G=gridDim.x;
 extern __shared__ __align__(128) uint8_t raw[];
 uint8_t* stages=raw;float* red=reinterpret_cast<float*>(raw+2*TILE_BYTES);
 uint64_t* full=reinterpret_cast<uint64_t*>(red+2048);uint64_t* empty=full+2;
 Route* rt=reinterpret_cast<Route*>(empty+2);int8_t* ef=reinterpret_cast<int8_t*>(rt+1);
 int16_t* pairs=reinterpret_cast<int16_t*>(ef+256);__nv_bfloat16* mid=reinterpret_cast<__nv_bfloat16*>(pairs+256);
 uint64_t policy;
 asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0,1.0;":"=l"(policy));
 if(tid<32)rt->n[tid]=0;
 asm volatile("griddepcontrol.wait;":::"memory");
 __syncthreads();
 if(tid<npairs){int e=__ldg(ids+tid)-start;if(e<0||e>=32)e=-1;ef[tid]=e;if(e>=0)atomicAdd(rt->n+e,1);}
 __syncthreads();
 if(warp==0){int n=rt->n[lane],off=n,u=((n+7)/8)*512;
#pragma unroll
 for(int d=1;d<32;d*=2){int a=__shfl_up_sync(~0u,off,d),b=__shfl_up_sync(~0u,u,d);if(lane>=d){off+=a;u+=b;}}
 rt->off[lane]=off-n;rt->run[lane]=off-n;rt->nblk[lane]=(n+7)/8;rt->us1[lane+1]=u;if(lane==0)rt->us1[0]=0;}
 __syncthreads();
 for(int base=0;base<npairs;base+=256){rt->wc[warp][lane]=0;__syncthreads();
 int i=base+tid,e=i<npairs?ef[i]:-1;unsigned mask=__match_any_sync(~0u,e);int rank=__popc(mask&((1u<<lane)-1));
 if(e>=0&&rank==0)rt->wc[warp][e]=__popc(mask);__syncthreads();
 if(tid<32){int p=rt->run[tid];for(int w=0;w<8;++w){int n=rt->wc[w][tid];rt->wc[w][tid]=p;p+=n;}rt->run[tid]=p;}
 __syncthreads();if(i<npairs&&e>=0)pairs[rt->wc[warp][e]+rank]=i;}
 __syncthreads();
 const int total=rt->us1[32]/(S/8);
 float chk=0.f;
 for(int unit=blockIdx.x;unit<total;unit+=G){
  int e,nb,m0;decode_unit(unit*(S/8),rt->us1,rt->nblk,S/8,512,e,nb,m0);
  const int ne=rt->n[e],off=rt->off[e],slice=m0/(S/8);
  { // W13: same eight K-warps and exact BF16 SwiGLU rounding as S19.
   uint32_t Bf[8][8];const int rowid=nb*8+(lane>>2);
#pragma unroll
   for(int g=0;g<8;++g){uint4 a=make_uint4(0,0,0,0),b=a;
    if(rowid<ne){const auto* p=reinterpret_cast<const uint4*>(x+(size_t)(pairs[off+rowid]>>3)*4096+warp*512+g*64+q*16);a=__ldg(p);b=__ldg(p+1);}
    Bf[g][0]=a.x;Bf[g][1]=a.y;Bf[g][2]=a.z;Bf[g][3]=a.w;Bf[g][4]=b.x;Bf[g][5]=b.y;Bf[g][6]=b.z;Bf[g][7]=b.w;}
   if(tid==0){for(int s=0;s<2;++s){mbar_init(full+s,1);mbar_init(empty+s,8);}asm volatile("fence.mbarrier_init.release.cluster;":::"memory");}
   __syncthreads();
   if(tid==224){for(int s=0;s<2;++s){mbar_wait(empty+s,1);mbar_expect_tx(full+s,TILE_BYTES);bulk_g2s(stages+s*TILE_BYTES,w13p+((size_t)e*512+m0+s)*TILE_BYTES,TILE_BYTES,full+s,policy);}}
   for(int mi=0;mi<S/8;++mi){int st=mi&1,ph=(mi>>1)&1; mbar_wait(full+st,ph);
    const uint8_t* tile=stages+st*TILE_BYTES;const uint4* wq=reinterpret_cast<const uint4*>(tile)+warp*256+lane;
    uint4 sv=*(reinterpret_cast<const uint4*>(tile+32768)+warp*32+lane);uint32_t sc[4]={sv.x,sv.y,sv.z,sv.w};float acc[4]={0,0,0,0};
#pragma unroll
    for(int g=0;g<8;++g){uint4 wv=wq[g*32];uint32_t sw=sc[g>>1];uint32_t s0=__byte_perm(sw,0u,g&1?0x4242u:0x4040u)*16u+0x70007000u,s8=__byte_perm(sw,0u,g&1?0x4343u:0x4141u)*16u+0x70007000u;
     uint32_t o[16];dq8(wv.x,wv.y,o);dq8(wv.z,wv.w,o+8);
#pragma unroll
     for(int k=0;k<4;++k){uint32_t* a=o+k*4;mma16816(acc,hmul2(a[0],s0),hmul2(a[1],s8),hmul2(a[2],s0),hmul2(a[3],s8),Bf[g][2*k],Bf[g][2*k+1]);}}
    __syncwarp();if(lane==0)mbar_arrive(empty+st);
    float* p=red+((mi&1)*8+warp)*128;
    *reinterpret_cast<float2*>(p+lane*2)=make_float2(acc[0],acc[2]);*reinterpret_cast<float2*>(p+64+lane*2)=make_float2(acc[1],acc[3]);
    __syncthreads();
    if(tid==224&&mi+2<S/8){int next=mi+2;int st2=next&1,ph2=(next>>1)&1;mbar_wait(empty+st2,ph2^1);mbar_expect_tx(full+st2,TILE_BYTES);bulk_g2s(stages+st2*TILE_BYTES,w13p+((size_t)e*512+m0+next)*TILE_BYTES,TILE_BYTES,full+st2,policy);}
    if(warp<2){const float* rb=red+(mi&1)*1024;float a=0,b=0;
#pragma unroll
     for(int w=0;w<8;++w){float2 v=*reinterpret_cast<const float2*>(rb+w*128+warp*64+lane*2);a+=v.x;b+=v.y;}
     int i=lane>>2,n=(lane&3)*2+warp;float scale=gs13[e]*256.f;float gate=a*scale,up=b*scale;
     if(nb*8+n<ne)mid[n*S+mi*8+i]=__float2bfloat16(gate/(1.f+__expf(-gate))*up);}
   }
  }
  __syncthreads(); // Own intermediate slice is now ready; no expert/global wait.
  { // W2: each warp owns an output tile, K is exactly its own intermediate slice.
   uint32_t Bf[GP][8];
#pragma unroll
   for(int g=0;g<GP;++g){uint4 a=make_uint4(0,0,0,0),b=a;
    if(nb*8+(lane>>2)<ne){const auto* p=reinterpret_cast<const uint4*>(mid+(lane>>2)*S+g*64+q*16);a=*p;b=*(p+1);}
    Bf[g][0]=a.x;Bf[g][1]=a.y;Bf[g][2]=a.z;Bf[g][3]=a.w;Bf[g][4]=b.x;Bf[g][5]=b.y;Bf[g][6]=b.z;Bf[g][7]=b.w;}
   if(tid==0){for(int s=0;s<2;++s){mbar_init(full+s,1);mbar_init(empty+s,8);}asm volatile("fence.mbarrier_init.release.cluster;":::"memory");}
   __syncthreads();const uint8_t* base=w2slice+((size_t)e*NS+slice)*32*W2B;
   if(tid==224){for(int s=0;s<2;++s){mbar_wait(empty+s,1);mbar_expect_tx(full+s,W2B);bulk_g2s(stages+s*W2B,base+s*W2B,W2B,full+s,policy);}}
   for(int it=0;it<32;++it){int st=it&1,ph=(it>>1)&1;mbar_wait(full+st,ph);const uint8_t* tile=stages+st*W2B;float acc[4]={0,0,0,0};
#pragma unroll
    for(int g=0;g<GP;++g){const uint4 wv=*(reinterpret_cast<const uint4*>(tile)+(warp*GP+g)*32+lane);const uint16_t sw=*(reinterpret_cast<const uint16_t*>(tile+W2W)+(warp*32+lane)*GP+g);const uint32_t s0=__byte_perm(sw,0u,0x4040u)*16u+0x70007000u,s8=__byte_perm(sw,0u,0x4141u)*16u+0x70007000u;
     uint32_t o[16];dq8(wv.x,wv.y,o);dq8(wv.z,wv.w,o+8);
#pragma unroll
     for(int k=0;k<4;++k){uint32_t* a=o+k*4;mma16816(acc,hmul2(a[0],s0),hmul2(a[1],s8),hmul2(a[2],s0),hmul2(a[3],s8),Bf[g][2*k],Bf[g][2*k+1]);}}
    __syncwarp();if(lane==0)mbar_arrive(empty+st);
    // No shared reduction: these warps own disjoint output tiles.
    int n=(lane&3)*2,col=(it*8+warp)*16+(lane>>2);
#pragma unroll
    for(int j=0;j<2;++j){if(nb*8+n+j<ne){int flat=pairs[off+nb*8+n+j];size_t pos=(size_t)flat*4096+col;
     if constexpr(MODE==0){asm volatile("red.relaxed.gpu.global.add.f32 [%0],%1;"::"l"(partial+pos),"f"(acc[j]):"memory");asm volatile("red.relaxed.gpu.global.add.f32 [%0],%1;"::"l"(partial+pos+8),"f"(acc[j+2]):"memory");}
     else if constexpr(MODE==1){partial[(size_t)slice*npairs*4096+pos]=acc[j];partial[(size_t)slice*npairs*4096+pos+8]=acc[j+2];}
     else {chk+=acc[j]+acc[j+2];}
    }}
    if(tid==224&&it+2<32){int next=it+2;int st2=next&1,ph2=(next>>1)&1;mbar_wait(empty+st2,ph2^1);mbar_expect_tx(full+st2,W2B);bulk_g2s(stages+st2*W2B,base+next*W2B,W2B,full+st2,policy);}
   }
  }
  __syncthreads(); // All local readers finished before scratch is reused for next unit.
 }
 if constexpr(MODE==2)checksum[(size_t)blockIdx.x*256+tid]=chk;
}
__global__ void slice_zero(float* p,int n){for(int i=blockIdx.x*256+threadIdx.x;i<n/4;i+=gridDim.x*256)reinterpret_cast<float4*>(p)[i]=make_float4(0,0,0,0);}
template<int S,int MODE>__global__ void slice_norm(const float* partial,float* z,float* norm,const int* ids,const __nv_bfloat16* pnw,const float* gs2,int M,int start){
 int flat=blockIdx.x,e=ids[flat]-start,tid=threadIdx.x;__shared__ float sums[8];
 if(e<0||e>=32){if(tid==0)norm[flat]=0;return;}
 float sum=0,scale=gs2[e]*256.f;
 for(int col=tid*4;col<4096;col+=1024){float4 value=make_float4(0,0,0,0);
  if constexpr(MODE==0)value=*reinterpret_cast<const float4*>(partial+(size_t)flat*4096+col);
  else {
#pragma unroll
   for(int s=0;s<4096/S;++s){float4 v=*reinterpret_cast<const float4*>(partial+((size_t)s*M*8+flat)*4096+col);value.x+=v.x;value.y+=v.y;value.z+=v.z;value.w+=v.w;}}
  const uint2 ww=*reinterpret_cast<const uint2*>(pnw+col);const __nv_bfloat16* w=reinterpret_cast<const __nv_bfloat16*>(&ww);float* v=reinterpret_cast<float*>(&value);
#pragma unroll
  for(int k=0;k<4;++k){v[k]*=scale;v[k]*=1.f+__bfloat162float(w[k]);sum+=v[k]*v[k];}
  *reinterpret_cast<float4*>(z+(size_t)flat*4096+col)=value;}
 for(int d=16;d;d/=2)sum+=__shfl_down_sync(~0u,sum,d);if((tid&31)==0)sums[tid>>5]=sum;__syncthreads();
 if(tid<32){sum=tid<8?sums[tid]:0.f;for(int d=16;d;d/=2)sum+=__shfl_down_sync(~0u,sum,d);if(tid==0)norm[flat]=rsqrtf(sum*(1.f/4096)+1e-8f);}}
__global__ void slice_combine(const float* z,const float* norm,const float* tw,const int* ids,__nv_bfloat16* out,int M,int start){
 int t=blockIdx.x,tid=threadIdx.x;
 for(int col=tid*4;col<4096;col+=1024){float acc[4]={0,0,0,0};
#pragma unroll
 for(int j=0;j<8;++j){int p=t*8+j,e=ids[p]-start;if(e>=0&&e<32){float4 v=*reinterpret_cast<const float4*>(z+(size_t)p*4096+col);float sc=tw[p]*norm[p];acc[0]+=v.x*sc;acc[1]+=v.y*sc;acc[2]+=v.z*sc;acc[3]+=v.w*sc;}}
 uint2 value;auto* v=reinterpret_cast<__nv_bfloat16*>(&value);
#pragma unroll
 for(int k=0;k<4;++k)v[k]=__float2bfloat16(acc[k]);*reinterpret_cast<uint2*>(out+(size_t)t*4096+col)=value;}}
} // anonymous namespace

template<int S,int MODE>void dep_launch(std::vector<int64_t>&v,int M,int start,int grid,cudaStream_t stream){
 if constexpr(MODE==0)slice_zero<<<std::min(1024,M*32),256,0,stream>>>((float*)v[8],M*8*4096);
 cudaLaunchConfig_t cfg{};cfg.gridDim=dim3(grid);cfg.blockDim=dim3(256);cfg.dynamicSmemBytes=slice_smem<S>();cfg.stream=stream;cudaLaunchAttribute attr{};attr.id=cudaLaunchAttributeProgrammaticStreamSerialization;attr.val.programmaticStreamSerializationAllowed=1;cfg.attrs=&attr;cfg.numAttrs=1;
 C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg,slice_main<S,MODE>,(const __nv_bfloat16*)v[0],(const int*)v[1],(const uint8_t*)v[4],(const uint8_t*)v[5],(const float*)v[6],(float*)v[8],(float*)v[12],M,start));
 if constexpr(MODE!=2){slice_norm<S,MODE><<<M*8,256,0,stream>>>((const float*)v[8],(float*)v[9],(float*)v[10],(const int*)v[1],(const __nv_bfloat16*)v[3],(const float*)v[7],M,start);slice_combine<<<M,256,0,stream>>>((const float*)v[9],(const float*)v[10],(const float*)v[2],(const int*)v[1],(__nv_bfloat16*)v[11],M,start);}C10_CUDA_KERNEL_LAUNCH_CHECK();}
void dep_run(std::vector<int64_t> v,int M,int start,int s,int mode,int grid,int64_t stream){
#define GO(S,K) if(s==S&&mode==K){dep_launch<S,K>(v,M,start,grid,(cudaStream_t)stream);return;}
 GO(64,0) GO(128,0) GO(128,1) GO(128,2)
#undef GO
 throw std::runtime_error("unknown slice arm");}
std::vector<int64_t> dep_info(int s,int mode){cudaFuncAttributes a{};int n=0,sm=0;
#define INFO(S,K) if(s==S&&mode==K){sm=slice_smem<S>();C10_CUDA_CHECK(cudaFuncGetAttributes(&a,slice_main<S,K>));C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,slice_main<S,K>,256,sm));}
 INFO(64,0) INFO(128,0) INFO(128,1) INFO(128,2)
#undef INFO
 return{n,a.numRegs,(int64_t)a.localSizeBytes,sm,(int64_t)a.sharedSizeBytes};}
void dep_init(){int dev,opt;C10_CUDA_CHECK(cudaGetDevice(&dev));C10_CUDA_CHECK(cudaDeviceGetAttribute(&opt,cudaDevAttrMaxSharedMemoryPerBlockOptin,dev));
#define ATTR(S,K) C10_CUDA_CHECK(cudaFuncSetAttribute(slice_main<S,K>,cudaFuncAttributeMaxDynamicSharedMemorySize,opt));
 ATTR(64,0) ATTR(128,0) ATTR(128,1) ATTR(128,2)
#undef ATTR
}
