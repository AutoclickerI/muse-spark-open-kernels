// Hopper RS WGMMA, 64 output rows, 8/16/32 token columns selected per expert. No K-split reduction.
// Same packed NT1 code bytes; scale pairs are reordered losslessly in prepare, decoded only here.
constexpr int BK=64;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)
#define BR_HOPPER 1
#else
#define BR_HOPPER 0
#endif
__device__ __forceinline__ unsigned smaddr(const void*p){return smem_u32(p);}
// WG_MMA_HELPERS
__device__ __forceinline__ void wg_keep(const unsigned*a){asm volatile(""::"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(a[4]),"r"(a[5]),"r"(a[6]),"r"(a[7]),"r"(a[8]),"r"(a[9]),"r"(a[10]),"r"(a[11]),"r"(a[12]),"r"(a[13]),"r"(a[14]),"r"(a[15]));}
struct WGRoute{int n[32],off[32],run[32],us1[33],us2[33],wc[4][32];};
template<int S>constexpr int wg_smem(int M){return S*(2048+512+32*64*2)+sizeof(WGRoute)+((M*8+15)&~15)+4*M*8+4*32*4;}
__device__ __forceinline__ void wg_cp(void*d,const void*s,int bytes,int valid){unsigned dst=smem_u32(d);
 if(bytes==16)asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;"::"r"(dst),"l"(s),"r"(valid?16:0):"memory");
 else asm volatile("cp.async.ca.shared.global [%0],[%1],4,%2;"::"r"(dst),"l"(s),"r"(valid?4:0):"memory");}

template<int S>__global__ void __launch_bounds__(128,3) moe_wg(
 const __nv_bfloat16* x,const int* ids,const float* tw,const __nv_bfloat16* pnw,
 const uint8_t* w13,const uint8_t* w2,const uint16_t* sf13,const uint16_t* sf2,
 const float* gs13,const float* gs2,__nv_bfloat16* inter,float* y,float* ss,int* cnt,__nv_bfloat16* out,int M,int start){
 extern __shared__ __align__(1024) uint8_t sm[];
 auto* qs=sm;auto* sc=qs+S*2048;auto* bs=reinterpret_cast<__nv_bfloat16*>(sc+S*512);
 auto* rt=reinterpret_cast<WGRoute*>(bs+S*32*64);auto* ef=reinterpret_cast<int8_t*>(rt+1);
 auto* pos=reinterpret_cast<int16_t*>(ef+((M*8+15)&~15));auto* pairs=pos+M*8;auto* rowsum=reinterpret_cast<float*>(pairs+M*8);
 int t=threadIdx.x,l=t&31,w=t>>5,c=blockIdx.x,G=gridDim.x,gen=0;
 if(t<32)rt->n[t]=0;
 asm volatile("griddepcontrol.wait;":::"memory");
 if(t==0)gen=ld_acquire(cnt+C_GEN);__syncthreads();
 for(int i=t;i<M*8;i+=128){int e=__ldg(ids+i)-start;if(e<0||e>=32)e=-1;ef[i]=e;if(e>=0)atomicAdd(rt->n+e,1);}
 __syncthreads();
 if(w==0){int n=rt->n[l],off=n,un=n>0;
#pragma unroll
  for(int d=1;d<32;d<<=1){int a=__shfl_up_sync(~0u,off,d),b=__shfl_up_sync(~0u,un,d);if(l>=d){off+=a;un+=b;}}
  rt->off[l]=off-n;rt->run[l]=off-n;rt->us1[l+1]=un*128;rt->us2[l+1]=un*64;if(l==0){rt->us1[0]=0;rt->us2[0]=0;}}
 __syncthreads();
 for(int base=0;base<M*8;base+=128){rt->wc[w][l]=0;__syncthreads();int i=base+t,e=i<M*8?ef[i]:-1;
  unsigned mk=__match_any_sync(~0u,e);int rank=__popc(mk&((1u<<l)-1u));if(e>=0&&rank==0)rt->wc[w][e]=__popc(mk);
  __syncthreads();if(t<32){int s=rt->run[t];for(int ww=0;ww<4;++ww){int n=rt->wc[ww][t];rt->wc[ww][t]=s;s+=n;}rt->run[t]=s;}
  __syncthreads();if(i<M*8){if(e>=0){int p=rt->wc[w][e]+rank;pos[i]=p;pairs[p]=i;}else pos[i]=-1;}}
 __syncthreads();
 int cur_e=-1,cur_pass=-1,done=0;
 auto publish=[&](){if(cur_pass==0&&done){__threadfence();__syncthreads();if(t==0)atomicAdd(cnt+C_DONE+cur_e,done);}};
 for(int pass=0;pass<2;++pass){const int*us=pass?rt->us2:rt->us1;int U=us[32],a=(long long)U*c/G,b=(long long)U*(c+1)/G;
  for(int unit=a;unit<b;++unit){int e=0;
#pragma unroll
   for(int d=16;d;d>>=1)if(us[e+d]<=unit)e+=d;
   int mt=unit-us[e],ne=rt->n[e],off=rt->off[e],ntile=pass?256:512;
   if(e!=cur_e||pass!=cur_pass){publish();cur_e=e;cur_pass=pass;done=0;if(pass)while(ld_acquire(cnt+C_DONE+e)<128)__nanosleep(64);}
   auto tile=[&]<int NT>(){
   const uint8_t* weights=(pass?w2:w13)+(static_cast<size_t>(e)*ntile+mt*4+w)*TILE_BYTES;
   const uint16_t* scales=(pass?sf2:sf13)+(static_cast<size_t>(e)*ntile+mt*4+w)*64*32;
   const __nv_bfloat16* xr[NT/4];bool xv[NT/4];
#pragma unroll
   for(int j=0;j<NT/4;++j){int r=t/32+j*4;xv[j]=r<ne;int p=xv[j]?off+r:off;
    xr[j]=pass?inter+(size_t)p*4096:x+(size_t)(xv[j]?(pairs[p]>>3):0)*4096;}
   auto issue=[&](int slot,int ki){
    const uint8_t* wp=weights+((ki>>3)*8+(ki&7))*512+l*16;
    wg_cp(qs+slot*2048+t*16,wp,16,1);
    wg_cp(sc+slot*512+t*4,scales+ki*32+(l&~1),4,1);
#pragma unroll
    for(int j=0;j<NT/4;++j){int r=t/32+j*4,k=(t&31)*2;
     // S11 register-A fragments use this intra-K64 permutation. Preserve it in B shared storage.
     int actual=(t&3)*16+((t&31)>>3)*4+(((t&31)>>2)&1)*2;
     wg_cp(bs+slot*32*64+r*64+(k^((r&7)*8)),xr[j]+ki*64+actual,4,xv[j]);}
    asm volatile("cp.async.commit_group;":::"memory");
   };
#pragma unroll
   for(int st=0;st<S-1;++st)issue(st,st);
   float acc[NT/2]={};unsigned af[2][16]={};
#pragma unroll 1
   for(int kbase=0;kbase<64;kbase+=2){
#pragma unroll
   for(int half=0;half<2;++half){int ki=kbase+half,st=ki%S;
    asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");__syncthreads();
#if BR_HOPPER
    asm volatile("fence.proxy.async.shared::cta;":::"memory");
#endif
    uint4 q=*reinterpret_cast<uint4*>(qs+st*2048+t*16);uint32_t sw=*reinterpret_cast<uint32_t*>(sc+st*512+t*4);sw>>=(l&1)*16;
    uint32_t sr=__byte_perm(sw,0u,0x4040u)*16u+0x70007000u,sr8=__byte_perm(sw,0u,0x4141u)*16u+0x70007000u;
    uint32_t o[16];dq8(q.x,q.y,o);dq8(q.z,q.w,o+8);unsigned* ar=af[half];
#pragma unroll
    for(int z=0;z<4;++z){int k=(z>>1)*8+(z&1)*4;ar[z*4]=hmul2(o[k],sr);ar[z*4+1]=hmul2(o[k+1],sr8);ar[z*4+2]=hmul2(o[k+2],sr);ar[z*4+3]=hmul2(o[k+3],sr8);}
    wg_keep(ar);
#if BR_HOPPER
    asm volatile("wgmma.fence.sync.aligned;":::"memory");
#endif
    mma_group<NT>(acc,ar,bs+st*32*64);
    wg_keep(af[half^1]);
    if(ki+S-1<64)issue((ki+S-1)%S,ki+S-1);else asm volatile("cp.async.commit_group;":::"memory");
   }}
#if BR_HOPPER
   asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");
#endif
   wg_keep(af[0]);wg_keep(af[1]);
   if(!pass){int ch=mt*32+w*8+(l>>2);float scale=gs13[e]*256.f;
#pragma unroll
    for(int j=0;j<NT/2;j+=4){int r=(l&3)*2+(j>>2)*8;
     float g=acc[j]*scale,u=acc[j+2]*scale;
     if(r<ne)inter[(size_t)(off+r)*4096+ch]=__float2bfloat16(g/(1.f+__expf(-g))*u);
     g=acc[j+1]*scale;u=acc[j+3]*scale;
     if(r+1<ne)inter[(size_t)(off+r+1)*4096+ch]=__float2bfloat16(g/(1.f+__expf(-g))*u);
    }
    if(mt==0&&t<ne)ss[off+t]=0.f;
   }else{float scale=gs2[e]*256.f;
#pragma unroll
    for(int j=0;j<NT/2;++j){int r=(l&3)*2+(j&1)+(j>>2)*8,col=mt*64+w*16+(l>>2)+((j>>1)&1)*8;
     float z=acc[j]*scale*(1.f+__bfloat162float(pnw[col]));acc[j]=z;if(r<ne)y[(size_t)(off+r)*4096+col]=z;}
#pragma unroll
    for(int j=0;j<NT/8;++j){float a=acc[j*4]*acc[j*4]+acc[j*4+2]*acc[j*4+2],b=acc[j*4+1]*acc[j*4+1]+acc[j*4+3]*acc[j*4+3];
#pragma unroll
     for(int d=4;d<=16;d*=2){a+=__shfl_xor_sync(~0u,a,d);b+=__shfl_xor_sync(~0u,b,d);}
     if(l<4){rowsum[w*32+j*8+l*2]=a;rowsum[w*32+j*8+l*2+1]=b;}}
    __syncthreads();if(t<ne){float s=0.f;
#pragma unroll
     for(int ww=0;ww<4;++ww)s+=rowsum[ww*32+t];atomicAdd(ss+off+t,s);}
   }
   };
   if(ne<=8)tile.template operator()<8>();else if(ne<=16)tile.template operator()<16>();else tile.template operator()<32>();
   __syncthreads();++done;
  }
 }
 publish();__threadfence();__syncthreads();
 if(t==0){if(atomicAdd(cnt+C_BAR,1)==G-1){for(int e=0;e<32;++e)cnt[e]=0;cnt[C_BAR]=0;__threadfence();atomicAdd(cnt+C_GEN,1);}else while(ld_acquire(cnt+C_GEN)==gen)__nanosleep(64);}
 __syncthreads();
 for(int it=c;it<4*M;it+=G){int token=it/4,col=(it%4)*1024+t*8;float acc[8]={};
#pragma unroll
  for(int j=0;j<8;++j){int p=pos[token*8+j];if(p>=0){float scale=tw[token*8+j]*rsqrtf(__ldcg(ss+p)*(1.f/4096)+1e-8f);const float4* src=reinterpret_cast<const float4*>(y+(size_t)p*4096+col);float4 a=__ldcg(src),b=__ldcg(src+1);
   acc[0]+=scale*a.x;acc[1]+=scale*a.y;acc[2]+=scale*a.z;acc[3]+=scale*a.w;acc[4]+=scale*b.x;acc[5]+=scale*b.y;acc[6]+=scale*b.z;acc[7]+=scale*b.w;}}
  uint4 ov;auto* ob=reinterpret_cast<__nv_bfloat16*>(&ov);
#pragma unroll
  for(int k=0;k<8;++k)ob[k]=__float2bfloat16(acc[k]);*reinterpret_cast<uint4*>(out+(size_t)token*4096+col)=ov;
 }
}
} // namespace
int64_t wg_init(){int d,opt;C10_CUDA_CHECK(cudaGetDevice(&d));C10_CUDA_CHECK(cudaDeviceGetAttribute(&opt,cudaDevAttrMaxSharedMemoryPerBlockOptin,d));
#define ATTR(S) {cudaFuncAttributes a{};C10_CUDA_CHECK(cudaFuncGetAttributes(&a,moe_wg<S>));C10_CUDA_CHECK(cudaFuncSetAttribute(moe_wg<S>,cudaFuncAttributeMaxDynamicSharedMemorySize,opt-a.sharedSizeBytes));}
 ATTR(4) ATTR(8)
#undef ATTR
 return opt;}
std::vector<int64_t> wg_info(int64_t stages,int64_t M){int resident=0;cudaFuncAttributes a{};int smem=0;
#define INFO(S) if(stages==S){smem=wg_smem<S>(M);C10_CUDA_CHECK(cudaFuncGetAttributes(&a,moe_wg<S>));C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&resident,moe_wg<S>,128,smem));}
 INFO(4) INFO(8)
#undef INFO
 return {resident,a.numRegs,(int64_t)a.localSizeBytes,smem};}
void wg_run(std::vector<int64_t>v,int64_t M,int64_t start,int64_t stages,int64_t grid,int64_t stream){
 if(M>32)throw std::runtime_error("wg M>32");cudaLaunchConfig_t cfg{};cfg.gridDim=dim3(grid);cfg.blockDim=dim3(128);cfg.stream=(cudaStream_t)stream;
 cudaLaunchAttribute attr{};attr.id=cudaLaunchAttributeProgrammaticStreamSerialization;attr.val.programmaticStreamSerializationAllowed=1;cfg.attrs=&attr;cfg.numAttrs=1;
#define LAUNCH(S) if(stages==S){cfg.dynamicSmemBytes=wg_smem<S>(M);C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg,moe_wg<S>,(const __nv_bfloat16*)v[0],(const int*)v[1],(const float*)v[2],(const __nv_bfloat16*)v[3],(const uint8_t*)v[4],(const uint8_t*)v[5],(const uint16_t*)v[6],(const uint16_t*)v[7],(const float*)v[8],(const float*)v[9],(__nv_bfloat16*)v[10],(float*)v[11],(float*)v[12],(int*)v[13],(__nv_bfloat16*)v[14],(int)M,(int)start));return;}
 LAUNCH(4) LAUNCH(8)
#undef LAUNCH
 throw std::runtime_error("wg stage depth");}
