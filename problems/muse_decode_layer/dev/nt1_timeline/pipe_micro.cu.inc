// head supplies the exact S14 decoder, MMA and TMA/mbarrier helpers.
template<int OP,int ILP> __global__ void pipe_ops(uint32_t* out,uint64_t* stamp,int reps){
 extern __shared__ __align__(128) uint8_t scratch[];uint32_t* sm=reinterpret_cast<uint32_t*>(scratch);
 int t=threadIdx.x,l=t&31,w=t>>5;for(int i=t;i<2048;i+=blockDim.x)sm[i]=OP==13?smem_u32(sm+(i+32)%2048):0x3f803f80u+i;
 __syncthreads();uint32_t u[ILP];float f[ILP];float ac[ILP][4];
#pragma unroll
 for(int i=0;i<ILP;++i){u[i]=OP==13?smem_u32(sm+(t+i*32)%2048):0x3f803f80u+t+i;f[i]=.25f+i*.01f;for(int j=0;j<4;++j)ac[i][j]=.001f*(i+1)*(j+1)+t*.00001f;}
 uint64_t g0=tg(),c0=tclk();
 for(int it=0;it<reps;++it){
#pragma unroll
  for(int z=0;z<8;++z){
#pragma unroll
   for(int i=0;i<ILP;++i){
    if constexpr(OP==0)asm volatile("lop3.b32 %0,%0,%1,%2,0x96;":"+r"(u[i]):"r"(0x13579bdfu),"r"(0x31415926u));
    if constexpr(OP==1)asm volatile("shf.r.wrap.b32 %0,%0,%0,5;":"+r"(u[i]));
    if constexpr(OP==2)asm volatile("prmt.b32 %0,%0,%1,0x5140;":"+r"(u[i]):"r"(0x3f803f80u));
    if constexpr(OP==3)u[i]=hmul2(u[i],0x3f803f80u+(t&1));
    if constexpr(OP==4)asm volatile("fma.rn.f32 %0,%0,0f3f7ff000,0f38000000;":"+f"(f[i]));
    if constexpr(OP==5)mma16816(ac[i],0x3b803b80u,0x3b803b80u,0x3b803b80u,0x3b803b80u,0x3b803b80u,0x3b803b80u);
    if constexpr(OP==6){uint4 v;asm volatile("ld.volatile.shared.v4.b32 {%0,%1,%2,%3},[%4];":"=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w):"r"(smem_u32(sm+w*128+l*4)):"memory");u[i]=v.x;}
    if constexpr(OP==7){asm volatile("ld.volatile.shared.u32 %0,[%1];":"=r"(u[i]):"r"(smem_u32(sm+w*32+l)):"memory");}
    if constexpr(OP==8){int ls=(l&7)*4+(l>>4),rg=(l>>3)&1;asm volatile("ld.volatile.shared.u32 %0,[%1];":"=r"(u[i]):"r"(smem_u32(sm+w*128+ls*4+rg)):"memory");}
    if constexpr(OP==9)u[i]=__shfl_xor_sync(~0u,u[i],4);
    if constexpr(OP==13)asm volatile("ld.volatile.shared.u32 %0,[%0];":"+r"(u[i])::"memory");
    if constexpr(OP==10)asm volatile("ex2.approx.f32 %0,%1;":"=f"(f[i]):"f"(-f[i]));
   }
  }
 }
 uint64_t c1=tclk(),g1=tg();if(t==0){auto p=stamp+blockIdx.x*8;p[0]=g0;p[1]=g1;p[2]=c0;p[3]=c1;p[4]=reps*8*ILP;p[5]=blockDim.x/32;uint32_t smid;asm("mov.u32 %0,%%smid;":"=r"(smid));p[6]=smid;}
 uint32_t v=0;for(int i=0;i<ILP;++i)v^=u[i]^__float_as_uint(f[i])^__float_as_uint(ac[i][0])^__float_as_uint(ac[i][1])^__float_as_uint(ac[i][2])^__float_as_uint(ac[i][3]);out[blockIdx.x*blockDim.x+t]=v;
}

template<int W,int GK,int D,int MAT,int HOT,int NB,int CHAINS> __global__ __launch_bounds__(W*32,NB>1?1:(W==4?4:2)) void pipe_work(
 const uint8_t* a,const uint8_t* b,const int* ids,int n,const __nv_bfloat16* x,int M,uint32_t* out,uint64_t* stamp,int iterations){
 constexpr int TW=32768,TB=36864;extern __shared__ __align__(128) uint8_t scratch[];
 auto full=reinterpret_cast<uint64_t*>(scratch+D*TB);int t=threadIdx.x,l=t&31,w=t>>5,c=blockIdx.x,G=gridDim.x;
 if(t==0){for(int j=0;j<D;++j)mbar_init(full+j,1);asm volatile("fence.mbarrier_init.release.cluster;":::"memory");}__syncthreads();
 int begin=n*c/G,end=n*(c+1)/G;uint64_t policy;asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;":"=l"(policy));
 auto copy=[&](int u,int s){int k=ids[u];if constexpr(HOT==1)k%=128;const uint8_t* p=k<16384?a+(size_t)k*36864:b+(size_t)(k-16384)*36864;
  mbar_expect_tx(full+s,TB);bulk_g2s(scratch+s*TB,p,TB,full+s,policy);};
 if(t==0){if constexpr(HOT==2){if(begin<end)copy(begin,0);}else for(int j=0;j<D&&begin+j<end;++j)copy(begin+j,j);}
 uint32_t bf[GK-(W==16)][8];uint32_t* bc=reinterpret_cast<uint32_t*>(full+D);
#pragma unroll
 for(int g=0;g<GK;++g){const uint4* p=reinterpret_cast<const uint4*>(x+(size_t)(l>>2)*4096+w*GK*64+g*64+(l&3)*16);uint4 v=__ldg(p),z=__ldg(p+1);
  if constexpr(W==16){if(g==GK-1)bc[0*W*32+t]=v.x;else bf[g][0]=v.x;}else bf[g][0]=v.x;if constexpr(W==16){if(g==GK-1)bc[1*W*32+t]=v.y;else bf[g][1]=v.y;}else bf[g][1]=v.y;if constexpr(W==16){if(g==GK-1)bc[2*W*32+t]=v.z;else bf[g][2]=v.z;}else bf[g][2]=v.z;if constexpr(W==16){if(g==GK-1)bc[3*W*32+t]=v.w;else bf[g][3]=v.w;}else bf[g][3]=v.w;if constexpr(W==16){if(g==GK-1)bc[4*W*32+t]=z.x;else bf[g][4]=z.x;}else bf[g][4]=z.x;if constexpr(W==16){if(g==GK-1)bc[5*W*32+t]=z.y;else bf[g][5]=z.y;}else bf[g][5]=z.y;if constexpr(W==16){if(g==GK-1)bc[6*W*32+t]=z.z;else bf[g][6]=z.z;}else bf[g][6]=z.z;if constexpr(W==16){if(g==GK-1)bc[7*W*32+t]=z.w;else bf[g][7]=z.w;}else bf[g][7]=z.w;}
 __syncthreads();uint64_t g0=tg(),c0=tclk(),wait=0,tiles=0;float acc[NB][4]={};uint32_t sink=0;
 int count=HOT==2?iterations:(end-begin);if constexpr(HOT==2){if(begin<end)mbar_wait(full,0);}
 for(int z=0;z<count;++z){int st=HOT==2?0:z%D;uint32_t ph=(z/D)&1u;
  uint64_t t0=tclk();if constexpr(HOT!=2)mbar_wait(full+st,ph);wait+=tclk()-t0;
  const auto tile=scratch+st*TB;
  if constexpr(MAT!=0){const uint4* wp=reinterpret_cast<const uint4*>(tile)+w*GK*32+l;uint4 scv;if constexpr(GK==8){asm volatile("ld.volatile.shared.v4.b32 {%0,%1,%2,%3},[%4];":"=r"(scv.x),"=r"(scv.y),"=r"(scv.z),"=r"(scv.w):"r"(smem_u32(reinterpret_cast<const uint4*>(tile+TW)+w*32+l)):"memory");}else{uint2 v;asm volatile("ld.volatile.shared.v2.b32 {%0,%1},[%2];":"=r"(v.x),"=r"(v.y):"r"(smem_u32(reinterpret_cast<const uint2*>(tile+TW)+w*32+l)):"memory");scv=make_uint4(v.x,v.y,0,0);}uint32_t scw[4]={scv.x,scv.y,scv.z,scv.w};
   float aa[CHAINS][NB][4]={};
#pragma unroll
   for(int g=0;g<GK;++g){uint4 v;asm volatile("ld.volatile.shared.v4.b32 {%0,%1,%2,%3},[%4];":"=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w):"r"(smem_u32(wp+g*32)):"memory");
    if constexpr(MAT==2){asm volatile(""::"r"(v.x),"r"(v.y),"r"(v.z),"r"(v.w),"r"(scv.x):"memory");sink=v.x;}
    else {uint32_t Sr=__byte_perm(scw[g>>1],0u,(g&1)?0x4242u:0x4040u)*16u+0x70007000u,Sr8=__byte_perm(scw[g>>1],0u,(g&1)?0x4343u:0x4141u)*16u+0x70007000u;
     uint32_t o[16];dq8(v.x,v.y,o);dq8(v.z,v.w,o+8);
#pragma unroll
     for(int j=0;j<NB;++j){uint32_t bq[8];
      if(j==0){for(int i=0;i<8;++i)bq[i]=(W==16&&g==GK-1)?bc[i*W*32+t]:bf[g][i];}
      else {const uint4* p=reinterpret_cast<const uint4*>(x+(size_t)(j*8+(l>>2))*4096+w*GK*64+g*64+(l&3)*16);uint4 v0=__ldcg(p),v1=__ldcg(p+1);bq[0]=v0.x;bq[1]=v0.y;bq[2]=v0.z;bq[3]=v0.w;bq[4]=v1.x;bq[5]=v1.y;bq[6]=v1.z;bq[7]=v1.w;}
      if constexpr(MAT==3){if(j>0)sink^=bq[0]^bq[1]^bq[2]^bq[3]^bq[4]^bq[5]^bq[6]^bq[7];}
      if(MAT!=3||j==0){
#pragma unroll
        for(int s=0;s<4;++s){auto os=o+s*4;mma16816(aa[g%CHAINS][j],hmul2(os[0],Sr),hmul2(os[1],Sr8),hmul2(os[2],Sr),hmul2(os[3],Sr8),bq[s*2],bq[s*2+1]);}}

     }
    }
   }
   if constexpr(MAT==1||MAT==3){for(int j=0;j<NB;++j)for(int k=0;k<4;++k){acc[j][k]=aa[0][j][k];if constexpr(CHAINS==2)acc[j][k]+=aa[1][j][k];sink^=__float_as_uint(acc[j][k]);}}
  }
  __syncthreads();if constexpr(HOT!=2){if(t==0&&begin+z+D<end)copy(begin+z+D,st);}__syncthreads();++tiles;
 }
 uint64_t c1=tclk(),g1=tg();if(t==0){auto p=stamp+c*8;p[0]=g0;p[1]=g1;p[2]=c0;p[3]=c1;p[4]=tiles;p[5]=wait;p[6]=W;p[7]=TB;}
 out[c*blockDim.x+t]=sink^__float_as_uint(acc[0][0]);
}

__global__ void pipe_latency(const uint8_t* a,const uint8_t* b,uint64_t* stamps,int bytes,int reps,int hot){
 extern __shared__ __align__(128) uint8_t mem[];auto mb=reinterpret_cast<uint64_t*>(mem+bytes);
 if(threadIdx.x)return;mbar_init(mb,1);asm volatile("fence.mbarrier_init.release.cluster;":::"memory");uint64_t pol;asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0,1.0;":"=l"(pol));
 for(int i=0;i<reps;++i){int tile=hot?(blockIdx.x%64):(blockIdx.x*reps+i)*79%16384;const uint8_t* p=a+(size_t)tile*36864;
  uint64_t g0=tg(),c0=tclk();mbar_expect_tx(mb,bytes);bulk_g2s(mem,p,bytes,mb,pol);mbar_wait(mb,i&1);uint64_t c1=tclk(),g1=tg();
  auto q=stamps+((size_t)blockIdx.x*reps+i)*4;q[0]=g0;q[1]=g1;q[2]=c0;q[3]=c1;
 }
}

int device_smem(){int d,v;cudaGetDevice(&d);cudaDeviceGetAttribute(&v,cudaDevAttrMaxSharedMemoryPerMultiprocessor,d);return v;}
template<class F>std::vector<int64_t> run_probe(F f,std::vector<int64_t> p,int threads,int shared,int cap,int reps,int mode){
 cudaFuncAttributes attr;C10_CUDA_CHECK(cudaFuncGetAttributes(&attr,f));int maxb;C10_CUDA_CHECK(cudaFuncSetAttribute(f,cudaFuncAttributeMaxDynamicSharedMemorySize,shared));
 C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxb,f,threads,shared));int d,sm;cudaGetDevice(&d);cudaDeviceGetAttribute(&sm,cudaDevAttrMultiProcessorCount,d);int grid=sm*std::min(maxb,cap);
 if(mode==0)f<<<grid,threads,shared,(cudaStream_t)p[4]>>>((uint32_t*)p[0],(uint64_t*)p[1],reps);
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {grid,maxb,attr.numRegs,(int64_t)attr.localSizeBytes,shared};
}
std::vector<int64_t> pipe_op_run(std::vector<int64_t> p,int op,int ilp,int cap,int reps){int shared=std::max(8192,device_smem()/(cap+1)+1024);
#define OP(K) case K:if(ilp==1)return run_probe(pipe_ops<K,1>,p,256,shared,cap,reps,0);else return run_probe(pipe_ops<K,4>,p,256,shared,cap,reps,0);
 switch(op){OP(0)OP(1)OP(2)OP(3)OP(4)OP(5)OP(6)OP(7)OP(8)OP(9)OP(10)OP(13)}
#undef OP
 throw std::runtime_error("op");}

template<int W,int GK,int D,int MAT,int HOT,int NB,int C>std::vector<int64_t> work_launch(std::vector<int64_t> p,int n,int M,int cap,int it){
 auto f=pipe_work<W,GK,D,MAT,HOT,NB,C>;constexpr int bytes=36864;int shared=D*bytes+16*D+(W==16?16*1024:0);
 shared=std::max(shared,device_smem()/(cap+1)+1024);
 int d,sm,r;cudaFuncAttributes a;cudaGetDevice(&d);cudaDeviceGetAttribute(&sm,cudaDevAttrMultiProcessorCount,d);cudaFuncGetAttributes(&a,f);
 C10_CUDA_CHECK(cudaFuncSetAttribute(f,cudaFuncAttributeMaxDynamicSharedMemorySize,shared));cudaOccupancyMaxActiveBlocksPerMultiprocessor(&r,f,W*32,shared);int G=sm*std::min(r,cap);
 f<<<G,W*32,shared,(cudaStream_t)p[6]>>>((uint8_t*)p[0],(uint8_t*)p[1],(int*)p[2],n,(__nv_bfloat16*)p[3],M,(uint32_t*)p[4],(uint64_t*)p[5],it);
 C10_CUDA_KERNEL_LAUNCH_CHECK();return {G,r,a.numRegs,(int64_t)a.localSizeBytes,shared,W*GK*576,W};
}
std::vector<int64_t> pipe_work_run(std::vector<int64_t> p,int kind,int n,int M,int cap,int it){
#define WK(K,W,G,D,MA,H,NB,C) case K:return work_launch<W,G,D,MA,H,NB,C>(p,n,M,cap,it);
 switch(kind){
 WK(0,8,8,1,0,0,1,1) WK(1,8,8,2,0,0,1,1) WK(2,8,8,4,0,0,1,1)
 WK(3,8,8,2,2,0,1,1) WK(4,8,8,2,1,0,1,1) WK(5,8,8,2,1,1,1,1)
 WK(6,8,8,1,1,2,1,1) WK(7,4,8,1,1,2,1,1) WK(8,16,4,1,1,2,1,1)
 WK(9,8,8,1,1,2,2,1) WK(10,8,8,1,1,2,4,1) WK(11,8,8,1,1,2,1,2) WK(12,8,8,1,3,2,4,1) WK(13,8,8,1,3,2,2,1)
 }
#undef WK
 throw std::runtime_error("work kind");}
void pipe_lat_run(std::vector<int64_t> p,int grid,int bytes,int reps,int hot){cudaFuncSetAttribute(pipe_latency,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes+16);
 pipe_latency<<<grid,32,bytes+16,(cudaStream_t)p[3]>>>((uint8_t*)p[0],(uint8_t*)p[1],(uint64_t*)p[2],bytes,reps,hot);C10_CUDA_KERNEL_LAUNCH_CHECK();}
