struct WP {
 const __nv_bfloat16* qkv; const int64_t *pos,*loc; const int *pt,*seq;
 uint8_t *kc,*vc; const float *ks,*vs,*freq; __nv_bfloat16* out;
 float *parts; int* counters;
 int L,HQ,HK,stride,pages,ps,window,ns,proj_splits,tokens; float qscale,smscale;
};
// Projection partials are reduced and rounded to BF16 at the same boundary as stock.
DEVI float2 qkv_pair(const WP& p, int token, int column) {
 if(!p.proj_splits) return __bfloat1622float2(*(const __nv_bfloat162*)(p.qkv+(int64_t)token*p.stride+column));
 const float* raw=(const float*)p.qkv;
 float2 sum=make_float2(0,0);
 for(int s=0;s<p.proj_splits;s++) {
   const float2 x=*(const float2*)(raw+((int64_t)s*p.tokens+token)*p.stride+column);
   sum.x+=x.x;sum.y+=x.y;
 }
 return __bfloat1622float2(__floats2bfloat162_rn(sum.x,sum.y));
}
template<int L, bool SHARE=false>
__global__ __launch_bounds__(128,2) void window_mma(WP p) {
 constexpr int SZ=256*64, STAGES=2;
 extern __shared__ __align__(128) uint8_t shared[];
 uint8_t* K=shared; uint8_t* V=K+STAGES*SZ;
 uint8_t* NK=V+STAGES*SZ; uint8_t* NV=NK+16*64;
 __half* Q=(__half*)(NV+16*64);
 const int tid=threadIdx.x, warp=tid/32, lane=tid%32, g=lane/4, q4=lane%4;
 const int split=blockIdx.x,h=blockIdx.y,b=blockIdx.z,hk=SHARE?h:h/8;
 constexpr int R=SHARE?8:L;
 const int groups=SHARE?p.HK:p.HQ;
 const int seq=p.seq[b], lo=max(0,seq-p.window+1), count=seq-lo;
 const int chunk=((count+p.ns-1)/p.ns+63)&~63;
 const int begin=min(seq,lo+split*chunk), end=min(seq,begin+chunk);
 const int tiles=(end-begin+255)/256;
 auto issue=[&](int tile,int slot) {
#pragma unroll
   for(int i=0;i<8;i++) {
     const int ix=tid+i*128, t=ix/4, col=ix%4, key=begin+tile*256+t;
     const bool valid=tile<tiles && key<end;
     const int row=valid ? p.pt[(int64_t)b*p.pages+key/p.ps]*p.ps+key%p.ps : 0;
     const int off=slot*SZ+t*64+((col^swz(t))*16);
     const int64_t src=(int64_t)row*p.HK*64+hk*64+col*16;
     cp_async16(smem_u32(K+off),p.kc+src,valid?16:0);
     cp_async16(smem_u32(V+off),p.vc+src,valid?16:0);
   }
   cp_commit();
 };
 issue(0,0);
 const float ks=*p.ks, vs=*p.vs;
#pragma unroll
 for(int i=0;i<2;i++) {
   const int t=warp+4*i;
   float qx=0,qy=0,kx=0,ky=0,vx=0,vy=0;
   float cs=1,sn=0;
   if(t<R) {
     const int qt=SHARE?0:t, qh=SHARE?h*8+t:h;
     const auto qr=qkv_pair(p,b*L+qt,qh*64+2*lane);
     const float qn=rsqrtf(warp_sum(qr.x*qr.x+qr.y*qr.y)/64+1.e-5f)*p.qscale;
     if(p.freq) sincosf((float)p.pos[b*L+qt]*p.freq[lane],&sn,&cs);
     qx=__bfloat162float(__float2bfloat16_rn((qr.x*cs-qr.y*sn)*qn));
     qy=__bfloat162float(__float2bfloat16_rn((qr.x*sn+qr.y*cs)*qn));
   }
   if(t<L) {
     const auto kr=qkv_pair(p,b*L+t,(p.HQ+hk)*64+2*lane);
     const auto vr=qkv_pair(p,b*L+t,(p.HQ+p.HK+hk)*64+2*lane);
     const float kn=rsqrtf(warp_sum(kr.x*kr.x+kr.y*kr.y)/64+1.e-5f);
     kx=(kr.x*cs-kr.y*sn)*kn/ks; ky=(kr.x*sn+kr.y*cs)*kn/ks;
     vx=vr.x/vs; vy=vr.y/vs;
   }
   *(__half2*)(Q+t*64+2*lane)=__floats2half2_rn(qx,qy);
   const auto k8=__nv_cvt_float2_to_fp8x2(make_float2(kx,ky),__NV_SATFINITE,__NV_E4M3);
   const auto v8=__nv_cvt_float2_to_fp8x2(make_float2(vx,vy),__NV_SATFINITE,__NV_E4M3);
   const int off=t*64+(((lane/8)^swz(t))*16)+(lane%8)*2;
   *(unsigned short*)(NK+off)=k8; *(unsigned short*)(NV+off)=v8;
   // Pad up to 16 rows for the short new-token MMA.
   const int off2=(t+8)*64+(((lane/8)^swz(t+8))*16)+(lane%8)*2;
   *(unsigned short*)(NK+off2)=0; *(unsigned short*)(NV+off2)=0;
   if(t<L && split==0 && (SHARE || h%8==0)) {
     const int64_t dst=p.loc[b*L+t]*p.HK*64+hk*64+2*lane;
     *(unsigned short*)(p.kc+dst)=k8; *(unsigned short*)(p.vc+dst)=v8;
   }
 }
 __syncthreads();
 WarpState<1> st;
#pragma unroll
 for(int k=0;k<4;k++) {
   const uint2 x=*(const uint2*)(Q+g*64+16*q4+4*k);
   st.qa[0][k][0]=x.x; st.qa[0][k][2]=x.y;
   st.qa[0][k][1]=st.qa[0][k][3]=0;
 }
 st.m[0][0]=st.m[0][1]=-INFINITY; st.l[0][0]=st.l[0][1]=0;
#pragma unroll
 for(int n=0;n<8;n++)
#pragma unroll
 for(int e=0;e<4;e++) st.o[0][n][e]=0;
 for(int tile=0;tile<tiles;tile++) {
   cp_wait<0>(); __syncthreads();
   if(tile+1<tiles) issue(tile+1,(tile+1)%2);
   const int kp=begin+tile*256;
   compute_tile<1,8>(st,K+(tile%2)*SZ,V+(tile%2)*SZ,64*warp,0,0,kp,end,seq,p.window,
                     (kp<seq+L-p.window || kp+256>end),p.smscale*ks*1.4426950408889634f,R,SHARE);
 }
 cp_wait<0>(); __syncthreads();
 if(split==p.ns-1 && warp==0)
   compute_tile<1,2>(st,NK,NV,0,0,0,seq,seq+L,seq,p.window,true,p.smscale*ks*1.4426950408889634f,R,SHARE);
 float den=st.l[0][0]; den+=__shfl_xor_sync(0xffffffff,den,1); den+=__shfl_xor_sync(0xffffffff,den,2);
 // The pipeline is drained; reuse it for the four warp partials.
 __syncthreads();
 float* sp=(float*)shared;
 float* dst=sp+(warp*8+g)*68;
 float4* d4=(float4*)(dst+16*q4);
 d4[0]=make_float4(st.o[0][0][0],st.o[0][1][0],st.o[0][2][0],st.o[0][3][0]);
 d4[1]=make_float4(st.o[0][4][0],st.o[0][5][0],st.o[0][6][0],st.o[0][7][0]);
 d4[2]=make_float4(st.o[0][0][1],st.o[0][1][1],st.o[0][2][1],st.o[0][3][1]);
 d4[3]=make_float4(st.o[0][4][1],st.o[0][5][1],st.o[0][6][1],st.o[0][7][1]);
 if(q4==0) {dst[64]=st.m[0][0];dst[65]=den;}
 __syncthreads();
 float results[2][2];
#pragma unroll
 for(int i=0;i<2;i++) {
   const int j=warp+4*i;
   if(j<R) {
     float mx=-INFINITY;
#pragma unroll
     for(int w=0;w<4;w++) mx=fmaxf(mx,sp[(w*8+j)*68+64]);
     float total=0,x=0,y=0;
#pragma unroll
     for(int w=0;w<4;w++) {
       const float* pp=sp+(w*8+j)*68;
       const float ww=ex2(pp[64]-(mx == -INFINITY ? 0.f : mx));
       const float2 oo=*(const float2*)(pp+2*lane);
       x=fmaf(ww,oo.x,x);y=fmaf(ww,oo.y,y);total=fmaf(ww,pp[65],total);
     }
     if(p.ns==1) {results[i][0]=x*vs/total;results[i][1]=y*vs/total;}
     else {
       float* pp=p.parts+(((int64_t)b*groups+h)*p.ns+split)*R*68+j*68;
       *(float2*)(pp+2*lane)=make_float2(x,y);
       if(lane==0) *(float2*)(pp+64)=make_float2(mx,total);
     }
   }
 }
 if(p.ns>1) {
   __threadfence(); __syncthreads();
   __shared__ int last;
   if(tid==0) last=(atomicAdd(p.counters+b*groups+h,1)==p.ns-1);
   __syncthreads();
   if(!last) return;
#pragma unroll
   for(int i=0;i<2;i++) {
     const int j=warp+4*i;
     if(j<R) {
       const float* base=p.parts+((int64_t)b*groups+h)*p.ns*R*68+j*68;
       float2 ml[16], ov[16];
#pragma unroll
       for(int s=0;s<16;s++) if(s<p.ns) {
         const float* pp=base+s*R*68;
         ml[s]=__ldcg((const float2*)(pp+64));
         ov[s]=__ldcg((const float2*)(pp+2*lane));
       }
       float mx=-INFINITY;
#pragma unroll
       for(int s=0;s<16;s++) if(s<p.ns) mx=fmaxf(mx,ml[s].x);
       float total=0,x=0,y=0;
#pragma unroll
       for(int s=0;s<16;s++) if(s<p.ns) {
         const float ww=ex2(ml[s].x-mx);
         x=fmaf(ww,ov[s].x,x);y=fmaf(ww,ov[s].y,y);total=fmaf(ww,ml[s].y,total);
       }
       results[i][0]=x*vs/total;results[i][1]=y*vs/total;
     }
   }
 }
#pragma unroll
 for(int i=0;i<2;i++) {
   const int j=warp+4*i;
   if(j<R) {
     const float x=results[i][0],y=results[i][1];
     const float norm=rsqrtf(warp_sum(x*x+y*y)/64+1.e-5f);
     const int qt=SHARE?0:j,qh=SHARE?h*8+j:h;
     const auto gate=qkv_pair(p,b*L+qt,(p.HQ+2*p.HK+qh)*64+2*lane);
     *(__nv_bfloat162*)(p.out+(int64_t)(b*L+qt)*p.HQ*64+qh*64+2*lane)=
        __floats2bfloat162_rn(x*norm/(1+__expf(-gate.x)),y*norm/(1+__expf(-gate.y)));
   }
 }
 __syncthreads(); if(tid==0 && p.ns>1) atomicExch(p.counters+b*groups+h,0);
}
}
void window_forward(int64_t qkv,int64_t pos,int64_t loc,int64_t pt,int64_t seq,
 int64_t kc,int64_t vc,int64_t ks,int64_t vs,int64_t freq,int64_t out,int64_t parts,int64_t counters,
 int64_t B,int64_t L,int64_t HQ,int64_t HK,int64_t stride,int64_t pages,int64_t ps,int64_t window,
 int64_t ns,int64_t proj_splits,double qscale,double smscale,int64_t stream) {
 WP p{(const __nv_bfloat16*)qkv,(const int64_t*)pos,(const int64_t*)loc,(const int*)pt,(const int*)seq,
 (uint8_t*)kc,(uint8_t*)vc,(const float*)ks,(const float*)vs,(const float*)freq,(__nv_bfloat16*)out,
 (float*)parts,(int*)counters,(int)L,(int)HQ,(int)HK,(int)stride,(int)pages,(int)ps,(int)window,(int)ns,(int)proj_splits,(int)(B*L),(float)qscale,(float)smscale};
 constexpr int smem=2*2*256*64+2*16*64+8*64*2;
 cudaError_t err;
 if(L==1 && B>1) {cudaFuncSetAttribute(window_mma<1,true>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);
   window_mma<1,true><<<dim3(ns,HK,B),128,smem,(cudaStream_t)stream>>>(p);}
 else if(L==1) {cudaFuncSetAttribute(window_mma<1>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);
   window_mma<1><<<dim3(ns,HQ,B),128,smem,(cudaStream_t)stream>>>(p);}
 else {cudaFuncSetAttribute(window_mma<8>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem);
   window_mma<8><<<dim3(ns,HQ,B),128,smem,(cudaStream_t)stream>>>(p);}

 err=cudaGetLastError();if(err!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
}
