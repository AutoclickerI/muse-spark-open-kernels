__device__ __forceinline__ int arrive_release(int* p,int value){
  int old;asm volatile("atom.acq_rel.gpu.global.add.s32 %0,[%1],%2;":"=r"(old):"l"(p),"r"(value):"memory");return old;
}

// Every consumer thread calls this after its segment. C_DONE is W13-ready,
// [64,96) is W2 completion, [96,128) is normalized expert slots per token.
__device__ __forceinline__ void close_segment(
 int pass,int e,int tiles,Route* rt,int* cnt,float* y,float* ss,const __nv_bfloat16* pnw,
 const int* ids,const float* tw,const int16_t* pos_of,const int16_t* pair_of,
 __nv_bfloat16* out,int M,int start){
 int tid=threadIdx.x;
 __threadfence();bar_consumers();
 if(pass==0){if(tid==0)arrive_release(cnt+e,tiles);return;}
 if(tid==0)rt->wc[0][0]=arrive_release(cnt+64+e,tiles)+tiles==NT2*rt->nblk[e];
 bar_consumers();
 if(!rt->wc[0][0])return;
 if(tid<32)rt->run[tid]=0;
 for(int i=tid;i<rt->n[e];i+=NCT){int p=rt->off[e]+i;ss[p]=rsqrtf(__ldcg(ss+p)*(1.f/HD)+1e-8f);}
 __threadfence();bar_consumers();
 for(int i=tid;i<rt->n[e];i+=NCT){
   int p=rt->off[e]+i,t=pair_of[p]>>3,target=0;
   #pragma unroll
   for(int s=0;s<8;++s){int x=ids[t*8+s]-start;target+=(x>=0&&x<32);}
   if(arrive_release(cnt+96+t,1)==target-1){rt->run[t]=1;atomicExch(cnt+96+t,0);}
 }
 bar_consumers();
 if(tid==0){atomicExch(cnt+e,0);atomicExch(cnt+64+e,0);}
 for(int t=0;t<M;++t)if(rt->run[t]){
   for(int col=tid*8;col<HD;col+=NCT*8){
     uint4 wv=*reinterpret_cast<const uint4*>(pnw+col);const auto* wb=reinterpret_cast<const __nv_bfloat16*>(&wv);
     float a[8]={0,0,0,0,0,0,0,0};
     #pragma unroll
     for(int s=0;s<8;++s){int p=pos_of[t*8+s];if(p<0)continue;
       float sc=tw[t*8+s]*__ldcg(ss+p);
       const float4* yp=reinterpret_cast<const float4*>(y+(size_t)p*HD+col);
       float4 v0=__ldcg(yp),v1=__ldcg(yp+1);float v[8]={v0.x,v0.y,v0.z,v0.w,v1.x,v1.y,v1.z,v1.w};
       #pragma unroll
       for(int k=0;k<8;++k)a[k]+=sc*v[k]*(1.f+__bfloat162float(wb[k]));
     }
     uint4 ov;auto* ob=reinterpret_cast<__nv_bfloat16*>(&ov);
     #pragma unroll
     for(int k=0;k<8;++k)ob[k]=__float2bfloat16(a[k]);
     *reinterpret_cast<uint4*>(out+(size_t)t*HD+col)=ov;
   }
 }
 bar_consumers();
}
