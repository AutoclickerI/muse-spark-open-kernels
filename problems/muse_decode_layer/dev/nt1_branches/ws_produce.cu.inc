__device__ __forceinline__ void ws_prefetch(
 int idx,int total,int first,int a1,int a2,Route* rt,int ns,uint8_t* raw,
 uint64_t* full,const uint8_t* w13,const uint8_t* w2){
 if(idx>=total*WS_GROUPS)return;
 int task=idx/WS_GROUPS,g=idx%WS_GROUPS,pass=task>=first,e,nb,m;
 decode_unit(pass?a2+task-first:a1+task,pass?rt->us2:rt->us1,rt->nblk,pass?rt->c2:rt->c1,pass?NT2:NT1,e,nb,m);
 const uint8_t* src=(pass?w2:w13)+(((size_t)e*(pass?NT2:NT1)+m)*WS_GROUPS+g)*WS_RAW;
 int st=idx%ns; mbar_expect_tx(full+st,WS_RAW);
 bulk_g2s(raw+st*WS_RAW,src,WS_RAW,full+st,rt->pre.pol_normal);
}

__device__ __forceinline__ void produce_ws(
 int pt,Route* rt,int total,int first,int a1,int a2,int ns,uint8_t* raw,uint32_t* decoded,
 uint64_t* full,uint64_t* empty,uint64_t* dfull,uint64_t* dempty,
 const uint8_t* w13,const uint8_t* w2){
 int lane=pt&31,pw=pt>>5;
 if(pt==0)for(int i=0;i<ns;++i)ws_prefetch(i,total,first,a1,a2,rt,ns,raw,full,w13,w2);
 for(int idx=0;idx<total*WS_GROUPS;++idx){
   int st=idx%ns;unsigned ph=(idx/ns)&1;
   mbar_wait(full+st,ph);mbar_wait(dempty+st,ph^1);
   #pragma unroll
   for(int z=0;z<WS_SUB;++z){
   #pragma unroll
   for(int r=0;r<2;++r){
     int w=pw+4*r,t=w*32+lane;const uint8_t* p=raw+st*WS_RAW+z*4608;
     uint4 v=*reinterpret_cast<const uint4*>(p+t*16);
     unsigned sf=*reinterpret_cast<const uint16_t*>(p+4096+t*2);
     unsigned s0=__byte_perm(sf,0u,0x4040u)*16u+0x70007000u;
     unsigned s1=__byte_perm(sf,0u,0x4141u)*16u+0x70007000u;
     unsigned o[16];dq8(v.x,v.y,o);dq8(v.z,v.w,o+8);
     #pragma unroll
     for(int j=0;j<16;++j)o[j]=hmul2(o[j],(j&1)?s1:s0);
     #pragma unroll
     for(int j=0;j<4;++j)*reinterpret_cast<uint4*>(decoded+st*WS_WORDS+z*4096+((j*8+w)*32+lane)*4)=make_uint4(o[4*j],o[4*j+1],o[4*j+2],o[4*j+3]);
   }
   }
   __syncwarp();if(lane==0){mbar_arrive(dfull+st);mbar_arrive(empty+st);}
   if(pt==0){
     mbar_wait(empty+st,ph);
     ws_prefetch(idx+ns,total,first,a1,a2,rt,ns,raw,full,w13,w2);
   }
 }
}
