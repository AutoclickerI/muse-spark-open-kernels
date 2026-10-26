template<int S,int NT,int OW>constexpr int bytes(){return ((S*WG*OW*(QB+SB)+1023)&~1023)+S*NT*KT*2;}
template<int MODE,int CS,bool F>__global__ void __launch_bounds__(256,2) kernel(
 const int*Q,const unsigned char*SC,const float*GS,const __nv_bfloat16*X,const int*C,const int*OFF,const int*ORDER,const int*TASK,const int*SLOTS,
 const __nv_bfloat16*PNW,__nv_bfloat16*ACT,float*Y,float*SS,int M,
 const __grid_constant__ CUtensorMap map32,const __grid_constant__ CUtensorMap map64){
 constexpr int MAX=MODE==5?32:64,OW=MODE>=5?2:1,S=BR_HOPPER?(MODE==6?3:4):(MODE==6?2:3),MT=(F?64:32)/OW;
 extern __shared__ __align__(1024) unsigned char ocsm[];
 float*red=(float*)ocsm+(threadIdx.x>>7)*256;
 __shared__ __align__(8) uint64_t bar[S];
 for(int u=blockIdx.x/CS;u<32*(MT/CS);u+=gridDim.x/CS){
  int e=ORDER[u/(MT/CS)],mt=((u%(MT/CS))*CS+blockIdx.x%CS)*2*OW,n=C[e];if(n<=16)continue;
  for(int row=0;row<n;row+=MAX){int rem=min(n-row,MAX);
#define AX_TILE(NT) if constexpr(MODE==0)packed_tile<NT,F,1,S>(Q,SC,GS,X,C,OFF,SLOTS,M,PNW,ACT,Y,SS,e,row,mt+(threadIdx.x>>7),nullptr,nullptr,0x70007000u,red);else if constexpr(MODE<5)multicast_tile<NT,F,CS,S>(Q,SC,GS,X,C,OFF,SLOTS,M,PNW,ACT,Y,SS,e,row,mt+(threadIdx.x>>7),bar,(NT==32?&map32:&map64),0x70007000u,red);else wide_tile<NT,F,1,S>(Q,SC,GS,X,C,OFF,SLOTS,M,PNW,ACT,Y,SS,e,row,mt+(threadIdx.x>>7),nullptr,nullptr,0x70007000u,red)
   if constexpr(MAX==32){AX_TILE(32);}else{if(rem<=32){AX_TILE(32);}else{AX_TILE(64);}}
#undef AX_TILE
  }
 }
}
__global__ void gather(const __nv_bfloat16*x,__nv_bfloat16*gx,const int*C,const int*OFF,const int*SLOTS,int M){
 int e=blockIdx.x/((M+15)/16),r0=(blockIdx.x%((M+15)/16))*16,n=C[e];if(n<=16||r0>=n)return;
 for(int i=threadIdx.x;i<16*512;i+=256){int r=r0+i/512,k=(i%512)*8;if(r<n){int src=SLOTS[e*M+r]/8;*(uint4*)(gx+(size_t)(OFF[e]+r)*4096+k)=*(const uint4*)(x+(size_t)src*4096+k);}}
}
}
