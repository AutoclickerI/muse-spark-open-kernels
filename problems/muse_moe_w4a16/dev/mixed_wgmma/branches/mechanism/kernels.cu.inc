template<int MODE,bool F>__global__ void __launch_bounds__(256,2) kernel(
 const int*Q,const unsigned char*SC,const float*GS,const __nv_bfloat16*X,const int*C,const int*OFF,const int*ORDER,const int*TASK,const int*SLOTS,
 const __nv_bfloat16*PNW,__nv_bfloat16*ACT,float*Y,float*SS,int M,
 const __grid_constant__ CUtensorMap map32,const __grid_constant__ CUtensorMap map64){
 constexpr int CS=MODE>=2?2:1,S=MODE>=2?(BR_HOPPER?4:3):3,MT=F?64:32;
 extern __shared__ __align__(1024) unsigned char ocsm[];float*red=(float*)ocsm+(threadIdx.x>>7)*256;
 __shared__ __align__(8) uint64_t bar[S];
 for(int u=blockIdx.x/CS;u<32*(MT/CS);u+=gridDim.x/CS){
  int e=ORDER[u/(MT/CS)],mt=((u%(MT/CS))*CS+blockIdx.x%CS)*2,n=C[e];if(n<=16)continue;
  for(int row=0;row<n;row+=64){int rem=min(n-row,64);
#define PR_TILE(NT) if constexpr(MODE<2)staged_tile<NT,F,S,MODE==1>(Q,SC,GS,X,C,OFF,SLOTS,M,PNW,ACT,Y,SS,e,row,mt+(threadIdx.x>>7),nullptr,nullptr,0x70007000u,red);else duplicate_tile<NT,F,CS,S,MODE==2>(Q,SC,GS,X,C,OFF,SLOTS,M,PNW,ACT,Y,SS,e,row,mt+(threadIdx.x>>7),bar,(NT==32?&map32:&map64),0x70007000u,red)
   if(rem<=32){PR_TILE(32);}else{PR_TILE(64);}
#undef PR_TILE
  }
 }
}
}
