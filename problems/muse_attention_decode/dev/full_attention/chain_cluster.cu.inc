// Every CTA in the split cluster remains alive for parallel output reduction.
// One exp per split/row, broadcast over dimensions; rows spread over all12 warps
// of all cluster CTAs. No single last warpgroup owns the whole output.
DI float2 dsm2(const float* x,int rank,int cs){
 if(cs==1)return *(const float2*)x;
 uint32_t a;asm volatile("mapa.shared::cluster.u32 %0,%1,%2;":"=r"(a):"r"(su(x)),"r"(rank));
 float2 v;asm volatile("ld.shared::cluster.v2.f32 {%0,%1},[%2];":"=f"(v.x),"=f"(v.y):"r"(a):"memory");return v;
}
DI void cluster_sync(int cs){if(cs==1){__syncthreads();return;}asm volatile("barrier.cluster.arrive.aligned;":::"memory");asm volatile("barrier.cluster.wait.aligned;":::"memory");}
DI void cluster_reduce(const VP& p,uint8_t* sm){
 const int cs=p.min_tiles,rank=blockIdx.x,tid=threadIdx.x,lane=tid&31,w=tid>>5,s=blockIdx.y;
 const int b=s/p.HP,hp=s%p.HP;const float vsc=*p.vs;
 cluster_sync(cs);
 for(int row=rank*12+w;row<128;row+=12*cs){
  float2 ml=lane<cs?dsm2((const float*)(sm+CLKPAD)+row*PR+64,lane,cs):make_float2(-INFINITY,0.f);
  float mx=ml.x;for(int d=16;d;d>>=1)mx=fmaxf(mx,__shfl_xor_sync(0xffffffff,mx,d));
  const float weight=ml.x==-INFINITY?0.f:ex2(ml.x-mx);float den=wsum(weight*ml.y),x=0.f,y=0.f;
  for(int c=0;c<cs;c++){float z=__shfl_sync(0xffffffff,weight,c);float2 o=dsm2((const float*)(sm+CLKPAD)+row*PR+2*lane,c,cs);x=fmaf(z,o.x,x);y=fmaf(z,o.y,y);}
  float sc=vsc/den;x*=sc;y*=sc;float norm=rsqrtf(wsum(x*x+y*y)/64.f+1e-5f);
  const int hh=row>>6,qr=row&63,tok=qr>>3,head=(2*hp+hh)*8+(qr&7);
  int tk[1]={b*p.L+tok},cl[1]={(p.HQ+2*p.HK+head)*64+2*lane};bool vd[1]={true};float2 g[1];input_pairs<1>(p,tk,cl,vd,g);
  *(__nv_bfloat162*)(p.out+((int64_t)(b*p.L+tok)*p.HQ+head)*64+2*lane)=__floats2bfloat162_rn(x*norm/(1.f+__expf(-g[0].x)),y*norm/(1.f+__expf(-g[0].y)));
 }
 // Keep every DSM allocation alive until every remote reader has finished.
 cluster_sync(cs);
}
