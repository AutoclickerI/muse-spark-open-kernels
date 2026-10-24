"""Replace repeated compiler FP8 MMA lowering with one shared KV widening per stage."""
from pathlib import Path
HERE=Path(__file__).resolve().parent

def widened(s):
    s=s.replace('uint32_t q[2][4]','uint32_t q[4][4]')
    start=s.index('// One CTA transposes V once per stage')
    end=s.index('template<int N,bool MASK,int VN,bool CLOCK>',start)
    s=s[:start]+'''struct Trace { unsigned long long c[7]; };
DI unsigned long long tick(){unsigned long long x;asm volatile("mov.u64 %0, %%clock64;":"=l"(x)::"memory");return x;}
DI uint32_t widen8(uint16_t x){uint32_t y;asm("cvt.rn.f16x2.e4m3x2 %0,%1;":"=r"(y):"h"(x));return y;}
DI int off16(int r,int c){return r*128+(((c>>3)^(r&7))*16)+((c&7)*2);}
DI uint32_t word16(const uint8_t* p,int r,int c){return *(const uint32_t*)(p+off16(r,c));}
template<int N>DI void widen_stage(const uint8_t* K,const uint8_t* V,uint8_t* KH,uint8_t* VH){
 const int tid=threadIdx.x;
 for(int i=tid;i<N*32;i+=128){
  const int r=i/32,c=(i%32)*2;
  const uint32_t k=widen8(*(const uint16_t*)(K+off(r,c)));
  const uint32_t v=widen8(*(const uint16_t*)(V+off(r,c)));
  *(uint32_t*)(KH+off16(r,c))=k;
  const int perm0=((c&7)*8)&(N-1),perm1=(((c+1)&7)*8)&(N-1);
  *(uint16_t*)(VH+2*(c*N+(r^perm0)))=(uint16_t)v;
  *(uint16_t*)(VH+2*((c+1)*N+(r^perm1)))=(uint16_t)(v>>16);
 }
}
DI void mma16(float (&c)[4],const uint32_t (&a)[4],uint32_t b0,uint32_t b1){
 asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
 "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
 :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])
 :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));
}
DI uint32_t pack16(float x,float y){__half2 h=__floats2half2_rn(x,y);return *(uint32_t*)&h;}
'''+s[end:]
    old='''  for(int k=0;k<2;k++){
   const uint32_t b0=word(K,n*8+g,k*32+t*4);
   const uint32_t b1=word(K,n*8+g,k*32+16+t*4);
   mma8(s[n],st.q[k],b0,b1);
  }'''
    new='''  for(int k=0;k<4;k++){
   const uint32_t b0=word16(K,n*8+g,k*16+t*2);
   const uint32_t b1=word16(K,n*8+g,k*16+8+t*2);
   mma16(s[n],st.q[k],b0,b1);
  }'''
    assert old in s;s=s.replace(old,new,1)
    start=s.index(' uint32_t packed[NT];');end=s.index(' if constexpr(CLOCK)trace.c[6]+=tick()-ts;',start)
    s=s[:start]+''' uint32_t packed[NT][2];
#pragma unroll
 for(int n=0;n<NT;n++){
  packed[n][0]=pack16(s[n][0],s[n][1]);packed[n][1]=pack16(s[n][2],s[n][3]);
 }
 if constexpr(CLOCK){const auto now=tick();trace.c[5]+=now-ts;ts=now;}
#pragma unroll
 for(int k=0;k<N/16;k++){
  const uint32_t pa[4]={packed[2*k][0],packed[2*k][1],packed[2*k+1][0],packed[2*k+1][1]};
#pragma unroll
  for(int d=0;d<8;d++){
   const int col=8*d+g,kk=16*k+2*t+vkey,perm=((col&7)*8)&(VN-1);
   const uint32_t b0=*(const uint32_t*)(V+2*(col*VN+(kk^perm)));
   const uint32_t b1=*(const uint32_t*)(V+2*(col*VN+((kk+8)^perm)));
   mma16(st.o[d],pa,b0,b1);
  }
 }
'''+s[end:]
    s=s.replace('uint8_t* NK=Q+4096;uint8_t* NV=NK+2048;uint8_t* VT=NV+2048;',
                'constexpr int QRES=KT*128>8192?KT*128:8192;uint8_t* NK=Q+QRES;uint8_t* NV=NK+2048;uint8_t* KH=Q;uint8_t* VT=NV+2048;')
    s=s.replace('*(uint16_t*)(Q+row*64+2*lane)=fp8pair(q.x*r,q.y*r);','*(uint32_t*)(Q+row*128+4*lane)=widen8(fp8pair(q.x*r,q.y*r));')
    old=''' for(int k=0;k<2;k++){
  const int qr=warp*16+g;
  st.q[k][0]=*(const uint32_t*)(Q+qr*64+32*k+4*t);
  st.q[k][1]=*(const uint32_t*)(Q+(qr+8)*64+32*k+4*t);
  st.q[k][2]=*(const uint32_t*)(Q+qr*64+32*k+16+4*t);
  st.q[k][3]=*(const uint32_t*)(Q+(qr+8)*64+32*k+16+4*t);
 }'''
    new=''' for(int k=0;k<4;k++){
  const int qr=warp*16+g;
  st.q[k][0]=*(const uint32_t*)(Q+qr*128+2*(16*k+2*t));
  st.q[k][1]=*(const uint32_t*)(Q+(qr+8)*128+2*(16*k+2*t));
  st.q[k][2]=*(const uint32_t*)(Q+qr*128+2*(16*k+8+2*t));
  st.q[k][3]=*(const uint32_t*)(Q+(qr+8)*128+2*(16*k+8+2*t));
 }'''
    assert old in s;s=s.replace(old,new,1)
    s=s.replace('transpose_v<KT>(V+(it%ST)*KB,VT)','widen_stage<KT>(K+(it%ST)*KB,V+(it%ST)*KB,KH,VT)')
    s=s.replace('const uint8_t* kp=K+(it%ST)*KB+sub*4096;','const uint8_t* kp=KH+sub*8192;')
    s=s.replace('transpose_v<32>(NV,VT);','widen_stage<32>(NK,NV,KH,VT);')
    s=s.replace('tile<32,true,32,CLOCK>(st,NK,VT,','tile<32,true,32,CLOCK>(st,KH,VT,')
    s=s.replace('const float sc=(*p.vs)/(total*256.f);','const float sc=(*p.vs)/total;')
    s=s.replace('ST*KT*128+8192+KT*64','ST*KT*128+(KT*128>8192?KT*128:8192)+4096+KT*128')
    (HERE/'widened.cu').write_text(s)
    return s
