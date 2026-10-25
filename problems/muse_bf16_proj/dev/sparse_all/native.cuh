#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
__device__ __forceinline__ unsigned dc_u32(const unsigned*p){unsigned v;asm volatile("ld.global.nc.L1::no_allocate.u32 %0,[%1];":"=r"(v):"l"(p));return v;}
__device__ __forceinline__ void dc_expand(unsigned a,unsigned b,unsigned c,unsigned ext,int rot,int fmt,int4*out) {
 unsigned cc[4]={};
#pragma unroll
 for(int i=0;i<32;++i){int sh=i*3;unsigned v=i<10?a>>sh:i==10?(a>>30)|(b<<2):i<21?b>>(sh-32):i==21?(b>>31)|(c<<1):c>>(sh-64);
  unsigned h=(((v&3)|(((ext>>i)&1)<<2))+rot)&7;int j=i&7;
  const int nt[8]={0,4,8,12,16,20,24,28},st[8]={3,19,7,23,11,27,15,31};
  const int nc[8]={0,8,16,24,4,12,20,28},sc[8]={7,15,23,31,11,19,27,3};
  const int nd[8]={0,8,16,24,28,4,12,20},sd[8]={7,15,23,31,3,11,19,27};
  int nb=fmt==0?nt[j]:fmt==1?nc[j]:nd[j],sb=fmt==0?st[j]:fmt==1?sc[j]:sd[j];
  cc[i/8]|=h<<nb;cc[i/8]|=((v>>2)&1)<<sb;
 }
 *out=make_int4(cc[0],cc[1],cc[2],cc[3]);
}
__device__ __forceinline__ const unsigned char* dc_address(const void* w,int unit,int lane,unsigned& ext,int& rot,int& fmt) {
 const unsigned*h=(const unsigned*)w;const unsigned char*p=(const unsigned char*)w;
 rot=dc_u32(h+4);fmt=dc_u32(h+12);const unsigned*meta=(const unsigned*)(p+dc_u32(h+5));
 unsigned present=dc_u32(meta+unit*2),off=dc_u32(meta+unit*2+1);ext=0;
 if(present&(1u<<lane))ext=dc_u32((const unsigned*)(p+dc_u32(h+7))+off+__popc(present&((1u<<lane)-1)));
 return p+dc_u32(h+6)+(size_t)unit*1408;
}
__device__ __forceinline__ void dc_read(const void*w,int unit,int lane,int4*out){
 unsigned ext;int rot,fmt;const unsigned char*p=dc_address(w,unit,lane,ext,rot,fmt);
 out[0]=*(const int4*)(p+lane*16);out[1]=*(const int4*)(p+512+lane*16);const unsigned*c=(const unsigned*)(p+1024);
 dc_expand(dc_u32(c+lane),dc_u32(c+32+lane),dc_u32(c+64+lane),ext,rot,fmt,out+2);
}
__device__ __forceinline__ void dc_stage(const void*w,int unit,int lane,int4*dst){
 unsigned ext;int rot,fmt;const unsigned char*p=dc_address(w,unit,lane,ext,rot,fmt);
 unsigned d0=(unsigned)__cvta_generic_to_shared(dst),d1=(unsigned)__cvta_generic_to_shared(dst+32),d2=(unsigned)__cvta_generic_to_shared(dst+64);
 asm volatile("cp.async.cg.shared.global [%0],[%1],16;"::"r"(d0),"l"(p+lane*16):"memory");
 asm volatile("cp.async.cg.shared.global [%0],[%1],16;"::"r"(d1),"l"(p+512+lane*16):"memory");
#pragma unroll
 for(int j=0;j<3;++j)asm volatile("cp.async.ca.shared.global [%0],[%1],4;"::"r"(d2+j*4),"l"(p+1024+j*128+lane*4):"memory");
 ((unsigned*)(dst+64))[3]=ext;
}
__device__ __forceinline__ __nv_bfloat16 dc_point(const void*w,int row,int col){
 const unsigned*h=(const unsigned*)w;const unsigned char*p=(const unsigned char*)w;int k=h[2],unit=row/16*(k/64)+col/64;
 int lane=(row%8)*4+(col%64)/16,i=(col/8%2)*16+(col/4%2)*8+(col/2%2)*4+(row/8%2)*2+col%2;
 unsigned ext;int rot,fmt;const unsigned char*b=dc_address(w,unit,lane,ext,rot,fmt);const unsigned*c=(const unsigned*)(b+1024);
 int sh=i*3,q=sh/32,at=sh%32;unsigned v=dc_u32(c+q*32+lane)>>at;if(at>29)v|=dc_u32(c+(q+1)*32+lane)<<(32-at);
 unsigned lo=b[(i/16)*512+lane*16+i%16];
 unsigned idx=(((v&3)|(((ext>>i)&1)<<2))+rot)&7;int base=((const int*)(p+h[8]))[row/16];
 unsigned bits=(((v>>2)&1)<<15)|((base+idx*2+(lo>>7))<<7)|(lo&127);
 const unsigned*em=(const unsigned*)(p+h[9]);unsigned pres=em[unit*2],off=em[unit*2+1];
 if(pres&(1u<<lane)){const unsigned*m=(const unsigned*)(p+h[10])+(off+__popc(pres&((1u<<lane)-1)))*2;unsigned mask=m[0];if(mask&(1u<<i))bits=((const uint16_t*)(p+h[11]))[m[1]+__popc(mask&((1u<<i)-1))];}
 return __ushort_as_bfloat16((uint16_t)bits);
}
