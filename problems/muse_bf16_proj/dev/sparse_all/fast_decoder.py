"""Parallel 3-to-4-bit expansion, including all three original codeword layouts."""
FAST_EXPAND=r'''
__device__ __forceinline__ unsigned dc_perm_nibbles(unsigned x) {
  x=__byte_perm(x,x,0x3120);unsigned t=(x^(x>>4))&0x00f000f0u;return x^t^(t<<4);
}
__device__ __forceinline__ void dc_expand(unsigned a,unsigned b,unsigned c,unsigned ext,int rot,int fmt,int4*out) {
  unsigned word[4];
#pragma unroll
  for(int q=0;q<4;++q){
    unsigned x=q==0?a:q==1?(a>>24)|(b<<8):q==2?(b>>16)|(c<<16):c>>8;
    // Three butterfly steps insert one zero bit into every three-bit symbol.
    x=(x&0x00000fffu)|((x&0x00fff000u)<<4);
    x=(x&0x003f003fu)|((x&0x0fc00fc0u)<<2);
    x=(x&0x07070707u)|((x&0x38383838u)<<1);
    unsigned e=(ext>>(q*8))&255u;
    e=(e|(e<<12))&0x000f000fu;e=(e|(e<<6))&0x03030303u;e=(e|(e<<3))&0x11111111u;
    unsigned h=(((x&0x33333333u)|(e<<2))+rot*0x11111111u)&0x77777777u;
    unsigned sg=(x&0x44444444u)<<1;
    if(fmt==0){
      sg=(sg&0x80000008u)|((sg&0x00000800u)>>4)|((sg&0x00080000u)>>8)|((sg&0x08000000u)>>12)
        |((sg&0x00000080u)<<12)|((sg&0x00008000u)<<8)|((sg&0x00800000u)<<4);
      word[q]=h|sg;
    }else{
      unsigned v=dc_perm_nibbles(h|sg);
      if(fmt==2){unsigned odd=v&0xf0f0f0f0u;v=(v&0x0f0f0f0fu)|__funnelshift_r(odd,odd,8);}
      sg=v&0x88888888u;word[q]=(v&0x77777777u)|__funnelshift_r(sg,sg,28);
    }
  }
  *out=make_int4(word[0],word[1],word[2],word[3]);
}
'''

def vector_source(source):
    a=source.index('__device__ __forceinline__ void dc_expand(')
    b=source.index('__device__ __forceinline__ const unsigned char* dc_address',a)
    return source[:a]+FAST_EXPAND+source[b:]

VIEW_READ=r'''
struct dc_view{const unsigned char*dense;const unsigned*meta,*sparse;int rot,fmt;};
__device__ __forceinline__ int4 dc_view_v4(const void*p){
  int4 x;asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3},[%4];":"=r"(x.x),"=r"(x.y),"=r"(x.z),"=r"(x.w):"l"(p):"memory");return x;
}
__device__ __forceinline__ dc_view dc_make_view(const void*w){
  const unsigned*h=(const unsigned*)w;const unsigned char*p=(const unsigned char*)w;
  return {p+dc_u32(h+6),(const unsigned*)(p+dc_u32(h+5)),(const unsigned*)(p+dc_u32(h+7)),(int)dc_u32(h+4),(int)dc_u32(h+12)};
}
__device__ __forceinline__ unsigned dc_extension(const dc_view&v,int unit,int lane){
  unsigned present=dc_u32(v.meta+unit*2),off=dc_u32(v.meta+unit*2+1),ext=0;
  if(present&(1u<<lane))ext=dc_u32(v.sparse+off+__popc(present&((1u<<lane)-1)));
  return ext;
}
__device__ __forceinline__ void dc_read_ctx(const dc_view&v,int unit,int lane,int4*out){
  const unsigned char*p=v.dense+(size_t)unit*1408;const unsigned*c=(const unsigned*)(p+1024);
  out[0]=dc_view_v4(p+lane*16);out[1]=dc_view_v4(p+512+lane*16);
  // The dense loads no longer wait for the sparse extension address chain.
  unsigned a=dc_u32(c+lane),b=dc_u32(c+32+lane),z=dc_u32(c+64+lane);
  unsigned ext=dc_extension(v,unit,lane);dc_expand(a,b,z,ext,v.rot,v.fmt,out+2);
}
__device__ __forceinline__ void dc_stage_ctx(const dc_view&v,int unit,int lane,int4*dst){
  const unsigned char*p=v.dense+(size_t)unit*1408;
  unsigned d0=(unsigned)__cvta_generic_to_shared(dst),d1=(unsigned)__cvta_generic_to_shared(dst+32),d2=(unsigned)__cvta_generic_to_shared(dst+64);
  asm volatile("cp.async.cg.shared.global [%0],[%1],16;"::"r"(d0),"l"(p+lane*16):"memory");
  asm volatile("cp.async.cg.shared.global [%0],[%1],16;"::"r"(d1),"l"(p+512+lane*16):"memory");
  #pragma unroll
  for(int j=0;j<3;++j)asm volatile("cp.async.ca.shared.global [%0],[%1],4;"::"r"(d2+j*4),"l"(p+1024+j*128+lane*4):"memory");
  ((unsigned*)(dst+64))[3]=dc_extension(v,unit,lane);
}
__device__ __forceinline__ void dc_read(const void*w,int unit,int lane,int4*out){dc_read_ctx(dc_make_view(w),unit,lane,out);}
__device__ __forceinline__ void dc_stage(const void*w,int unit,int lane,int4*dst){dc_stage_ctx(dc_make_view(w),unit,lane,dst);}
'''

def view_source(source):
    source=vector_source(source)
    a=source.index('__device__ __forceinline__ void dc_read(')
    b=source.index('__device__ __forceinline__ __nv_bfloat16 dc_point(',a)
    return source[:a]+VIEW_READ+source[b:]

def install_view(ns):
    original_nf=ns['dc_nf_source'];original_cd=ns['dc_cd_source']
    def nf(src,family):
        out=original_nf(src,family)
        if family=='mw':
            needle='  uint4 raw[PF][NL]={};';assert needle in out
            out=out.replace(needle,'  const dc_view dc_ctx=dc_make_view(w);\n'+needle)
            out=out.replace('dc_read(w,','dc_read_ctx(dc_ctx,')
        else:
            needle='    const int4* wq = wp +';assert needle in out
            out=out.replace(needle,'    const dc_view dc_ctx=dc_make_view(wp);\n'+needle)
            out=out.replace('dc_read(wp,','dc_read_ctx(dc_ctx,').replace('dc_stage(wp,','dc_stage_ctx(dc_ctx,')
            if family=='cp':out=out.replace('((const unsigned*)wp)[4],1,','dc_ctx.rot,1,')
        return out
    def cd():
        out=original_cd();needle='  const uint4* wp[RT];';assert needle in out
        return out.replace(needle,'  const dc_view dc_ctx=dc_make_view(Wc);\n'+needle).replace('dc_read(Wc,','dc_read_ctx(dc_ctx,')
    ns['dc_nf_source']=nf;ns['dc_cd_source']=cd
