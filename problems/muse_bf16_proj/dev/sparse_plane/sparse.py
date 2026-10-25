"""11 fixed bits + sparse high-index plane on attributed NF's warp-MMA structure.
Low BF16 bytes and every original escape remain exact. No floating-point re-encoding.
"""
import torch,re,hashlib
from torch.utils.cpp_extension import load_inline

SPARSE_CUDA=r'''
__device__ __forceinline__ uint32_t sp_ld(const uint32_t* p) {
 uint32_t v;asm volatile("ld.global.nc.L1::no_allocate.u32 %0,[%1];":"=r"(v):"l"(p));return v;
}
__device__ __forceinline__ void sp_read(const int4* wp,int unit,int units,int lane,int4* out,int rot,bool cpcode) {
 const uint32_t* hdr=(const uint32_t*)wp;uint32_t pres=sp_ld(hdr+unit*2),off=sp_ld(hdr+unit*2+1);
 const unsigned char* data=(const unsigned char*)wp+units*8;
 const unsigned char* block=data+(size_t)unit*1408;
 out[0]=*(const int4*)(block+lane*16);out[1]=*(const int4*)(block+512+lane*16);
 const uint32_t* code=(const uint32_t*)(block+1024);
 uint32_t a=sp_ld(code+lane),b=sp_ld(code+32+lane),c=sp_ld(code+64+lane),ext=0;
 if(pres&(1u<<lane)) {
  const uint32_t* sparse=(const uint32_t*)(data+(size_t)units*1408);
  ext=sp_ld(sparse+off+__popc(pres&((1u<<lane)-1u)));
 }
 uint32_t cc[4]={};
#pragma unroll
 for(int i=0;i<32;++i) {
  int at=i*3;uint32_t v;
  if(i<10)v=a>>at;
  else if(i==10)v=(a>>30)|(b<<2);
  else if(i<21)v=b>>(at-32);
  else if(i==21)v=(b>>31)|(c<<1);
  else v=c>>(at-64);
  unsigned h=(((v&3u)|(((ext>>i)&1u)<<2))+rot)&7u;
  int j=i&7;int nib=cpcode?((j&3)*8+(j/4)*4):j*4;
  int sign;
  if(cpcode) {
   // D9 nibble order and its sign-bit order are separate permutations.
   const int pos[8]={7,15,23,31,11,19,27,3};sign=pos[j];
  } else {const int pos[8]={3,19,7,23,11,27,15,31};sign=pos[j];}
  cc[i/8]|=h<<nib;cc[i/8]|=((v>>2)&1u)<<sign;
 }
 out[2]=make_int4(cc[0],cc[1],cc[2],cc[3]);
}
'''


def sparse_source(src,family):
    # Replace only packed W transport; original decoder, MMA, X path and reduction stay.
    cp=family=='cp';pref='dd_cap2_b_cpssm' if cp else 'v18s'
    p=src.index('#include');src=src[:p]+'#include <cuda_runtime.h>\n#include <stdint.h>\n'+SPARSE_CUDA+src[p:]
    needle='    const int4* wq = wp + (size_t)gw * (RT * NL * 32) + lane;'
    assert needle in src
    src=src.replace(needle,needle+'\n    const int sp_rot=(int)U0;')
    if not cp:
        old=f'#pragma unroll\n        for (int l = 0; l < NL; ++l) w[slot][j][l] = {pref}_ldw(wq + (size_t)s * sstride + (j * NL + l) * 32);'
        assert old in src
        src=src.replace(old,'        { sp_read(wp,(s*G+gw)*RT+j,KU*G*RT,lane,w[slot][j],sp_rot,false); }')
    else:
        # CP==0 direct loads use the same native sparse reader. CP>0 is a separate
        # measured structural arm; do not pretend staged expanded codes save bytes.
        old=f'        for (int l = 0; l < NL; ++l) if constexpr (CP == 0) w[slot][j][l] = {pref}_ldw(wq + (size_t)s * sstride + (j * NL + l) * 32);'
        assert old in src
        src=src.replace(old,'        { if constexpr(CP==0)sp_read(wp,(s*G+gw)*RT+j,KU*G*RT,lane,w[slot][j],sp_rot,true); }')
    return src


def sparse_pack(plan,family):
    """Compress the exact stream-ordered NF bank; preserve every low byte and escape."""
    wp=plan['wp'];wr,wk,ku,s=plan['cfg'];g=plan['esc'].shape[0];rt=1
    raw=wp.view(ku,g,rt,3,32,16);u=ku*g*rt
    c=raw[:,:,:,2].contiguous().view(torch.int32).to(torch.int64).reshape(u,32,4)&0xffffffff
    nib=(0,8,16,24,4,12,20,28) if family=='cp' else tuple(range(0,32,4))
    sign=(7,15,23,31,11,19,27,3) if family=='cp' else (3,19,7,23,11,27,15,31)
    shifts=torch.tensor(nib,device=wp.device);ss=torch.tensor(sign,device=wp.device)
    high=((c[...,None]>>shifts)&7).reshape(u,32,32)
    sg=((c[...,None]>>ss)&1).reshape(u,32,32)
    hist=torch.bincount(high.flatten(),minlength=8)
    # Pick the most populated contiguous4-index circular window, not a bit heuristic.
    counts=torch.stack([hist[(torch.arange(4,device=wp.device)+i)%8].sum() for i in range(8)])
    rot=int(counts.argmax());v=(high-rot)&7;ext=(v>>2)&1
    symbols=(v&3)|(sg<<2);bits=torch.zeros((u,32,3),device=wp.device,dtype=torch.int64)
    for i in range(32):
        shift=i*3;word=shift//32;at=shift%32
        bits[:,:,word]|=symbols[:,:,i]<<at
        if at>29:bits[:,:,word+1]|=symbols[:,:,i]>>(32-at)
    compact=bits.to(torch.int32).permute(0,2,1).contiguous().view(torch.uint8).reshape(u,384)
    masks=(ext<<torch.arange(32,device=wp.device)).sum(-1)
    occupied=masks!=0;presence=(occupied.long()<<torch.arange(32,device=wp.device)).sum(-1)
    counts=occupied.sum(-1);offset=counts.cumsum(0)-counts
    hdr=torch.stack((presence,offset),-1).to(torch.int32).contiguous().view(torch.uint8).flatten()
    lo=raw[:,:,:,:2].contiguous().reshape(u,1024)
    dense=torch.cat((lo,compact),1).flatten();sparse=masks[occupied].to(torch.int32).contiguous().view(torch.uint8)
    packed=torch.cat((hdr,dense,sparse)).contiguous()
    # Independent inverse for each emitted bit and nibble before handing bytes to CUDA.
    rb=torch.zeros_like(symbols)
    for i in range(32):
        shift=i*3;word=shift//32;at=shift%32;val=bits[:,:,word]>>at
        if at>29:val|=bits[:,:,word+1]<<(32-at)
        rb[:,:,i]=val&7
    h2=(((rb&3)|ext*4)+rot)&7
    assert torch.equal(h2,high) and torch.equal(rb>>2,sg)
    return dict(plan,wp=packed,u01=rot),dict(weights=u*1024,original_bytes=wp.numel(),packed_bytes=packed.numel(),bits_per_weight=packed.numel()*8/(u*1024),rotation=rot,occupied=float(occupied.float().mean()))

SPARSE_VECTOR=r'''
 uint32_t cc[4]={};
#pragma unroll
 for(int q=0;q<4;++q) {
  uint32_t v=q==0?a:q==1?(a>>24)|(b<<8):q==2?(b>>16)|(c<<16):(c>>8);
  uint32_t lo=(v&7u)|((v&0x38u)<<1)|((v&0x1c0u)<<2)|((v&0xe00u)<<3);
  uint32_t z=v>>12;
  uint32_t hi=(z&7u)|((z&0x38u)<<1)|((z&0x1c0u)<<2)|((z&0xe00u)<<3);
  uint32_t n=lo|(hi<<16);
  uint32_t e=(ext>>(q*8))&255u;
  e=(e|(e<<12))&0x000f000fu;e=(e|(e<<6))&0x03030303u;e=(e|(e<<3))&0x11111111u;
  uint32_t h=(((n&0x33333333u)|(e<<2))+rot*0x11111111u)&0x77777777u;
  uint32_t sign=(n&0x44444444u)<<1;
  uint32_t sg=(sign&0x80000008u)|((sign&0x00000800u)>>4)|((sign&0x00080000u)>>8)|((sign&0x08000000u)>>12)
              |((sign&0x00000080u)<<12)|((sign&0x00008000u)<<8)|((sign&0x00800000u)<<4);
  cc[q]=h|sg;
 }
 out[2]=make_int4(cc[0],cc[1],cc[2],cc[3]);
'''

_sparse_scalar_source=sparse_source

def sparse_source(src,family,vector=False):
    out=_sparse_scalar_source(src,family)
    if vector:
        a=out.index(' uint32_t cc[4]={};');b=out.index(' out[2]=make_int4(cc[0],cc[1],cc[2],cc[3]);',a)+len(' out[2]=make_int4(cc[0],cc[1],cc[2],cc[3]);')
        out=out[:a]+SPARSE_VECTOR+out[b:]
    return out
