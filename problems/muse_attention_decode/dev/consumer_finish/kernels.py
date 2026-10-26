"""Consumer-side altered-work oracle. No output from this code is a candidate.

Ready inputs are packed into a BF16 allocation preserving the producer width:
FP8 families: Q/K/V byte columns 0..1279; half families: Q half0..1023,
K/V byte2048..2303. Gate stays BF16 at its original columns1280..2303.
"""
import ast,hashlib,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3]
sys.path.insert(0,str(H.parent/'setup_tree'));from pack_setup import base

def body(s,name,new):
    start=s.index(name);a=s.index('{',start);depth=1;b=a+1
    while depth:
        depth+=(s[b]=='{')-(s[b]=='}');b+=1
    return s[:a+1]+'\n'+new+'\n'+s[b-1:]

READY_PAIR='*(const uint16_t*)((const uint8_t*)p.qkv+2*(int64_t)token*p.stride+col)'

def sliding(s,stream=False):
    barrier='wg_bar();' if stream else '__syncthreads();'
    helper='DI uint16_t cf_ready(const VP& p,int token,int col){return '+READY_PAIR+';}\n'
    s=s.replace('template<int QT,bool GATE>\nDI void prep_decode',helper+'template<int QT,bool GATE>\nDI void prep_decode',1)
    decode=r'''
  constexpr int H=8/QT;
  const int tid=threadIdx.x&127,lane=tid&31,w=tid>>5;
  for(int i=tid;i<256;i+=128){
    if(i*16>=H*64)*(uint4*)(Q+16*i)=make_uint4(0,0,0,0);
    *(uint4*)(KN+16*i)=make_uint4(0,0,0,0);*(uint4*)(VN+16*i)=make_uint4(0,0,0,0);
  }
  CF_BAR
  for(int r=w;r<H;r+=4)*(uint16_t*)(Q+off(r,2*lane))=cf_ready(p,b,(kh*8+sub*H+r)*64+2*lane);
  if(w==0){
    const uint16_t k=cf_ready(p,b,(p.HQ+kh)*64+2*lane),v=cf_ready(p,b,(p.HQ+p.HK+kh)*64+2*lane);
    *(uint16_t*)(KN+off(0,2*lane))=k;VN[off(2*lane,0)]=v;VN[off(2*lane+1,0)]=v>>8;
  }
  if constexpr(GATE)for(int r=w;r<H;r+=4){
    const int col=(p.HQ+2*p.HK+kh*8+sub*H+r)*64+2*lane;
    *(float2*)(KN+1024+(r*64+2*lane)*4)=__bfloat1622float2(*(const __nv_bfloat162*)((const __nv_bfloat16*)p.qkv+(int64_t)b*p.stride+col));
  }
  CF_BAR proxy_fence();
'''.replace('CF_BAR',barrier)
    s=body(s,'DI void prep_decode(',decode)
    verify=r'''
  constexpr int L=8,H=8/QT,NR=L*H;
  const int tid=threadIdx.x&127,lane=tid&31,w=tid>>5;
  for(int i=tid;i<256;i+=128){
    if(i*16>=NR*64)*(uint4*)(Q+16*i)=make_uint4(0,0,0,0);
    *(uint4*)(KN+16*i)=make_uint4(0,0,0,0);*(uint4*)(VN+16*i)=make_uint4(0,0,0,0);
  }
  CF_BAR
  for(int r=w;r<NR;r+=4)*(uint16_t*)(Q+off(r,2*lane))=cf_ready(p,b*8+r/H,(kh*8+sub*H+r%H)*64+2*lane);
  CF_NEW
  CF_BAR proxy_fence();
'''.replace('CF_BAR',barrier)
    new=r'''
  for(int t=w;t<8;t+=4){
    const uint16_t k=cf_ready(p,b*8+t,(p.HQ+kh)*64+2*lane),v=cf_ready(p,b*8+t,(p.HQ+p.HK+kh)*64+2*lane);
    *(uint16_t*)(KN+off(t,2*lane))=k;
    const int c=4*(t>>1)+(t&1);VN[off(2*lane,c)]=v;VN[off(2*lane+1,c)]=v>>8;
  }
'''
    if 'DI void prep_verify(' in s:
        qbody=verify.replace('CF_NEW','' if stream else new)
        if stream:qbody=qbody.replace('    *(uint4*)(KN+16*i)=make_uint4(0,0,0,0);*(uint4*)(VN+16*i)=make_uint4(0,0,0,0);','')
        s=body(s,'DI void prep_verify(',qbody)
    if stream:s=body(s,'DI void prep_new(',r'''
  const int tid=threadIdx.x&127,lane=tid&31,w=tid>>5;
  for(int i=tid;i<256;i+=128){*(uint4*)(KN+16*i)=make_uint4(0,0,0,0);*(uint4*)(VN+16*i)=make_uint4(0,0,0,0);}
  wg_bar();
'''+new+'\nwg_bar();proxy_fence();')
    # General L/QT prep is used by global-CS; helper must be visible first.
    # Its generic body is identical except L comes from the template.
    if 'DI void prep(' in s:
        generic=verify.replace('constexpr int L=8,H=8/QT,NR=L*H;','constexpr int H=8/QT,NR=L*H;').replace('b*8+r/H','b*L+r/H').replace('CF_NEW',new.replace('t<8','t<L').replace('b*8+t','b*L+t'))
        s=body(s,'DI void prep(',generic)
        s=s.replace(helper,'');s=s.replace('template<int L,int QT>\nDI void prep',helper+'template<int L,int QT>\nDI void prep',1)
    return s

def port(s,partial=False):
    # Keep exactly the old history issue and merge, replacing only the main
    # input prologue by loads of already packed Q/current K/V and raw gates.
    a=s.index('  constexpr int QR =');b=s.index('  WarpState<L> st;',a)
    issue='''
#pragma unroll
  for(int st=0;st<STAGES-1;st++){if(st<ntiles)issue(st,rows[st]);cp_commit();}
  int rows_next[4];
#pragma unroll
  for(int k=0;k<4;k++)rows_next[k]=rows[STAGES-1][k];
''' if not partial else ''
    code=r'''
  const float ksc=pa_f32(p.k_scale),vsc=pa_f32(p.v_scale);
  const uint8_t* ready=(const uint8_t*)p.qkv;
  const bool gated=CF_GATED;
  if(gated){
    const int rank=split%p.cs,nrows=(L*16-rank+p.cs-1)/p.cs;
    for(int i=tid;i<nrows*8;i+=NTHR){const int rr=rank+(i>>3)*p.cs,j=rr>>4,hh=rr&15,c=i&7;
      cp_async16(smem_u32(sG+(i>>3)*D+c*8),p.qkv+(int64_t)(b*L+j)*p.qkv_stride+(p.hq+2*p.hkv+kvg*16+hh)*D+c*8,16);
    }
  }
  CF_ISSUE
  for(int r=warp;r<16*L;r+=4){const int j=r>>4,hh=r&15;
    *(uint32_t*)(sQ+((hh>>3)*(L*G)+j*G+(hh&7))*D+2*lane)=
      *(const uint32_t*)(ready+2*(int64_t)(b*L+j)*p.qkv_stride+2*((kvg*16+hh)*D+2*lane));
  }
  if(last){
    for(int i=tid;i<2*16*ROWB/16;i+=NTHR)if(i%(16*ROWB/16)>=L*ROWB/16)((uint4*)sNK)[i]=make_uint4(0,0,0,0);
    for(int i=0;i<L;i++){const int r=warp+4*i,j=r>>2,hk=(r>>1)&1,isv=r&1;
      const uint16_t f8=*(const uint16_t*)(ready+2*(int64_t)(b*L+j)*p.qkv_stride+2048+(isv?p.hkv:0)*64+(kvg*2+hk)*D+2*lane);
      const int byte=hk*D+2*lane;
      *(uint16_t*)((isv?sNV:sNK)+j*ROWB+((((byte>>4)^swz(j)))<<4)+(byte&15))=f8;
    }
  }
  __syncthreads();
'''.replace('CF_GATED','p.gated' if partial else 'p.qkv_stride>(p.hq+2*p.hkv)*D').replace('CF_ISSUE',issue).replace('pa_f32(p.k_scale)','*p.k_scale').replace('pa_f32(p.v_scale)','*p.v_scale')
    # pa_f32 exists in both current sources.
    return s[:a]+code+s[b:]

def vq(s):
    code=r'''
  const int kvh=2*hp+hh;
  const uint8_t* ready=(const uint8_t*)p.qkv;
  for(int i=0;i<16;i++){
    const int row=16*w+i,tok=row>>3,head=kvh*8+(row&7);
    *(uint16_t*)(Q+off(row,2*lane))=tok<p.L?*(const uint16_t*)(ready+2*(int64_t)(b*p.L+tok)*p.stride+head*64+2*lane):0;
  }
  if(virt)for(int tt=0;tt<2;tt++){
    const int tok=2*w+tt;if(tok>=p.L)break;
    const uint8_t* row=ready+2*(int64_t)(b*p.L+tok)*p.stride;
    const uint16_t k=*(const uint16_t*)(row+(p.HQ+kvh)*64+2*lane),v=*(const uint16_t*)(row+(p.HQ+p.HK+kvh)*64+2*lane);
    *(uint16_t*)(KN+off(tok,2*lane))=k;
    const int c=4*(tok>>1)+(tok&1);VN[off(2*lane,c)]=v;VN[off(2*lane+1,c)]=v>>8;
  }
'''
    s=body(s,'DI void prologue(',code)
    # Remove only the main's input wait, preserving combine's dependency.
    anchor='  asm volatile("griddepcontrol.wait;" ::: "memory");'
    assert s.count(anchor)==2;s=s.replace(anchor,'  // Oracle input is ready outside the timed call.',1)
    return s

def sources():
    vals={}
    for n in ast.parse(base()).body:
        if isinstance(n,ast.Assign):
            for t in n.targets:
                if getattr(t,'id','') in ('_s16_global_C128_cs_CUDA','r2_KERNELS','wr_KERNELS','PARTIAL_CUDA','ATTN_PORT_CUDA_SRC','VQ8_CUDA'):vals[t.id]=ast.literal_eval(n.value)
    out={'_s16_global_C128_cs_CUDA':sliding(vals['_s16_global_C128_cs_CUDA']),
         'PARTIAL_CUDA':port(vals['PARTIAL_CUDA'],True),'ATTN_PORT_CUDA_SRC':port(vals['ATTN_PORT_CUDA_SRC']),
         'VQ8_CUDA':vq(vals['VQ8_CUDA'])}
    out['r2_KERNELS']={'dsm':sliding(vals['r2_KERNELS']['dsm'])}
    out['wr_KERNELS']={'STREAM2':sliding(vals['wr_KERNELS']['STREAM2'],True)}
    return out

if __name__=='__main__':
    import json
    out=sources();(H/'out').mkdir(exist_ok=True);hashes={}
    for name,v in out.items():
        for k,s in (v.items() if isinstance(v,dict) else [(name,v)]):
            (H/'out'/f'{name}_{k}.cu').write_text(s);hashes[name+'_'+k]=hashlib.sha256(s.encode()).hexdigest()
    (H/'source_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n');print(hashes)
