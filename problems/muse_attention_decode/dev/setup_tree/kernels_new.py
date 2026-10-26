"""S19 sliding arms: cluster-owned Q preparation and row-parallel DSM tree.

Local geometry is explicitly C1 (two WGs at verify), with every key. Native
geometry is unchanged: D8/D4 decode, two CTAs/four WGs at verify.
"""
import sys
from pathlib import Path
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H.parent/'sliding_oracle'))
import source as old

COPY=r'''
DI void query_copy16(void* dst,const void* src){
  uint32_t a=ca(src,0),x,y,z,w;
  asm volatile("ld.shared::cluster.v4.b32 {%0,%1,%2,%3},[%4];":"=r"(x),"=r"(y),"=r"(z),"=r"(w):"r"(a):"memory");
  *(uint4*)dst=make_uint4(x,y,z,w);
}
'''

def localize(s,family):
    s=s.replace('asm volatile("barrier.cluster.arrive.release.aligned;barrier.cluster.wait.acquire.aligned;":::"memory");','__syncthreads();')
    s=s.replace('cfg.attrs=at;cfg.numAttrs=2;','cfg.attrs=at+1;cfg.numAttrs=1;')
    if family=='v':
        assert 'CCTA=2,WG=2,CS=4,CAP=512' in s
        s=s.replace('CCTA=2,WG=2,CS=4,CAP=512','CCTA=1,WG=2,CS=2,CAP=1024')
    return s

def broadcast(s,family):
    # Metadata RoPE is already before the wait. Only the Q load/reduction,
    # RMS normalization, rotation and E4M3 pack become owner-only.
    a=s.index('template <int N>\nDI void input_pairs(');b=s.index('// (cos, sin)',a)
    helper=s[a:b].replace('DI void input_pairs(','DI void input_pairs_owned(')
    helper=helper.replace('raw[i] = *(const uint32_t*)((const __nv_bfloat16*)p.qkv + (int64_t)token[i] * p.stride + col[i]);','raw[i] = valid[i] ? *(const uint32_t*)((const __nv_bfloat16*)p.qkv + (int64_t)token[i] * p.stride + col[i]) : 0;')
    helper=helper.replace('v[i] = *(const float2*)(base + (int64_t)s * p.split_stride + (int64_t)token[i] * p.stride + col[i]);','v[i] = valid[i] ? *(const float2*)(base + (int64_t)s * p.split_stride + (int64_t)token[i] * p.stride + col[i]) : make_float2(0.f,0.f);')
    s=s[:b]+helper+s[b:]
    a=s.index('DI void prep_decode(');b=s.index('template<int QT>\nDI void prep_verify',a)
    body=s[a:b]
    body=body.replace('  input_pairs<R>(p,tk,col,valid,x);','  const bool qowner=blockIdx.x==0;\n  for(int j=0;j<R;j++)valid[j]=valid[j]&&(w+4*j>=H||qowner);\n  input_pairs<R>(p,tk,col,valid,x);')
    body=body.replace('const int row=w+4*j;float2 y=x[j];','const int row=w+4*j;if(row<H&&!qowner)continue;float2 y=x[j];')
    body=body.replace('input_pairs<R>(p,tk,col,valid,x);','input_pairs_owned<R>(p,tk,col,valid,x);')
    assert 'qowner' in body;s=s[:a]+body+s[b:]
    if family=='v':
        a=s.index('DI void prep_verify(');b=s.index('DI void prep_new(',a);body=s[a:b]
        body=body.replace('#pragma unroll 1\n  for(int base=0;', '  if(blockIdx.x*WG+(threadIdx.x>>7)==0){\n#pragma unroll 1\n  for(int base=0;')
        body=body.replace('  wg_bar();proxy_fence();','  }\n  wg_bar();proxy_fence();')
        assert 'if(blockIdx.x*WG' in body;s=s[:a]+body+s[b:]
    a=s.index('DI uint32_t ca(');b=s.index('\n\n',a)
    s=s[:b]+'\n'+COPY+s[b:]
    anchor='  else prep_verify<QT>(p,Q,KN,VN,csbuf,b,kh,sub,seq,split=='+('CS-1' if family=='v' else 'p.ns-1')+');'
    assert s.count(anchor)==1
    s=s.replace(anchor,anchor+'''
  // Every Q owner has finished; no cluster member can retire until the
  // ordinary final cluster barrier. History copies target different storage.
  __syncthreads();cbar();
  if(split!=0)for(int i=tid;i<256;i+=128)query_copy16(Q+16*i,sm+16*i);
  __syncthreads();proxy_fence();
''')
    return s

def tree(s):
    anchor='  __syncthreads();cbar();mark(p,7);\n  const int rank=split;'
    assert s.count(anchor)==1
    # Pair the two WGs locally, with eight warps owning disjoint rows.
    # In-place publication is row-exclusive: the same warp reads both
    # inputs before overwriting that row in WG0's partial. No other warp
    # consumes the row until the following CTA+cluster barrier.
    stage=r'''
  __syncthreads();
  for(int r=(threadIdx.x>>5);r<NR;r+=8){
    float* dst=(float*)(sm+4096)+r*68;
    const float* peer=(const float*)(sm+WG_BYTES+4096)+r*68;
    float2 a=*(float2*)(dst+64),bml=*(const float2*)(peer+64);
    float2 av=*(float2*)(dst+2*lane),bv=*(const float2*)(peer+2*lane);
    float mx=fmaxf(a.x,bml.x),az=ex2(a.x-mx),bz=ex2(bml.x-mx);
    *(float2*)(dst+2*lane)=make_float2(fmaf(az,av.x,bz*bv.x),fmaf(az,av.y,bz*bv.y));
    if(lane==0)*(float2*)(dst+64)=make_float2(mx,fmaf(az,a.y,bz*bml.y));
  }
  __syncthreads();cbar();mark(p,7);
  const int rank=split;'''
    s=s.replace(anchor,stage)
    a=s.index('  const int rank=split;',s.index('__global__ __launch_bounds__'));b=s.index('  cbar();mark(p,8);',a)
    block=s[a:b].replace('float2 ml[CS],ov[CS];','float2 ml[CCTA],ov[CCTA];')
    block=block.replace('c<CS;c++','c<CCTA;c++').replace('sm+(c%WG)*WG_BYTES+4096','sm+4096').replace('c/WG','c')
    s=s[:a]+block+s[b:];return s

def hoist(s,local=False):
    # One page/physical-row calculation per key instead of four redundant
    # sector threads. Four adjacent lanes own four different tiles and keep
    # the page map in registers. Every copy sector obtains its physical row
    # from that tile's owner lane. No input-derived work escapes the call.
    tiles=16 if local else 4
    anchor='  auto issue=[&](int tile,int slot) {';assert s.count(anchor)==1
    code='  constexpr int PAGE_GROUPS='+str(tiles//4)+r''',PAGE_ROWS=NK/(NT/4);
  int preg[PAGE_GROUPS][PAGE_ROWS];
#pragma unroll
  for(int gg=0;gg<PAGE_GROUPS;gg++){
#pragma unroll
    for(int j=0;j<PAGE_ROWS;j++){
      const int tile=4*gg+(tid&3),key=begin+tile*NK+tid/4+(NT/4)*j;
      const int clamped=min(key,max(seq-1,0));
      int physical=0;
      if(tile<nt)physical=p.pt[(int64_t)b*p.pages+(clamped>>p.lg)]*p.ps+(clamped&(p.ps-1));
      preg[gg][j]=physical;
    }
  }
'''
    s=s.replace(anchor,code+anchor)
    a=s.index('      constexpr int R=NK/(NT/4);int pages[R];',s.index(anchor));b=s.index('#pragma unroll',s.index('      asm volatile("":::"memory");',a))
    s=s[:a]+'      constexpr int R=NK/(NT/4);\n'+s[b:]
    anchor='        const int physical=pages[j]*p.ps+(clamped&(p.ps-1));';assert s.count(anchor)==1
    s=s.replace(anchor,r'''        int physical=0;
#pragma unroll
        for(int gg=0;gg<PAGE_GROUPS;gg++)if(tile/4==gg)
          physical=__shfl_sync(0xffffffffu,preg[gg][j],(lane&~3)+(tile&3));''')
    return s

def source(family,arm='A',local=False):
    s=old.original(family)
    if arm=='Q':s=broadcast(s,family)
    elif arm=='T':assert family=='v';s=tree(s)
    elif arm=='H':s=hoist(s,local)
    else:assert arm=='A'
    return localize(s,family) if local else s

MODES={'M':0,'M2':0,'Z':29,'ZV':29|1024,'ZR':29|1024|2048,'ZQ':29|128,'ZI':29|128|256,'ZL':29|128|512,'E':32,'G':0}

def oracle(family,local=False):
    s=old.source(family,False)
    a=s.index('__global__ __launch_bounds__');b=s.index('template<int L,int QT,int NK> void attrs()',a);body=s[a:b]
    oldnt='const int nt=(end-begin+NK-1)/NK;';assert oldnt in body
    body=body.replace(oldnt,'const int nt=(p.oracle&512)?0:(end-begin+NK-1)/NK;')
    a2=body.index('  if constexpr(L==1)');b2=body.index('  oracle_trace(p,4);',a2)
    body=body[:a2]+'  if(!(p.oracle&128)){\n'+body[a2:b2]+'  }\n'+body[b2:]
    anchor='        cp16(K+slot*KB+off(row,16*(tid&3)),p.kc+offg,key<end&&!(p.oracle&16));\n        cp16(V+slot*KB+off(row,16*(tid&3)),p.vc+offg,key<end&&!(p.oracle&16));'
    assert anchor in body
    body=body.replace(anchor,'        asm volatile(""::"l"(offg):"memory");\n        if(!(p.oracle&256)){\n'+anchor+'\n        }')
    s=s[:a]+body+s[b:]
    anchor='DI void input_pairs(const VP& p, const int (&token)[N], const int (&col)[N], const bool (&valid)[N], float2 (&x)[N]) {'
    assert s.count(anchor)==1
    s=s.replace(anchor,anchor+'\n  if(p.oracle&1024){for(int i=0;i<N;i++)x[i]=make_float2(0.f,0.f);return;}')
    s=s.replace('csbuf[lane]=rope_cs((float)p.positions[b]*p.inv_freq[lane]);','csbuf[lane]=(p.oracle&2048)?make_float2(1.f,0.f):rope_cs((float)p.positions[b]*p.inv_freq[lane]);')
    # This shared syntax occurs in unused generic prep and the live verify
    # prep. Both variants retain every position/frequency calculation unless
    # the explicit RoPE oracle bit is set.
    anchor='__sincosf((float)p.positions[b*8+(i>>5)]*p.inv_freq[i&31],&sn,&cs);'
    if family=='v':
        assert s.count(anchor)==1;s=s.replace(anchor,'if(p.oracle&2048){sn=0.f;cs=1.f;}else '+anchor)
    return localize(s,family) if local else s
