"""Reviewed public kernels: independent PDL and vector/direct-input arms.

The control is exact S21 eacc087b plus accepted 9b043487. No donor driver runs.
"""
import ast, base64, hashlib, json, lzma, re
from pathlib import Path

H=Path(__file__).resolve().parent; R=H.parents[3]
P=H.parent/'public_ports'; Y=H.parent/'public_hybrid'; O=H/'out'
O.mkdir(exist_ok=True)
def sha(s):return hashlib.sha256(s.encode()).hexdigest()
def assign(s,name):
    return next(ast.literal_eval(n.value) for n in ast.parse(s).body
        if isinstance(n,ast.Assign) and any(getattr(t,'id','')==name for t in n.targets))
def unpack(s):
    for n in ast.parse(s).body:
        if isinstance(n,ast.Assign) and any(getattr(t,'id','')=='_source' for t in n.targets):
            code=max((x.value for x in ast.walk(n.value) if isinstance(x,ast.Constant) and isinstance(x.value,str)),key=len)
            return lzma.decompress(base64.b85decode(code)).decode()
    return s
def control():
    s=(R/'problems/muse_decode_layer/cand_r3_best7_s21_final.py').read_text()
    assert sha(s)=='eacc087bd59b3cd58ee6b7690913868d86a8deb2ad1ff50040329ece62f7c8a4'
    patch=(Y/'candidate.patch').read_text();assert sha(patch).startswith('9b043487')
    s+=''.join(x[1:] for x in patch.splitlines(True) if x.startswith('+') and not x.startswith('+++'))
    assert sha(s)=='fbaf2793b3c32680d352d8e9ac0b669cd0148e6519593bcec1dfd4f779b68488'
    return s

VECTOR=r'''
template<typename V,typename Params>
__device__ __forceinline__ V qread(const Params& p,const __nv_bfloat16* src){
 const int64_t off=src-p.qkv;const int tok=off/2304,c=off%2304;
 if(!p.qv.S && p.qv.W==2304)return *(const V*)src;
 static_assert(sizeof(V)==8 || sizeof(V)==16,"reviewed vector readers only");
 V out;uint32_t* dst=(uint32_t*)&out;
 #pragma unroll
 for(int j=0;j<(int)sizeof(V)/8;j++){
  float4 a=make_float4(INFINITY,INFINITY,INFINITY,INFINITY);
  if(c+4*j<p.qv.W){
   const int64_t at=(int64_t)tok*p.qv.RS+c+4*j;
   if(p.qv.S){
    const float* fp=(const float*)p.qv.data;a=*(const float4*)(fp+at);
    #pragma unroll 1
    for(int s=1;s<p.qv.S;s++){
     const float4 v=*(const float4*)(fp+(int64_t)s*p.qv.SS+at);
     a.x+=v.x;a.y+=v.y;a.z+=v.z;a.w+=v.w;
    }
   }else{
    const uint2 v=*(const uint2*)((const __nv_bfloat16*)p.qv.data+at);
    const __nv_bfloat162* h=(const __nv_bfloat162*)&v;
    const float2 x=__bfloat1622float2(h[0]),y=__bfloat1622float2(h[1]);
    a=make_float4(x.x,x.y,y.x,y.y);
   }
  }
  const __nv_bfloat162 x=__floats2bfloat162_rn(a.x,a.y),y=__floats2bfloat162_rn(a.z,a.w);
  dst[2*j]=*(const uint32_t*)&x;dst[2*j+1]=*(const uint32_t*)&y;
 }
 return out;
}
'''
PDL=r'''
#ifndef PV_CHAIN
#define PV_CHAIN 0
#endif
#ifndef PV_SHARE
#define PV_SHARE 0
#endif
__device__ __forceinline__ void pv_release(){
#if PV_CHAIN
 asm volatile("griddepcontrol.launch_dependents;":::"memory");
#endif
}
__device__ __forceinline__ void pv_wait(){
#if PV_CHAIN
 asm volatile("griddepcontrol.wait;":::"memory");
#endif
}
template<typename K> static void pv_launch(K kernel,int g,int nt,int sm,cudaStream_t st,const A& a,const WF& f){
 cudaLaunchConfig_t cfg={};cfg.gridDim=dim3(g);cfg.blockDim=dim3(nt);cfg.dynamicSmemBytes=sm;cfg.stream=st;
 cudaLaunchAttribute attr={};attr.id=cudaLaunchAttributeProgrammaticStreamSerialization;
 attr.val.programmaticStreamSerializationAllowed=1;
 if(PV_CHAIN){cfg.attrs=&attr;cfg.numAttrs=1;}
 auto e=cudaLaunchKernelEx(&cfg,kernel,a,f);TORCH_CHECK(e==cudaSuccess,cudaGetErrorString(e));
}
'''

def lee_sources():
    # Full reviewed hybrid supplies only attnp and attnd; prune unused families.
    s=(Y/'out/lee_hybrid.cu').read_text()
    prefix=s[:s.index('template <bool PS1, bool FUSE, bool MRG, int X>')]
    mb=s[s.index('__device__ __forceinline__ void mb_wait('):s.index('// 2^x on')]
    verify=s[s.index('// 2^x on'):s.index('__device__ __forceinline__ void wg_n16_qk(')]
    decode=s[s.index('__device__ __forceinline__ void wg_n16_qk('):s.index('struct F {')]
    rope=s[s.index('// RoPE angle pos'):s.index('template <bool CL, bool MU>')]
    # PDL helpers need WF/A declarations from prefix.
    common=prefix+mb+rope+PDL+verify+decode
    # Release at entry, wait in consumers AFTER producer's CTA initialization.
    common=common.replace('attnp(A a, WF f) {','attnp(A a, WF f) {\n  pv_release();')
    common=common.replace('attnd(A a, WF f) {','attnd(A a, WF f) {\n  pv_release();')
    a=common.index('  if (EQ && !prod) {');z=common.index('\n#ifdef PROF',a)
    early=common[a:z]
    common=common[:a]+common[z:]
    needle='  // query rows: row qr_'
    a=common.index(needle)
    # Independent seq, page schedule, mbarriers and producer gather precede wait.
    common=common[:a]+'  pv_wait();\n'+early.replace('if (EQ && !prod)','if (EQ && (!PV_SHARE || LN==0))')+'\n'+common[a:]
    common=common.replace('  const __nv_bfloat16* row0 = f.qkv + (size_t)b * (2 * 16 + 4) * D;','  pv_wait();\n  const __nv_bfloat16* row0 = f.qkv + (size_t)b * (2 * 16 + 4) * D;',1)
    for family,nt in [('attnp','128 * (2 * NL + 1)'),('attnd','128 * (NL + 1)')]:
        old=f'  {family}<NL, X><<<g, {nt}, smw, st>>>(a, f);'
        assert old in common;common=common.replace(old,f'  pv_launch({family}<NL,X>,g,{nt},smw,st,a,f);')
    # Host supports only selected production families/flags, no experimental copies.
    oldhost=s[s.index('// Owned host ABI'):]
    a=oldhost.index('  A a{};');z=oldhost.index('  if(kind==1 || kind==-1){',a)
    host='''
void lee_attention(torch::Tensor qkv,torch::Tensor loc,torch::Tensor kc,torch::Tensor vc,torch::Tensor pt,torch::Tensor seq,
 torch::Tensor wo,torch::Tensor wml,torch::Tensor cnt,torch::Tensor out,torch::Tensor invf,torch::Tensor empty,
 int64_t kind,int64_t window,int64_t ns,int64_t x,double theta,double qk_factor,double sm_scale,double kv_scale,int64_t nmgcap){
 auto st=at::cuda::getCurrentCUDAStream();const int B=seq.size(0),L=qkv.size(1)/B;
'''+oldhost[a:z]+'''
 if(kind==1){
  f.nmg=2;const int smd=nl*(DRS*16384+32768+4096+2048+2048)+256+nl*1024+1024;
  adf<nl,64>(B*ns/nl,smd,st,a,f);
 }else{
  TORCH_CHECK(kind==3,"reviewed production families only");
  f.nmg=min((B*ns<=nl*sms)?(int)nmgcap:8,1<<(31-__builtin_clz((unsigned)ns)));
  const int smp=nl*(PRS*16384+8192+32768+2048)+1024+256+1024;
  const int grid=(x&256)?sms:B*ns/nl;
  if(x==68)apf<nl,68>(grid,smp,st,a,f);
  else if(x==32836)apf<nl,32836>(grid,smp,st,a,f);
  else {TORCH_CHECK(x==324,"reviewed flags only");apf<nl,324>(grid,smp,st,a,f);}
 }
 const auto e=cudaGetLastError();TORCH_CHECK(e==cudaSuccess,cudaGetErrorString(e));
}
'''
    common+=host
    # Vector arm: one shared query tile/CTA, no duplicate global partial reads by LN1.
    vec=common
    a=vec.index('template<typename V,typename Params>');z=vec.index('\nstruct WF',a)
    vec=vec[:a]+VECTOR+vec[z:]
    old='const uint32_t ring = sb + LN * LS, qs = ring + RS * 16384, vf0 = qs + 8192, ptr = vf0 + 32768;'
    new='const uint32_t ring = sb + LN * LS, qs0 = ring + RS * 16384, qs = PV_SHARE ? sb + RS * 16384 : qs0, vf0 = qs0 + 8192, ptr = vf0 + 32768;'
    assert old in vec;vec=vec.replace(old,new)
    a=vec.index('  // query rows:');z=vec.index("  // the block's own key",a)
    vec=vec[:a]+'  if(!PV_SHARE || LN==0){\n'+vec[a:z]+'  }\n'+vec[z:]
    old='  const uint2 kraw = EQ ? kpre : qread<uint2>(f,krow + (16 + h) * D + nd0), vraw = EQ ? vpre : qread<uint2>(f,krow + (18 + h) * D + nd0);'
    assert old in vec
    new='''  uint2 kraw,vraw;
  if constexpr(PV_SHARE && NL>1){
    const uint32_t own=sb+LS+RS*16384;
    if(LN==0){
      kraw=EQ?kpre:qread<uint2>(f,krow+(16+h)*D+nd0);
      vraw=EQ?vpre:qread<uint2>(f,krow+(18+h)*D+nd0);
      *(uint2*)(smem+(own-sb)+((nt*4+h)*64+nd0)*2)=kraw;
      *(uint2*)(smem+(own-sb)+((nt*4+2+h)*64+nd0)*2)=vraw;
    }
    asm volatile("bar.sync 13, %0;"::"n"(256*NL):"memory");
    kraw=*(const uint2*)(smem+(own-sb)+((nt*4+h)*64+nd0)*2);
    vraw=*(const uint2*)(smem+(own-sb)+((nt*4+2+h)*64+nd0)*2);
  }else{
    kraw=EQ?kpre:qread<uint2>(f,krow+(16+h)*D+nd0);
    vraw=EQ?vpre:qread<uint2>(f,krow+(18+h)*D+nd0);
  }
'''
    vec=vec.replace(old,new)
    # Decode query uses one consumer WG at LN0; LN1 reuses it. Reuse unused QS1 for own BF16 K/V.
    old='const uint32_t ring = sb + LN * LS, vf0 = ring + RS * 16384, pt0 = vf0 + 32768, qs = pt0 + 4096, ptr = qs + 2048;'
    new='const uint32_t ring = sb + LN * LS, vf0 = ring + RS * 16384, pt0 = vf0 + 32768, qs0 = pt0 + 4096, qs = PV_SHARE ? sb+RS*16384+32768+4096 : qs0, ptr = qs0 + 2048;'
    assert old in vec;vec=vec.replace(old,new)
    a=vec.index("  {  // Q': row r");z=vec.index("  // the new token's key",a)
    vec=vec[:a]+'  if(!PV_SHARE || LN==0)'+vec[a:z]+'''
  if constexpr(PV_SHARE && NL>1){
    const uint32_t own=sb+LS+RS*16384+32768+4096;
    if(LN==0 && w==0){
      const int nh=lane>>4,nd0=(lane&15)*4;
      *(uint2*)(smem+(own-sb)+(nh*64+nd0)*2)=qread<uint2>(f,row0+(16+nh)*D+nd0);
      *(uint2*)(smem+(own-sb)+((2+nh)*64+nd0)*2)=qread<uint2>(f,row0+(18+nh)*D+nd0);
    }
    asm volatile("bar.sync 13, %0;"::"n"(128*NL):"memory");
  }
'''+vec[z:]
    old='    const uint2 kraw = qread<uint2>(f,row0 + (16 + nh) * D + nd0), vraw = qread<uint2>(f,row0 + (18 + nh) * D + nd0);'
    new='''    const uint32_t own=sb+LS+RS*16384+32768+4096;
    const uint2 kraw=(PV_SHARE && NL>1)?*(const uint2*)(smem+(own-sb)+(nh*64+nd0)*2):qread<uint2>(f,row0+(16+nh)*D+nd0);
    const uint2 vraw=(PV_SHARE && NL>1)?*(const uint2*)(smem+(own-sb)+((2+nh)*64+nd0)*2):qread<uint2>(f,row0+(18+nh)*D+nd0);'''
    assert old in vec;vec=vec.replace(old,new)
    # No scalar instantiations remain in these families.
    assert 'qread<__nv_bfloat16>' not in vec
    return common,vec

def operator_source(chain,vector):
    s=(Y/'out/operator_selected.py').read_text()
    if vector:
        old='    qe = _partial_load(qptr,qmask,PS,PSS).to(tl.float32)\n    qo = _partial_load(qptr+1,qmask,PS,PSS).to(tl.float32)'
        new='''    qd=tl.arange(0,D)
    qa=_partial_load(qkv_ptr+tok.to(tl.int64)[:,None]*qkv_stride+qh[:,None]*D+qd[None,:],rvalid[:,None],PS,PSS)
    qev,qov=tl.split(tl.reshape(qa,[ROWS,D//2,2]))
    qe=tl.where(qmask,qev,0.);qo=tl.where(qmask,qov,0.)'''
        assert old in s;s=s.replace(old,new)
        old='        ke = _partial_load(kptr,jm[:,None],PS,PSS).to(tl.float32)\n        ko = _partial_load(kptr+1,jm[:,None],PS,PSS).to(tl.float32)'
        new='''        kd=tl.arange(0,DB)
        ka=_partial_load(qkv_ptr+ntok.to(tl.int64)[:,None]*qkv_stride+(HQ+hblk*HPB)*D+kd[None,:],jm[:,None],PS,PSS)
        ke,ko=tl.split(tl.reshape(ka,[LP,DB//2,2]))'''
        assert old in s;s=s.replace(old,new)
    if chain:
        s=s.replace('    b = tl.program_id(0)','    tl.extra.cuda.gdc_launch_dependents()\n    b = tl.program_id(0)',1)
        s=s.replace('    qe = _partial_load','    tl.extra.cuda.gdc_wait()\n    qe = _partial_load',1) if not vector else s.replace('    qd=tl.arange(0,D)','    tl.extra.cuda.gdc_wait()\n    qd=tl.arange(0,D)',1)
        # Only _fused_attn launch; nested helper calls don't launch kernels.
        s=s.replace('num_warps=fc["warps"], num_stages=fc["stages"])','num_warps=fc["warps"], num_stages=fc["stages"], launch_pdl=True)')
        assert 'launch_pdl=True' in s
    return s

def native_vectors(source):
    cs=assign(source,'_s16_global_C128_cs_CUDA');pa=assign(source,'PARTIAL_CUDA')
    # Pair-per-lane native query maps are adjacent; 16 even lanes fetch float4,
    # reduce planes once, then broadcast the other BF16 pair to the odd lane.
    body=r'''
  const int lane=threadIdx.x&31;
  #pragma unroll
  for(int i=0;i<N;i++){
    float4 a=make_float4(0.f,0.f,0.f,0.f);
    if(!(lane&1)){
      #pragma unroll 1
      for(int s=0;s<p.proj_splits;s++){
        const float4 v=*(const float4*)((const float*)p.qkv+(int64_t)s*p.split_stride+(int64_t)TOKEN[i]*STRIDE+col[i]);
        a.x+=v.x;a.y+=v.y;a.z+=v.z;a.w+=v.w;
      }
    }
    const __nv_bfloat162 lo=__floats2bfloat162_rn(a.x,a.y),hi=__floats2bfloat162_rn(a.z,a.w);
    uint32_t bits=(lane&1)?0:*(const uint32_t*)&hi;
    bits=__shfl_xor_sync(0xffffffffu,bits,1);
    __nv_bfloat162 h=lo;if(lane&1)memcpy(&h,&bits,4);
    RESULT
  }
'''
    a=cs.index('  const float* base = (const float*)p.qkv;',cs.index('DI void input_pairs'))
    z=cs.index('\n}',a)
    original=cs[a:z]
    replacement=body.replace('TOKEN','token').replace('STRIDE','p.stride').replace('RESULT','x[i]=valid[i]?__bfloat1622float2(h):make_float2(0.f,0.f);')
    cs=cs[:a]+'  if(p.L!=1){\n'+original+'\n    return;\n  }\n'+replacement+cs[z:]
    a=pa.index('    float2 acc[N];',pa.index('void proj_pairs'));z=pa.index('\n  }\n}',a)
    replacement=body.replace('TOKEN','tk').replace('STRIDE','p.qkv_stride').replace('RESULT','out[i]=h;')
    # N=4 is contiguous gate vectors within each thread, not across lanes.
    gate=r'''
    if constexpr(N==4){
      #pragma unroll
      for(int j=0;j<2;j++){
        float4 a=make_float4(0,0,0,0);
        for(int s=0;s<p.proj_splits;s++){
          const float4 v=*(const float4*)((const float*)p.qkv+(int64_t)s*p.split_stride+(int64_t)tk[2*j]*p.qkv_stride+col[2*j]);
          a.x+=v.x;a.y+=v.y;a.z+=v.z;a.w+=v.w;
        }
        out[2*j]=__floats2bfloat162_rn(a.x,a.y);out[2*j+1]=__floats2bfloat162_rn(a.z,a.w);
      }
    }else{
'''+replacement+'\n    }'
    pa=pa[:a]+gate+pa[z:]
    return cs,pa

def main():
    base=control();expanded=unpack(base);(O/'control.py').write_text(base)
    common,vec=lee_sources();run=(P/'lee_runtime.py.inc').read_text()
    run=run[:run.index('def attention(')]+(H/'lee_runtime.py.inc').read_text()
    modules={}
    for arm,chain,vector in [('D',1,0),('V',0,1),('F',1,1)]:
        cu=f'#define PV_CHAIN {chain}\n#define PV_SHARE {vector}\n'+(vec if vector else common)
        modules['lee_'+arm]='CUDA='+repr(cu)+'\n'+run
        modules['op_'+arm]=operator_source(chain,vector)
        (O/('lee_'+arm+'.cu')).write_text(cu)
    pack=(P/'input.py.inc').read_text().replace(' x=tl.program_id(0)*BLOCK',' tl.extra.cuda.gdc_launch_dependents()\n tl.extra.cuda.gdc_wait()\n x=tl.program_id(0)*BLOCK').replace('s,w,t,full,512)','s,w,t,full,512,launch_pdl=True)')
    modules['pack']=pack
    cs,pa=native_vectors(expanded)
    data=dict(base=base,expanded=expanded,modules=modules,native_cs=cs,native_partial=pa)
    (O/'bundle.json').write_text(json.dumps(data))
    for n,s in modules.items():compile(s,n,'exec');(O/(n+'.py')).write_text(s)
    (O/'native_cs_vector.cu').write_text(cs);(O/'native_partial_vector.cu').write_text(pa)
    m=dict(base_sha256=sha(base),base_shipment='eacc087b',accepted_patch='9b043487',module_sha256={n:sha(s) for n,s in modules.items()},native_cs_sha256=sha(cs),native_partial_sha256=sha(pa),licenses=sha((P/'NOTICES.txt').read_text()))
    (H/'build_manifest.json').write_text(json.dumps(m,indent=2)+'\n');print(m)
if __name__=='__main__':main()
