"""Matched S19 diagnostic cuts;original BASE is retained byte-for-byte in each TU."""
import ast,hashlib,importlib.util,json,re
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[5];O=R/'runs/screen/p3a_ladder';O.mkdir(parents=True,exist_ok=True)
sp=importlib.util.spec_from_file_location('ip',D.parent/'intpipe/build.py');ip=importlib.util.module_from_spec(sp);sp.loader.exec_module(ip)
sha=lambda s:hashlib.sha256(s.encode()).hexdigest()

def diagnostic(cu):
 a=cu.index('template <int NT, int F>\n__global__');b=cu.index('\nsize_t smem_bytes',a)
 k=cu[a:b].replace('template <int NT, int F>','template <int NT,int F,int MODE,bool HIST>').replace(' moe_var(',' lad_var(')
 def sub(old,new,n=1):
  nonlocal k
  assert k.count(old)==n,(old[:70],k.count(old));k=k.replace(old,new)
 sub('TRACE = F & F_TRACE;','TRACE = false;\n  constexpr bool EPI=MODE==3||MODE==4||MODE==5||MODE==9;\n  constexpr bool COPY=MODE<4||MODE==4||MODE==9||MODE==10;\n  constexpr bool SINK=MODE<4||MODE==9;')
 for fn in ('prefetch_slot','prefetch_one'):
  k=re.sub(r'(?<!void )('+fn+r'\([^;\n]+\);)',r'if constexpr(COPY) \1',k)
 # Preserve syntax of if constexpr(SLOT)..else after wrapping the two calls.
 k=k.replace('if constexpr (SLOT) if constexpr(COPY)','if constexpr (SLOT) {if constexpr(COPY)').replace('w13p, w2p);\n      else if constexpr(COPY)','w13p, w2p);}\n      else if constexpr(COPY)')
 sub('if (cur_pass == 0 && seg_tiles > 0)', 'if (EPI && cur_pass == 0 && seg_tiles > 0)',2)
 k=k.replace('if constexpr (NREG)', 'if constexpr (NREG&&EPI)').replace('if constexpr(VNR)','if constexpr(VNR&&EPI)')
 wait=re.compile(r'while \(ld_acquire\(cnt \+ C_DONE \+ e(?:\*4\+nb)?\) < target\) __nanosleep\(64\);')
 assert len(wait.findall(k))==1
 k=wait.sub(lambda m:'if constexpr(MODE==4||MODE==9){'+m[0]+'}',k)
 sub('while (ld_acquire(cnt + C_GEN) == bar_gen0) __nanosleep(64);','if constexpr(MODE==4||MODE==9)while (ld_acquire(cnt + C_GEN) == bar_gen0) __nanosleep(64);')
 # All real activation loads retained from L2 upwards;ACT is an isolated transport contrast.
 pos=k.index('#pragma unroll\n',k.index('probe[16] += tg() - wait0;'))
 k=k[:pos]+'      if constexpr(MODE==5){for(int j=0;j<NT;++j)for(int g=0;g<8;++g)for(int q=0;q<8;++q)Bf[j][g][q]=0x3f803f80;}else if constexpr(MODE>=2&&MODE!=10&&MODE!=11){\n'+k[pos:]
 pos=k.index('    }\n    if constexpr (TRACE) { if (tid == 0 && pass == 1 && firstp)')
 k=k[:pos]+'      if constexpr(MODE==7){for(int g=0;g<8;++g)for(int j=0;j<NT;++j)for(int p=0;p<2;++p)lad_sink(red+buf*NCW*32*RED,tid,Bf[j][g][p*4],Bf[j][g][p*4+1],Bf[j][g][p*4+2],Bf[j][g][p*4+3]);}\n      }\n'+k[pos:]
 sub('    mbar_wait(&fullb[st], ph);','    if constexpr(COPY)mbar_wait(&fullb[st], ph);')
 sub('    if (lane == 0) mbar_arrive(&emptyb[st]);','    if constexpr(COPY)if (lane == 0) mbar_arrive(&emptyb[st]);')
 # L0 consumes every raw code and scale word. Common volatile sinks prevent ptxas DCE.
 sub('    {\n      const uint4* wq','    if constexpr(MODE!=7&&MODE!=11){\n      const uint4* wq')
 sub('  int buf = 0;', '  int buf = 0;uint32_t raw_ck=0;')
 sub('      const uint32_t scw[4] = {scv.x, scv.y, scv.z, scv.w};','      const uint32_t scw[4] = {scv.x, scv.y, scv.z, scv.w};\n      if constexpr(SINK)lad_sink(red+buf*NCW*32*RED,tid,scv.x,scv.y,scv.z,scv.w);if constexpr(MODE==10)raw_ck^=scv.x^scv.y^scv.z^scv.w;')
 sub('        const uint32_t sw = scw[g >> 1];','        if constexpr(MODE==0||MODE==10){if constexpr(MODE==0){for(int s=0;s<4;++s)lad_sink(red+buf*NCW*32*RED,tid,wv.x,wv.y,wv.z,wv.w);}else{raw_ck^=wv.x^wv.y^wv.z^wv.w;}}else{\n        const uint32_t sw = scw[g >> 1];')
 needle='          for (int j = 0; j < NT; ++j) mma16816(acc[j], a0, a1, a2, a3, Bf[j][g][2 * s], Bf[j][g][2 * s + 1]);'
 sub(needle,'          if constexpr(SINK)lad_sink(red+buf*NCW*32*RED,tid,a0,a1,a2,a3);\n'+needle.replace('for (int j','if constexpr(MODE>=2)for (int j'))
 sub('        }\n      }\n    }\n    __syncwarp();','        }}\n      }\n    }\n    __syncwarp();')
 # Epilogue cut retains identical raw-slot release,partial-store barrier and producer location.
 sub('    const float* rb = red +','    if constexpr(EPI){\n    const float* rb = red +')
 sub('    if constexpr(TRACE)if(tid==0){probe[56+pass]','    }\n    if constexpr(TRACE)if(tid==0){probe[56+pass]')
 # Metadata and stream rows use exactly the same CTA route/partition construction.
 pos=k.index('  if constexpr (!Z) {')
 extra='''
  if constexpr(MODE==6){
    if(tid<64){uint64_t v=U1+U2+rt->pre.index;int P=SLOT?rt->P:(rt->off[31]+rt->n[31]);for(int i=tid;i<npairs;i+=64)v+=pos_of[i]+1;for(int i=tid;i<P;i+=64)v+=pair_of[i];trace[(size_t)c*64+tid]=v;}return;
  }
  if constexpr(MODE==8){
    int* es=reinterpret_cast<int*>(red);if(tid==0){int n=0;if constexpr(SLOT){for(int i=0;i<rt->P;++i)es[n++]=rt->erank[i];}else{for(int e=0;e<32;++e)if(rt->n[e])es[n++]=e;}es[32]=n;}__syncthreads();
    if(tid==0){
      const int total=es[32]*768,n=total>c?(total-c+G-1)/G:0;uint32_t ck=0;
      auto issue=[&](int i){int u=c+i*G,e=es[u/768],m=u%768,st=i&1;const uint8_t* p=m<512?w13p+((size_t)e*512+m)*TILE_BYTES:w2p+((size_t)e*256+m-512)*TILE_BYTES;mbar_expect_tx(fullb+st,TILE_BYTES);bulk_g2s(stages+st*TILE_BYTES,p,TILE_BYTES,fullb+st,rt->pre.pol_first);};
      for(int i=0;i<2&&i<n;++i)issue(i);
      for(int i=0;i<n;++i){int st=i&1;mbar_wait(fullb+st,(i>>1)&1);ck^=*reinterpret_cast<volatile uint32_t*>(stages+st*TILE_BYTES);if(i+2<n)issue(i+2);}
      trace[(size_t)c*64]=ck;
    }return;
  }
  if constexpr(MODE==5){for(int i=tid;i<nstage*TILE_BYTES;i+=NTHR)stages[i]=i%TILE_BYTES<TILE_W_BYTES?0x11:0x88;__syncthreads();}
'''
 k=k[:pos]+extra+k[pos:]
 pos=k.index('  // ---------------- grid barrier')
 k=k[:pos]+'  if constexpr(!EPI){if constexpr(MODE==10){raw_ck=__reduce_xor_sync(~0u,raw_ck);if(lane==0)trace[c*64+warp]=raw_ck;}return;}\n'+k[pos:]
 # Minimal observer:entry/phase/end stamps and work mask,not per-tile phase clocks.
 sub('  int bar_gen0 = 0;', '  int bar_gen0 = 0;uint32_t hist_experts=0;bool hist_last=false;\n  if constexpr(HIST)if(tid==0){trace[c*64]=tg();uint32_t sm;asm("mov.u32 %0, %%smid;":"=r"(sm));trace[c*64+1]=sm;}')
 sub('      cur_pass = pass;', '      if constexpr(HIST)if(tid==0)hist_experts|=1u<<e;\n      cur_pass = pass;')
 sub('\n  if constexpr (!Z) {', '\n  if constexpr(HIST)if(tid==0)trace[c*64+2]=tg();\n  if constexpr (!Z) {')
 sub('    const uint8_t* tile =', '    if constexpr(HIST)if(tid==0&&(firstp||hist_last))trace[c*64+12+pass*2]=tg();\n    const uint8_t* tile =')
 sub('    __syncwarp();\n    if constexpr(COPY)', '    if constexpr(HIST)if(tid==0&&(firstp||hist_last))trace[c*64+13+pass*2]=tg();\n    __syncwarp();\n    if constexpr(COPY)')
 sub('      unit_body(pass,', '      if constexpr(HIST)hist_last=u0==ub-1;\n      unit_body(pass,')
 sub('    if constexpr (TRACE) { if (tid == 0) probe[pass ? 5 : 3] = tg(); }','    if constexpr(HIST)if(tid==0)trace[c*64+4+pass]=tg();\n    if constexpr (TRACE) { if (tid == 0) probe[pass ? 5 : 3] = tg(); }')
 sub('  if constexpr (TRACE) { if (tid == 0) probe[7] = tg(); }', '  if constexpr(HIST)if(tid==0)trace[c*64+6]=tg();\n  if constexpr (TRACE) { if (tid == 0) probe[7] = tg(); }')
 sub('  dump(LAST || c < 2 * M);','  if constexpr(HIST)if(tid==0){trace[c*64+7]=tg();trace[c*64+8]=b1-a1;trace[c*64+9]=b2-a2;trace[c*64+10]=hist_experts;}\n  dump(LAST || c < 2 * M);')
 helper='''__device__ __forceinline__ void lad_sink(float* red,int tid,uint32_t a,uint32_t b,uint32_t c,uint32_t d){asm volatile("st.volatile.shared.v4.u32 [%0],{%1,%2,%3,%4};"::"r"(smem_u32(red)+tid*16),"r"(a),"r"(b),"r"(c),"r"(d):"memory");}
'''
 # Inject into original TU so BASE remains literal original source.
 cu=cu[:b]+ '\n'+helper+k+cu[b:]
 suffix='''
template<int F,int K,bool H=false>void lad_launch(std::vector<int64_t>&v,int M,int start,int stages,int grid,cudaStream_t st){cudaLaunchConfig_t cfg{};cfg.gridDim=dim3(grid);cfg.blockDim=dim3(256);cfg.dynamicSmemBytes=smem_bytes(1,stages,M);cfg.stream=st;cudaLaunchAttribute a{};a.id=cudaLaunchAttributeProgrammaticStreamSerialization;a.val.programmaticStreamSerializationAllowed=1;cfg.attrs=&a;cfg.numAttrs=1;C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg,lad_var<1,F,K,H>,(const __nv_bfloat16*)v[0],(const int*)v[1],(const float*)v[2],(const __nv_bfloat16*)v[3],(const uint8_t*)v[4],(const uint8_t*)v[5],(const float*)v[6],(const float*)v[7],(__nv_bfloat16*)v[8],(float*)v[9],(float*)v[10],(int*)v[11],(__nv_bfloat16*)v[12],M,start,stages,(uint64_t*)v[13]));}
void lad_run(std::vector<int64_t>v,int M,int start,int stages,int grid,int64_t stream,int F,int mode,bool hist){
#define GO(K) if(mode==K){if(F==95)lad_launch<95,K>(v,M,start,stages,grid,(cudaStream_t)stream);else lad_launch<69,K>(v,M,start,stages,grid,(cudaStream_t)stream);return;}
if(hist){if(F==95)lad_launch<95,4,true>(v,M,start,stages,grid,(cudaStream_t)stream);else lad_launch<69,4,true>(v,M,start,stages,grid,(cudaStream_t)stream);return;}
GO(0) GO(1) GO(2) GO(3) GO(5) GO(6) GO(7) GO(8) GO(9) GO(10) GO(11)
#undef GO
throw std::runtime_error("unknown ladder mode");}
std::vector<int64_t> lad_info(int M,int F,int mode,bool hist){cudaFuncAttributes a{};int n=0;
#define INFO(K,H) {auto k=F==95?lad_var<1,95,K,H>:lad_var<1,69,K,H>;C10_CUDA_CHECK(cudaFuncGetAttributes(&a,k));C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,k,256,smem_bytes(1,2,M)));}
if(hist){INFO(4,true)}else{switch(mode){
#define CASE(K) case K:INFO(K,false)break;
CASE(0) CASE(1) CASE(2) CASE(3) CASE(5) CASE(6) CASE(7) CASE(8) CASE(9) CASE(10) CASE(11)
#undef CASE
default:throw std::runtime_error("unknown ladder info");}}
#undef INFO
return{n,a.numRegs,(int64_t)a.localSizeBytes,(int64_t)smem_bytes(1,2,M),(int64_t)a.sharedSizeBytes};}
void lad_init(){int d,opt;C10_CUDA_CHECK(cudaGetDevice(&d));C10_CUDA_CHECK(cudaDeviceGetAttribute(&opt,cudaDevAttrMaxSharedMemoryPerBlockOptin,d));
#define INIT(K,H) {cudaFuncAttributes a;C10_CUDA_CHECK(cudaFuncGetAttributes(&a,lad_var<1,69,K,H>));C10_CUDA_CHECK(cudaFuncSetAttribute(lad_var<1,69,K,H>,cudaFuncAttributeMaxDynamicSharedMemorySize,opt-a.sharedSizeBytes));C10_CUDA_CHECK(cudaFuncGetAttributes(&a,lad_var<1,95,K,H>));C10_CUDA_CHECK(cudaFuncSetAttribute(lad_var<1,95,K,H>,cudaFuncAttributeMaxDynamicSharedMemorySize,opt-a.sharedSizeBytes));}
INIT(0,false) INIT(1,false) INIT(2,false) INIT(3,false) INIT(5,false) INIT(6,false) INIT(7,false) INIT(8,false) INIT(9,false) INIT(10,false) INIT(11,false) INIT(4,true)
#undef INIT
}
'''
 return cu+suffix

def main():
 s=ip.original();specs={}
 for i,prefix in enumerate(ip.PREFIXES):
  cell=ip.cell(s,prefix);tree=ast.parse(cell);n=next(n for n in tree.body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','').endswith('_CUDA'));old=ast.literal_eval(n.value);cu=diagnostic(old);tag='lad'+str(i)
  cppnode=next(n for n in tree.body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','').endswith('_CPP'));cpp=ast.literal_eval(cppnode.value)+'\nvoid lad_run(std::vector<int64_t>,int,int,int,int,int64_t,int,int,bool);void lad_init();std::vector<int64_t> lad_info(int,int,int,bool);\n'
  cell=cell.replace(ast.get_source_segment(cell,n),n.targets[0].id+'='+repr(cu),1).replace(ast.get_source_segment(ip.cell(s,prefix),cppnode),cppnode.targets[0].id+'='+repr(cpp),1)
  cell=cell.replace("'epi_info'","'epi_info','lad_run','lad_init','lad_info'")
  cell=cell.replace('mod.smem_optin = mod.max_smem_attr()','mod.smem_optin = mod.max_smem_attr();mod.lad_init()')
  cell=cell.replace(prefix,tag).replace(prefix.upper(),tag.upper());p=D/(tag+'_cell.py');p.write_text(cell);(O/(tag+'.cu')).write_text(cu)
  specs[str(i)]=dict(cell=str(p.relative_to(R)),cell_sha256=sha(cell),cuda_sha256=sha(cu),original_cuda_sha256=sha(old),prefix=tag,original_S19_prefix=prefix)
 (D/'cells.json').write_text(json.dumps(specs,indent=2)+'\n');print('BUILT',list(specs))
if __name__=='__main__':main()
