"""Two independent resident FP8 attention warpgroups per CTA; cluster merge across CTAs."""
import importlib.util
from pathlib import Path
H=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('p5_window_resident_source',H.parent/'verify_resident/resgen.py')
resident=importlib.util.module_from_spec(spec);spec.loader.exec_module(resident)

NATIVE_EPILOGUE=r'''
  // Each WG has its own named barriers and resident KV slice. Only the
  // publication/DSM merge synchronizes the entire CTA and its cluster.
  wg_bar();float* part=(float*)K;
  const int g=lane>>2,t=lane&3;
#pragma unroll
  for(int rh=0;rh<2;rh++) {
    const int row=w*16+g+8*rh;
    float den=ls[rh];den+=__shfl_xor_sync(0xffffffffu,den,1);den+=__shfl_xor_sync(0xffffffffu,den,2);
    float* pp=part+row*68;
#pragma unroll
    for(int n=0;n<8;n++)*(float2*)(pp+8*n+2*t)=make_float2(O[n][2*rh],O[n][2*rh+1]);
    if(t==0)*(float2*)(pp+64)=make_float2(m[rh],den);
  }
  __syncthreads();cbar();mark(p,7);
  const int rank=split;
  for(int j=w;rank+j*CS<NR;j+=4) {
    const int r=rank+j*CS,token=r/H,head=kh*8+r%H;
    float2 ml[CS],ov[CS];
#pragma unroll
    for(int c=0;c<CS;c++) {
      const float* src=(float*)(sm+(c%WG)*WG_BYTES+4096)+r*68;
      ml[c]=cload(src+64,c/WG);ov[c]=cload(src+2*lane,c/WG);
    }
    float mx=-INFINITY;
#pragma unroll
    for(int c=0;c<CS;c++)mx=fmaxf(mx,ml[c].x);
    float x=0.f,y=0.f,den=0.f;
#pragma unroll
    for(int c=0;c<CS;c++){const float z=ex2(ml[c].x-mx);x=fmaf(z,ov[c].x,x);y=fmaf(z,ov[c].y,y);den=fmaf(z,ml[c].y,den);}
    const float scl=*p.vs/den;x*=scl;y*=scl;
    const float norm=rsqrtf(wsum(x*x+y*y)*(1.f/64.f)+1e-5f);
    int tk[1]={b*8+token},col[1]={(p.HQ+2*p.HK+head)*64+2*lane};bool valid[1]={true};float2 gate[1];input_pairs<1>(p,tk,col,valid,gate);
    *(__nv_bfloat162*)(p.out+((int64_t)(b*8+token)*p.HQ+head)*64+2*lane)=
      __floats2bfloat162_rn(x*norm/(1.f+__expf(-gate[0].x)),y*norm/(1.f+__expf(-gate[0].y)));
  }
  cbar();mark(p,8);
}
'''
LOCAL_EPILOGUE=r'''
  wg_bar();const int g=lane>>2,t=lane&3;
#pragma unroll
  for(int rh=0;rh<2;rh++) {
    const int row=w*16+g+8*rh;
    float den=ls[rh];den+=__shfl_xor_sync(0xffffffffu,den,1);den+=__shfl_xor_sync(0xffffffffu,den,2);
    float* pp=p.part+((int64_t)((b*p.HK+kh)*CS+split)*64+row)*68;
#pragma unroll
    for(int n=0;n<8;n++)*(float2*)(pp+8*n+2*t)=make_float2(O[n][2*rh],O[n][2*rh+1]);
    if(t==0)*(float2*)(pp+64)=make_float2(m[rh],den);
  }
  __threadfence();wg_bar();
  uint8_t* last=K;
  if(tid==0)*last=(atomicAdd(p.count+(b*p.HK+kh)*32,1)==CS-1);
  wg_bar();if(!*last)return;
  for(int r=w;r<64;r+=4) {
    float mx=-INFINITY;float2 ml[CS],ov[CS];
#pragma unroll
    for(int c=0;c<CS;c++){const float* pp=p.part+((int64_t)((b*p.HK+kh)*CS+c)*64+r)*68;ml[c]=*(float2*)(pp+64);ov[c]=*(float2*)(pp+2*lane);mx=fmaxf(mx,ml[c].x);}
    float x=0.f,y=0.f,den=0.f;
#pragma unroll
    for(int c=0;c<CS;c++){const float z=ex2(ml[c].x-mx);x=fmaf(z,ov[c].x,x);y=fmaf(z,ov[c].y,y);den=fmaf(z,ml[c].y,den);}
    const float scl=*p.vs/den;x*=scl;y*=scl;const float norm=rsqrtf(wsum(x*x+y*y)/64.f+1e-5f);
    const int token=r/8,head=kh*8+r%8;int tk[1]={b*8+token},col[1]={(p.HQ+2*p.HK+head)*64+2*lane};bool valid[1]={true};float2 gate[1];input_pairs<1>(p,tk,col,valid,gate);
    *(__nv_bfloat162*)(p.out+((int64_t)(b*8+token)*p.HQ+head)*64+2*lane)=__floats2bfloat162_rn(x*norm/(1.f+__expf(-gate[0].x)),y*norm/(1.f+__expf(-gate[0].y)));
  }
  wg_bar();if(tid==0)atomicExch(p.count+(b*p.HK+kh)*32,0);mark(p,8);
}
'''

QUERY_PREP=r'''
template<int QT>
DI void prep_verify(const VP& p,uint8_t* Q,uint8_t* KN,uint8_t* VN,float2* csbuf,int b,int kh,int sub,int seq,bool last) {
  static_assert(QT==1,"all64 query rows share each KV read");
  const int tid=threadIdx.x,w=tid>>5,lane=tid&31;
  for(int i=tid;i<256;i+=128){float sn,cs;__sincosf((float)p.positions[b*8+(i>>5)]*p.inv_freq[i&31],&sn,&cs);csbuf[i]=make_float2(cs,sn);}
  __syncthreads();asm volatile("griddepcontrol.wait;":::"memory");
  // Only Q is required during historical attention. New K/V is prepared
  // once by the owner, after the history's shared buffers have drained.
#pragma unroll 1
  for(int base=0;base<64;base+=16) {
    int tk[4],col[4];bool valid[4]={true,true,true,true};float2 x[4];
#pragma unroll
    for(int j=0;j<4;j++){const int r=base+w+4*j;tk[j]=b*8+r/8;col[j]=(kh*8+r%8)*64+2*lane;}
    input_pairs<4>(p,tk,col,valid,x);
#pragma unroll
    for(int j=0;j<4;j++) {
      const int r=base+w+4*j;const float norm=rsqrtf(wsum(x[j].x*x[j].x+x[j].y*x[j].y)/64.f+1e-5f);
      const float2 cs=csbuf[(r/8)*32+lane];const float a=x[j].x*norm,z=x[j].y*norm;
      *(uint16_t*)(Q+off(r,2*lane))=fp8pair(a*cs.x-z*cs.y,a*cs.y+z*cs.x);
    }
  }
  __syncthreads();proxy_fence();
}

DI void prep_new(const VP& p,uint8_t* KN,uint8_t* VN,const float2* csbuf,int b,int kh) {
  const int tid=threadIdx.x,w=tid>>5,lane=tid&31;
  for(int i=tid;i<256;i+=128){*(uint4*)(KN+16*i)=make_uint4(0,0,0,0);*(uint4*)(VN+16*i)=make_uint4(0,0,0,0);}
  __syncthreads();
  int tk[4],col[4];bool valid[4]={true,true,true,true};float2 x[4];
#pragma unroll
  for(int i=0;i<2;i++){const int t=w+4*i;tk[i]=tk[i+2]=b*8+t;col[i]=(p.HQ+kh)*64+2*lane;col[i+2]=(p.HQ+p.HK+kh)*64+2*lane;}
  input_pairs<4>(p,tk,col,valid,x);
#pragma unroll
  for(int i=0;i<2;i++) {
    const int t=w+4*i;const float norm=rsqrtf(wsum(x[i].x*x[i].x+x[i].y*x[i].y)/64.f+1e-5f);
    const float2 cs=csbuf[t*32+lane];const float a=x[i].x*norm,z=x[i].y*norm;
    const uint16_t k=fp8pair((a*cs.x-z*cs.y)/ *p.ks,(a*cs.y+z*cs.x)/ *p.ks),v=fp8pair(x[i+2].x/ *p.vs,x[i+2].y/ *p.vs);
    *(uint16_t*)(KN+off(t,2*lane))=k;
    const int perm=4*(t>>1)+(t&1);VN[off(2*lane,perm)]=(uint8_t)v;VN[off(2*lane+1,perm)]=(uint8_t)(v>>8);
    *(uint16_t*)(p.kc+p.loc[b*8+t]*p.HK*64+kh*64+2*lane)=k;
    *(uint16_t*)(p.vc+p.loc[b*8+t]*p.HK*64+kh*64+2*lane)=v;
  }
  __syncthreads();proxy_fence();
}
'''

def source(ctas=2,local=False,clocks=False,stream=False,pipeline=False):
    assert ctas in (2,4)
    # C2's 153600B CTA exceeds sm120's shared limit. Its explicit local
    # stand-in uses the C4/256-key slice, still executing both independent WGs.
    physical=4 if local and not stream else ctas
    logical=physical*2
    if stream:
        s=resident.r2.source('dsm')
        s=s.replace('constexpr int NSMAX = 128;',f'constexpr int CS={logical},CAP={2048//logical};\nconstexpr int NSMAX = 128;')
        a=s.index('template<int QT>\nDI void prep_verify')
        b=s.index('template<int L,int QT,int NK>\nDI void prob(',a);s=s[:a]+QUERY_PREP+s[b:]
        s=s.replace('return 4096+5*NK*64+8192+L*32*8;', 'return 4096+5*NK*64+L*32*8;')
        s=s.replace('uint8_t* KN=VT+KB;uint8_t* VN=KN+4096;float2* csbuf=(float2*)(VN+4096);',
                    'uint8_t* KN=K;uint8_t* VN=KN+4096;float2* csbuf=(float2*)(VT+KB);')
        s=s.replace('if(split==p.ns-1)virtual_tile<L,QT>(O,m,ls,Q,KN,VN,sc,seq);',
                    'if(split==p.ns-1){prep_new(p,KN,VN,csbuf,b,kh);virtual_tile<L,QT>(O,m,ls,Q,KN,VN,sc,seq);}')
        a=s.index('__global__ __launch_bounds__');b=s.index('template<int L,int QT,int NK> void attrs()',a)
        s=s[:a]+s[a:b].replace('p.ns','CS')+s[b:]
        a=s.index('template<int L,int QT,int NK> void run(VP p,cudaStream_t stream)')
        s=s[:a]+(H.parent/'verify_resident/early_edge.cu.inc').read_text()+'\n'+s[a:]
        s=s.replace('void run(VP p,cudaStream_t stream) {','void run(VP p,cudaStream_t stream,bool early) {')
        a=s.index('  auto e=cudaLaunchKernelEx',s.index('void run(VP p,'))
        s=s[:a]+'''  void* args[]={&p};cudaKernelNodeParams node={};node.func=(void*)sliding_main<L,QT,NK>;node.gridDim=cfg.gridDim;node.blockDim=cfg.blockDim;node.sharedMemBytes=cfg.dynamicSmemBytes;node.kernelParams=args;
  if(vr_capture_consumer(stream,node,CS,early))return;
'''+s[a:]
        # Reuse the fixed-shape ABI, not the broader old generator dispatch.
        abi=resident.source(logical).split('std::vector<int64_t> sliding_resources(',1)[1]
        s=s[:s.index('std::vector<int64_t> sliding_resources(')]+'std::vector<int64_t> sliding_resources('+abi
        s=s.replace('int qt, ns, cs, qwidth;', 'int* count; int qt, ns, cs, qwidth;')
    else:
        s=resident.source(logical,pipeline=True,local=False)
    s=s.replace(f'constexpr int CS={logical},CAP={2048//logical};',
        f'constexpr int CCTA={physical},WG=2,CS={logical},CAP={2048//logical};\nconstexpr int WG_BYTES='+('4096+5*128*64+2048;' if stream else '4096+2*CAP*64+5120+2048;'))
    s=s.replace('int* count; int qt, ns, cs, qwidth;', 'int* count; uint64_t* clocks; int qt, ns, cs, qwidth;')
    assert 'uint64_t* clocks;' in s
    s=s.replace('threadIdx.x','(threadIdx.x&127)').replace('__syncthreads();','wg_bar();')
    anchor='namespace {\n';assert s.count(anchor)==1
    s=s.replace(anchor,anchor+'''DI void wg_bar(){int id=1+(threadIdx.x>>7);asm volatile("bar.sync %0, 128;"::"r"(id):"memory");}
''')
    a=s.index('DI uint32_t su(')
    # VP and constants are already declared here; clock writes are absent
    # from uninstrumented source after preprocessing.
    mark='''DI void mark(const VP& p,int phase) {
#if WINDOW_CLOCK
  if((threadIdx.x&127)==0&&p.clocks){uint64_t t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t));const int group=((blockIdx.z*gridDim.y+blockIdx.y)*gridDim.x+blockIdx.x)*2+(threadIdx.x>>7);p.clocks[group*10+phase]=t;}
#endif
}
'''
    s=s[:a]+mark+s[a:]
    s=s.replace('return 4096+2*CAP*64+5120+L*32*8;', 'return WG*(4096+2*CAP*64+5120+L*32*8);')
    if stream:s=s.replace('return 4096+5*NK*64+L*32*8;', 'return WG*(4096+5*NK*64+L*32*8);')
    s=s.replace('__launch_bounds__(128,2)','__launch_bounds__(256,1)')
    s=s.replace('uint8_t* Q=sm;', 'uint8_t* Q=sm+(threadIdx.x>>7)*WG_BYTES;')
    s=s.replace('const int split=blockIdx.x,', 'const int split=blockIdx.x*WG+(threadIdx.x>>7),')
    s=s.replace('  // Every history row owned by this split', '  mark(p,0);\n  // Every history row owned by this split')
    s=s.replace('  cp_wait<0>();wg_bar();\n  // Transpose', '  mark(p,1);cp_wait<0>();wg_bar();mark(p,2);\n  // Transpose')
    s=s.replace('  proxy_fence();\n  if constexpr(L==1)', '  proxy_fence();mark(p,3);\n  if constexpr(L==1)')
    s=s.replace('wg_bar();asm volatile("griddepcontrol.wait;":::"memory");', 'wg_bar();asm volatile("griddepcontrol.wait;":::"memory");mark(p,4);')
    s=s.replace('  float O[8][4]={}', '  mark(p,5);\n  float O[8][4]={}')
    if stream:
        s=s.replace('  issue(0,0);issue(1,1);', '  mark(p,0);issue(0,0);issue(1,1);mark(p,1);')
        s=s.replace('    cp_wait<1>();wg_bar();', '    cp_wait<1>();wg_bar();if(it==0)mark(p,2);')
        s=s.replace('    wg_bar();proxy_fence();', '    wg_bar();proxy_fence();if(it==0)mark(p,3);')
        if pipeline:
            a=s.index('  for(int it=0;it<nt;it++)',s.index('__global__ __launch_bounds__'))
            b=s.index('  cp_wait<0>();wg_bar();',a)
            body=r'''
  if(nt) {
    cp_wait<1>();wg_bar();mark(p,2);
    for(int slab=0;slab<NK/64;slab++)trans64(V+slab*4096,VT+slab*4096,w,lane);
    wg_bar();proxy_fence();mark(p,3);
    float sx[16][4];uint32_t pa[16],pb[16];
    guard(sx);wg_fence();qk128(sx,Q,K);wg_commit();wg_wait<0>();guard(sx);
    prob<8,1,128>(sx,m,ls,alpha,pa,sc,begin,seq,end,p.window);rescale(O,alpha);protect(O);wg_bar();
    int it=0;
    // K(it) has drained and V(it) is already transposed. Refill that raw
    // slot while issuing QK(it+1), PV(it), and the next softmax independently.
#define WINDOW_STEP(PC,PN) { \
    issue(it+2,it%2);cp_wait<1>();wg_bar(); \
    protect(O);guardp(PC);guard(sx);wg_fence();qk128(sx,Q,K+((it+1)%2)*KB);wg_commit(); \
    wg_fence();multiply_v<128>(O,PC,VT);wg_commit();wg_wait<1>();guard(sx); \
    prob<8,1,128>(sx,m,ls,alpha,PN,sc,begin+(it+1)*128,seq,end,p.window); \
    wg_wait<0>();protect(O);guardp(PC);rescale(O,alpha);protect(O);wg_bar(); \
    for(int slab=0;slab<2;slab++)trans64(V+((it+1)%2)*KB+slab*4096,VT+slab*4096,w,lane); \
    wg_bar();proxy_fence();it++; \
  }
#pragma unroll 1
    while(it+2<nt){WINDOW_STEP(pa,pb);WINDOW_STEP(pb,pa);}
    if(it+1<nt){WINDOW_STEP(pa,pb);protect(O);guardp(pb);wg_fence();multiply_v<128>(O,pb,VT);wg_commit();wg_wait<0>();protect(O);guardp(pb);}
    else{protect(O);guardp(pa);wg_fence();multiply_v<128>(O,pa,VT);wg_commit();wg_wait<0>();protect(O);guardp(pa);}
#undef WINDOW_STEP
  }
'''
            s=s[:a]+body+s[b:]
    a=s.index('  // All asynchronous use has drained;');b=s.index('template<int L,int QT,int NK> void attrs()',a)
    s=s[:a]+'  mark(p,6);\n'+(LOCAL_EPILOGUE if local else NATIVE_EPILOGUE)+'\n'+s[b:]
    s=s.replace('>,128,cluster_smem', '>,256,cluster_smem').replace('cfg.blockDim=dim3(128)', 'cfg.blockDim=dim3(256)')
    s=s.replace('cfg.gridDim=dim3(p.ns,p.HK*QT,p.B)', 'cfg.gridDim=dim3(CCTA,p.HK*QT,p.B)')
    s=s.replace('vr_capture_consumer(stream,node,CS,early)', 'vr_capture_consumer(stream,node,1,early)' if local else 'vr_capture_consumer(stream,node,CCTA,early)')
    s=s.replace('if(L!=8 || qt!=1 || nk!=128 || cs!=CS)', f'if(L!=8 || qt!=1 || nk!=128 || cs!={ctas})')
    s=s.replace('if(L!=8 || qt!=1 || nk!=128 || cs!=CS || ns!=CS || window>2048)', f'if(L!=8 || qt!=1 || nk!=128 || cs!={ctas} || window>2048)')
    s=s.replace('return resource<8,1,128>(cs);', 'return resource<8,1,128>(1);' if local else 'return resource<8,1,128>(CCTA);')
    s=s.replace('int64_t counter,int64_t ns,int64_t early)', 'int64_t counter,int64_t ns,int64_t early,int64_t clocks)')
    s=s.replace('VP p={};p.part=', 'VP p={};p.clocks=(uint64_t*)clocks;p.part=')
    assert 'p.clocks=(uint64_t*)clocks' in s
    s=s.replace('p.qt=1;p.ns=CS;p.cs=CS;', 'p.qt=1;p.ns=CCTA;p.cs=CCTA;')
    # cudaLaunchKernelEx needs the same cluster dimension as the graph path.
    if local:
        s=s.replace('cfg.attrs=at;cfg.numAttrs=2;', 'cfg.attrs=at+1;cfg.numAttrs=1;')
    return '#define WINDOW_CLOCK '+str(int(clocks))+'\n'+s

if __name__=='__main__':
    import hashlib
    (H/'out').mkdir(exist_ok=True)
    for c in (2,4):
        for local in (False,True):
            s=source(c,local);p=H/'out'/f'wg2_c{c}{"_local" if local else ""}.cu';p.write_text(s);print(p.name,hashlib.sha256(s.encode()).hexdigest())
