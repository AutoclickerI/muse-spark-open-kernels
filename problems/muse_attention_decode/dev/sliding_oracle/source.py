"""Same-binary sliding ablations. Altered-work modes are lab oracles only."""
import importlib.util,re,sys
from pathlib import Path
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H.parent/'sliding_round2'))
import kernels
spec=importlib.util.spec_from_file_location('p5_window_kernel',H.parent/'window_rebuild/kernel.py')
window=importlib.util.module_from_spec(spec);spec.loader.exec_module(window)
COORD=1;CONST=2;NO_MERGE=4;LOADS=8;ZERO=16;EMPTY=32;NO_WAIT=64
MODES={'M':0,'M2':0,'C':COORD,'S':COORD|CONST,'X':COORD|CONST|NO_MERGE,
       'L':COORD|LOADS|NO_MERGE,'Z':COORD|LOADS|NO_MERGE|ZERO,'E':EMPTY,'W':NO_WAIT}
HELPERS=r'''
DI int oracle_group(){return ((blockIdx.z*gridDim.y+blockIdx.y)*gridDim.x+blockIdx.x)*OR_WG+(threadIdx.x>>7);}
DI uint64_t oracle_cycle(){
#if OR_CLOCK
  uint64_t x;asm volatile("mov.u64 %0, %%clock64;":"=l"(x));return x;
#else
  return 0;
#endif
}
DI void oracle_trace(const VP&p,int phase){
#if OR_CLOCK
  if((threadIdx.x&127)==0&&p.clocks){uint64_t x;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(x));p.clocks[oracle_group()*24+phase]=x;}
#endif
}
DI void oracle_acc(const VP&p,int phase,uint64_t t){
#if OR_CLOCK
  if((threadIdx.x&127)==0&&p.clocks)p.clocks[oracle_group()*24+phase]+=oracle_cycle()-t;
#endif
}
template<int NK> DI void oracle_constant(float(&m)[2],float(&ls)[2],float(&alpha)[2],uint32_t(&pk)[NK/8]){
  m[0]=m[1]=0.f;alpha[0]=alpha[1]=1.f;ls[0]+=NK;ls[1]+=NK;
#pragma unroll
  for(int i=0;i<NK/8;i++)pk[i]=0x38383838u;
}
'''

def original(family,local=False):
    if family=='v':return window.source(2,stream=True,local=local)
    s=kernels.source('dsm')
    # Only the actually tested L1/QT1/N128 body; runtime split geometry is intact.
    a=s.index('#define RES(');b=s.index('#undef RES',a)+len('#undef RES')
    s=s[:a]+'return resource<1,1,128>(cs);'+s[b:]
    a=s.index('#define GO(');b=s.index('#undef GO',a)+len('#undef GO')
    s=s[:a]+'run<1,1,128>(p,(cudaStream_t)stream);'+s[b:]
    if local:
        s=s.replace('asm volatile("barrier.cluster.arrive.release.aligned;barrier.cluster.wait.acquire.aligned;":::"memory");','__syncthreads();')
        # C1 stand-in; source still uses real attention math on all window keys.
        s=s.replace('cfg.attrs=at;cfg.numAttrs=2;','cfg.attrs=at+1;cfg.numAttrs=1;')
    return s

def source(family,local=False,clocks=False):
    s=original(family,local);wg=2 if family=='v' else 1
    s=s.replace('struct VP {','struct VP {\n  const int64_t* coords;int oracle;'+('uint64_t* clocks;' if family=='d' else ''))
    a=s.index('DI uint32_t su(');s=s[:a]+HELPERS+'\n'+s[a:]
    # All modes share the same loaded function, resources, copy and tensor instructions.
    wait='asm volatile("griddepcontrol.wait;":::"memory");'
    assert wait in s
    s=s.replace(wait,'oracle_trace(p,2);if(!(p.oracle&64)){'+wait+'}oracle_trace(p,3);')
    a=s.index('__global__ __launch_bounds__');b=s.index('template<int L,int QT,int NK> void attrs()',a)
    body=s[a:b]
    begin='  asm volatile("griddepcontrol.launch_dependents;":::"memory");'
    assert body.count(begin)==1
    body=body.replace(begin,begin+'''
  oracle_trace(p,0);
  if(p.oracle&32){if((threadIdx.x&127)==0)((uint32_t*)p.out)[oracle_group()]=0;return;}
''')
    page='pages[j]=p.pt[(int64_t)b*p.pages+(clamped>>p.lg)];'
    assert body.count(page)==1
    body=body.replace('      constexpr int R=NK/(NT/4);int pages[R];','      uint64_t opage=oracle_cycle();\n      constexpr int R=NK/(NT/4);int pages[R];')
    body=body.replace(page,'pages[j]=(p.oracle&1)?0:p.pt[(int64_t)b*p.pages+(clamped>>p.lg)];')
    body=body.replace('      asm volatile("":::"memory");','      oracle_acc(p,10,opage);uint64_t ocopy=oracle_cycle();\n      asm volatile("":::"memory");')
    off='const int64_t offg=((int64_t)physical*p.HK+kh)*64+16*(tid&3);'
    assert body.count(off)==1
    body=body.replace(off,'const int64_t offg=(p.oracle&1)?p.coords[((int64_t)b*p.HK+kh)*2048+clamped-lo]+16*(tid&3):((int64_t)physical*p.HK+kh)*64+16*(tid&3);')
    body=body.replace(',key<end);',',key<end&&!(p.oracle&16));')
    body=body.replace('    }\n    commit();','      oracle_acc(p,11,ocopy);\n    }\n    commit();')
    assert 'oracle_acc(p,11' in body
    body=body.replace('  issue(0,0);issue(1,1);','  issue(0,0);issue(1,1);oracle_trace(p,1);')
    body=body.replace('  float O[8][4]={}', '  oracle_trace(p,4);\n  float O[8][4]={}')
    # Both native families have the same history loop body, modulo named WG barriers.
    bar='wg_bar();' if family=='v' else '__syncthreads();'
    old='    cp_wait<1>();'+bar
    assert body.count(old)==1
    body=body.replace(old,'    uint64_t owait=oracle_cycle();'+old.strip()+'oracle_acc(p,12,owait);\n    if(!(p.oracle&8)){uint64_t ot=oracle_cycle();')
    needle=bar+'proxy_fence();'
    # First occurrence is inside the history loop, query preparation is a separate function.
    assert body.count(needle)>=1
    body=body.replace(needle,needle+'oracle_acc(p,13,ot);uint64_t oqk=oracle_cycle();',1)
    old='    wg_commit();wg_wait<0>();guard(s);'
    assert body.count(old)==1
    body=body.replace(old,old+'oracle_acc(p,14,oqk);uint64_t osf=oracle_cycle();')
    old='prob<L,QT,NK>(s,m,ls,alpha,pk,sc,begin+it*NK,seq,end,p.window);'
    assert body.count(old)==1
    body=body.replace(old,'if(p.oracle&2)oracle_constant<NK>(m,ls,alpha,pk);else '+old+'oracle_acc(p,15,osf);uint64_t opv=oracle_cycle();')
    old='wg_commit();wg_wait<0>();protect(O);guardp(pk);'
    assert body.count(old)==1
    body=body.replace(old,old+'oracle_acc(p,16,opv);}\n')
    # Loads-only preserves all volatile cp.async instructions and their waits;
    # all MMA, transpose, probabilities and merge are bypassed explicitly.
    body=body.replace('  cp_wait<0>();'+bar,'  cp_wait<0>();'+bar+'oracle_trace(p,5);')
    if family=='v':
        old='if(split==CS-1){prep_new(p,KN,VN,csbuf,b,kh);virtual_tile<L,QT>(O,m,ls,Q,KN,VN,sc,seq);}'
    else:old='if(split==p.ns-1)virtual_tile<L,QT>(O,m,ls,Q,KN,VN,sc,seq);'
    assert old in body,(family,'new-tile anchor')
    body=body.replace(old,'if(!(p.oracle&8)){'+old+'}oracle_trace(p,6);\n  if(p.oracle&4){if((threadIdx.x&127)==0)((float*)p.out)[oracle_group()]=O[0][0];return;}')
    # Native cluster arrivals/publication and final norm/gate stay in the baseline.
    body=body.replace(bar+'cbar();',bar+'cbar();oracle_trace(p,7);')
    if local and family=='v':
        body=body.replace('  __threadfence();wg_bar();','  __threadfence();wg_bar();oracle_trace(p,7);')
    at=body.rfind('\n}');body=body[:at]+'\n  oracle_trace(p,8);'+body[at:]
    s=s[:a]+body+s[b:]
    # Constant probabilities also replace the single appended-key tile's softmax.
    s=s.replace('const uint8_t* VN,float sc,int seq) {','const uint8_t* VN,float sc,int seq,int oracle) {')
    a=s.index('  constexpr int H=8/QT;',s.index('DI void virtual_tile'))
    b=s.index('  rescale(O,alpha);protect(O);',a)
    s=s[:a]+'  if(!(oracle&2)){\n'+s[a:b]+'''  }else{m[0]=m[1]=0.f;alpha[0]=alpha[1]=1.f;ls[0]+=8;ls[1]+=8;pk[0]=pk[1]=0x3838u;}
'''+s[b:]
    s=s.replace('virtual_tile<L,QT>(O,m,ls,Q,KN,VN,sc,seq);','virtual_tile<L,QT>(O,m,ls,Q,KN,VN,sc,seq,p.oracle);')
    if family=='d':
        old='double sc,int64_t stream) {';new='double sc,int64_t stream,int64_t coords,int64_t oracle,int64_t clocks) {'
    else:
        old='int64_t early,int64_t clocks)';new='int64_t early,int64_t clocks,int64_t coords,int64_t oracle)'
    assert s.count(old)==1,(family,'ABI')
    s=s.replace(old,new)
    at=s.index('VP p={};',s.index('void sliding_forward('))+len('VP p={};')
    s=s[:at]+'p.coords=(const int64_t*)coords;p.oracle=oracle;'+('p.clocks=(uint64_t*)clocks;' if family=='d' else '')+s[at:]
    return f'#define OR_WG {wg}\n#define OR_CLOCK {int(clocks)}\n'+s

if __name__=='__main__':
    import hashlib
    (H/'out').mkdir(exist_ok=True)
    for family in ('d','v'):
        for local in (False,True):
            s=source(family,local);p=H/'out'/(family+('_local' if local else '')+'.cu');p.write_text(s);print(p.name,hashlib.sha256(s.encode()).hexdigest())
