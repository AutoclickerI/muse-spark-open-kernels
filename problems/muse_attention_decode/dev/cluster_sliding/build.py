"""Generate compact sliding-only, native FP8 attention with cluster/DSM output reduction."""
from pathlib import Path
import hashlib
HERE=Path(__file__).resolve().parent
VQ=HERE.parent/'vq8'

def source():
    base=(VQ/'vq8.cu').read_text()
    a=base.index('struct Sched {');b=base.index('DI uint32_t su(',a)
    base=base[:a]+base[b:]
    base=base[:base.index('// part: (grid + NS)')]
    base=base.replace('  int fused;', '  int qt, ns, cs, qwidth; const int64_t* positions;\n  int fused;')
    # Generic QK output: m64n64 or m64n128, plus a minimal m64n8 virtual tile.
    ops=[]
    for n in (8,64,128):
        nb=n//8
        outs=','.join('%%d%d'%i for i in range(nb*4))
        ptxouts=','.join('%%%d'%i for i in range(nb*4))
        constraints=', '.join(f'"+f"(s[{i//4}][{i%4}])' for i in range(nb*4))
        ai,bi,ac=nb*4,nb*4+1,nb*4+2
        ops.append(f'''DI void ss{n}(float (&s)[{nb}][4],uint64_t a,uint64_t b,int acc) {{
#if USE_WGMMA
  asm volatile("{{.reg .pred p;setp.ne.b32 p,%{ac},0;wgmma.mma_async.sync.aligned.m64n{n}k32.f32.e4m3.e4m3 {{{ptxouts}}},%{ai},%{bi},p,1,1;}}" : {constraints}:"l"(a),"l"(b),"r"(acc));
#else
  (void)a;(void)b;(void)acc;
#endif
}}
DI void qk{n}(float (&s)[{nb}][4],const uint8_t* Q,const uint8_t* K) {{
#if USE_WGMMA
  uint64_t qa=desc(su(Q)),ka=desc(su(K)),qb=qa+2,kb=ka+2;
  pin(qa);pin(ka);pin(qb);pin(kb);ss{n}(s,qa,ka,0);ss{n}(s,qb,kb,1);
#else
  const int lane=threadIdx.x&31,g=lane>>2,t=lane&3,w=threadIdx.x>>5;
#pragma unroll
  for(int j=0;j<{nb};j++)for(int e=0;e<4;e++)s[j][e]=0.f;
#pragma unroll
  for(int kk=0;kk<2;kk++) {{
    const uint32_t a0=word(Q,w*16+g,32*kk+4*t),a1=word(Q,w*16+g+8,32*kk+4*t);
    const uint32_t a2=word(Q,w*16+g,32*kk+16+4*t),a3=word(Q,w*16+g+8,32*kk+16+4*t);
#pragma unroll
    for(int j=0;j<{nb};j++)mma8(s[j],a0,a1,a2,a3,word(K,j*8+g,32*kk+4*t),word(K,j*8+g,32*kk+16+4*t));
  }}
#endif
}}
''')
    return base+'\n'.join(ops)+'\n'+(HERE/'kernel.cu.inc').read_text()

if __name__=='__main__':
    s=source();p=HERE/'cluster_sliding.cu';p.write_text(s);print(p,hashlib.sha256(s.encode()).hexdigest())
