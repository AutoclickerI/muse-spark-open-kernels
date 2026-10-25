"""Build activation-fragment lifetime experiments; each weight still decodes once for all tokens."""
from pathlib import Path
import ast
H=Path(__file__).resolve().parent
src=(H/'coded.py').read_text()
a=src.index('        uint4 xv[NT][2];')
b=src.index('#pragma unroll  // refill',a)
variants={
 'half':r'''
#pragma unroll
        for (int jh=0; jh<2; ++jh) {
          uint4 xv[NT];
#pragma unroll
          for (int nt=0; nt<NT; ++nt) {
            const uint32_t sa=smem_u32(xrow[nt]+bb*64+jh*8);
            asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(xv[nt].x),"=r"(xv[nt].y),"=r"(xv[nt].z),"=r"(xv[nt].w) : "r"(sa));
          }
#pragma unroll
          for (int jl=0; jl<2; ++jl) {
            const int j=jh*2+jl;
#pragma unroll
            for (int rt=0; rt<RT; ++rt) {
              uint32_t a[4];
              decode_step(word(buf[p][rt][jh],jl*2),word(buf[p][rt][jh],jl*2+1),word(buf[p][rt][2],j),a);
#pragma unroll
              for (int nt=0; nt<NT; ++nt)
                mma16816(acc[rt][nt],a,word(xv[nt],jl*2),word(xv[nt],jl*2+1));
            }
          }
        }
''',
 'step':r'''
#pragma unroll
        for (int j=0; j<4; ++j) {
          uint2 xv[NT];
#pragma unroll
          for (int nt=0; nt<NT; ++nt) {
            const uint32_t sa=smem_u32(xrow[nt]+bb*64+j*4);
            asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(xv[nt].x),"=r"(xv[nt].y) : "r"(sa));
          }
#pragma unroll
          for (int rt=0; rt<RT; ++rt) {
            uint32_t a[4];
            decode_step(word(buf[p][rt][j>>1],(j&1)*2),word(buf[p][rt][j>>1],(j&1)*2+1),word(buf[p][rt][2],j),a);
#pragma unroll
            for (int nt=0; nt<NT; ++nt)
              mma16816(acc[rt][nt],a,xv[nt].x,xv[nt].y);
          }
        }
'''}
for name,loop in variants.items():
    v=src[:a]+loop+src[b:]
    (H/f'stream_{name}.py').write_text(v)
    tree=ast.parse(v)
    cuda=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='CD_CUDA')
    cases=' '.join(f'CDG_CASE({nt}, {rt}, {p}, 3)' for nt in (1,2,4) for rt in (1,2) for p in (2,4))
    (H/f'stream_{name}.cu').write_text(cuda.replace('CDG_INSTANCES',cases))
print('built half and step activation lifetime experiments')
