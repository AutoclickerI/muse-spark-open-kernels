"""Route B second variant: software-pipeline our own direct register-X loads."""
import ast
from pathlib import Path
H=Path(__file__).resolve().parent
body=(H/'register_x.py').read_text();tree=ast.parse(body)
n=next(n for n in tree.body if isinstance(n,ast.Assign) and any(getattr(t,'id','')=='CD_CUDA' for t in n.targets))
s=ast.literal_eval(n.value)
pos=s.index('\n  float acc[RT][NT][4];',s.index('coded_gemv('))
s=s[:pos]+'''
  uint4 xpipe[PIPE][NT][2];
  auto fetch_x = [&](int slot, int block) {
#pragma unroll
    for (int nt=0; nt<NT; ++nt) {
      const uint4* src=reinterpret_cast<const uint4*>(xrow[nt]+block*64);
      xpipe[slot][nt][0]=ld_w_if(src,block<nb);
      xpipe[slot][nt][1]=ld_w_if(src+1,block<nb);
    }
  };
#pragma unroll
  for(int p=0;p<PIPE;++p)fetch_x(p,p);
''' + s[pos:]
a=s.index('        uint4 xv[NT][2];',s.index('coded_gemv('));b=s.index('#pragma unroll\n        for (int j = 0;',a)
s=s[:a]+'''        uint4 xv[NT][2];
#pragma unroll
        for(int nt=0;nt<NT;++nt) {
          xv[nt][0]=xpipe[p][nt][0];xv[nt][1]=xpipe[p][nt][1];
        }
        fetch_x(p,bb+PIPE);
''' + s[b:]
# Activation loads should allocate in L1; streaming weight loads keep their no-allocate hint.
s=s.replace('xpipe[slot][nt][0]=ld_w_if(src,block<nb);','xpipe[slot][nt][0]=ld_x_if(src,block<nb);').replace('xpipe[slot][nt][1]=ld_w_if(src+1,block<nb);','xpipe[slot][nt][1]=ld_x_if(src+1,block<nb);')
pos=s.index('__device__ __forceinline__ uint32_t prmt')
s=s[:pos]+'''__device__ __forceinline__ uint4 ld_x_if(const uint4* p,int pred) {
  uint4 v;
  asm volatile("{ .reg .pred q; setp.ne.b32 q,%5,0; @q ld.global.nc.v4.u32 {%0,%1,%2,%3},[%4]; }"
    : "=r"(v.x),"=r"(v.y),"=r"(v.z),"=r"(v.w) : "l"(p),"r"(pred));
  return v;
}

''' + s[pos:]
lines=body.splitlines(keepends=True)
(H/'pipeline_x.py').write_text(''.join(lines[:n.lineno-1])+'CD_CUDA = '+repr(s)+'\n'+''.join(lines[n.end_lineno:]))
(H/'pipeline_x.cu').write_text(s)
