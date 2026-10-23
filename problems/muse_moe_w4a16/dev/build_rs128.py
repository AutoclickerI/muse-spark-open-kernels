"""Two m64 fragments share each activation stage, retaining the canonical bank."""
import ast
from pathlib import Path
p=Path(__file__).resolve().parent.parent
text=(p/'cand_rs.py').read_text()
tree=ast.parse(text)
node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='GROUPED_SRC' for t in n.targets))
s=ast.literal_eval(node.value)
def replace(a,b):
 global s
 assert s.count(a)==1,(a,s.count(a))
 s=s.replace(a,b)
replace('NFB=MTILES/4;', 'RB=2, NFB=MTILES/(4*RB);')
replace('fb*4+warp)*TB;', 'fb*(4*RB)+warp)*TB;')
replace('float acc[N/2];', 'float acc[RB][N/2];')
replace('for(int j=0;j<N/2;++j)acc[j]=0.f;', 'for(int rb=0;rb<RB;++rb)\n#pragma unroll\n  for(int j=0;j<N/2;++j)acc[rb][j]=0.f;')
replace('uint32_t A[2][4] = {};', 'uint32_t A[2][RB][4] = {};')
a=s.index('   uint4 w=')
b=s.index('   __syncthreads();\n   if(kg+STAGES',a)
s=s[:a]+'''   uint32_t o[RB][16],sr[RB],su[RB];
#pragma unroll
   for(int rb=0;rb<RB;++rb){
    const uint8_t* wp=wt+rb*4*TB;
    uint4 w=*(const uint4*)(wp+(kg*32+lane)*16);
    uint16_t sw=*(const uint16_t*)(wp+WB+(kg/8)*512+lane*16+(kg%8)*2);
    sr[rb]=0x70007000u+(uint32_t)(sw&255)*0x100010u;
    su[rb]=0x70007000u+(uint32_t)(sw>>8)*0x100010u;
    dq8(w.x,w.y,o[rb]);dq8(w.z,w.w,o[rb]+8);
   }
#pragma unroll
   for(int k=0;k<4;++k){
#pragma unroll
    for(int rb=0;rb<RB;++rb){
     const uint32_t* os=o[rb]+(k>>1)*8+(k&1)*4;
     uint32_t* a=A[k&1][rb];
     a[0]=hmul2(os[0],sr[rb]);a[1]=hmul2(os[1],su[rb]);a[2]=hmul2(os[2],sr[rb]);a[3]=hmul2(os[3],su[rb]);
     protect<N>(acc[rb]);
    }
    wfence();
#pragma unroll
    for(int rb=0;rb<RB;++rb)atom<N>(acc[rb],A[k&1][rb],shared+st*XBYTES,k);
    wcommit();
#pragma unroll
    for(int rb=0;rb<RB;++rb)protect<N>(acc[rb]);
    wwait<1>();
#pragma unroll
    for(int rb=0;rb<RB;++rb)keep(A[(k&1)^1][rb]);
   }
   wwait<0>();
#pragma unroll
   for(int rb=0;rb<RB;++rb){protect<N>(acc[rb]);keep(A[1][rb]);}
''' +s[b:]
a=s.index('  const float sc=gs[e]*256.f;')
b=s.index('  __syncthreads();\n }',a)
chunk=s[a:b]
chunk=chunk.replace('  for(int j=0;', '  for(int rb=0;rb<RB;++rb)\n#pragma unroll\n  for(int j=0;',1)
chunk=chunk.replace('acc[4*j','acc[rb][4*j')
chunk=chunk.replace('fb*32+','(fb*RB+rb)*32+').replace('fb*64+','(fb*RB+rb)*64+')
s=s[:a]+chunk+s[b:]
lines=text.splitlines(keepends=True)
start=sum(map(len,lines[:node.lineno-1]));end=sum(map(len,lines[:node.end_lineno]))
new=text[:start]+'GROUPED_SRC = r"""'+s+'"""\n'+text[end:]
(p/'cand_rs128.py').write_text(new)
