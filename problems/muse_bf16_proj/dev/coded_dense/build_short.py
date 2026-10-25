"""Reduce live WGMMA A operands from 16/32 registers to 8/16, committing two K16 steps together."""
from pathlib import Path
import runpy
H=Path(__file__).resolve().parent
runpy.run_path(str(H/'build_wg.py'))
k=(H/'wg_kernel.cu').read_text()
helper='''
__device__ __forceinline__ void wg_keep4(const uint32_t (&a)[4],uint32_t hold) {
  // A real read after wait keeps the older A group distinct in ptxas register allocation.
  // Only lane 0 publishes 16 bytes per warp; scratch is separate from X and barriers.
  asm volatile("{.reg .pred p; setp.eq.u32 p,%5,0; @p st.volatile.shared.v4.u32 [%0],{%1,%2,%3,%4};}"
    ::"r"(hold),"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(threadIdx.x&31):"memory");
}
'''
k=k.replace('template<int MT,int WB>\n__global__',helper+'\ntemplate<int MT,int WB>\n__global__',1)
k=k.replace('uint32_t A[WB][4][4];','uint32_t A[WB][2][4];')
k=k.replace('  float acc[MT/2];','  const uint32_t hold=bar0+((nch*8+15)&~15)+warp*16;\n  float acc[MT/2];')
a=k.index('      const int z=WB==1?0:p;');b=k.index('\n    }\n  }\n  wg_wait<0>();',a)
k=k[:a]+'''#pragma unroll
      for(int jh=0;jh<2;++jh) {
        const int z=WB==1?0:jh;
        if(b+p>0 || jh>=WB) {
          wg_wait<WB-1>();wg_keep4(A[z][0],hold);wg_keep4(A[z][1],hold);
        }
#pragma unroll
        for(int jl=0;jl<2;++jl) {
          const int j=2*jh+jl;
          decode_step(word(buf[p][j>>1],(j&1)*2),word(buf[p][j>>1],(j&1)*2+1),word(buf[p][2],j),A[z][jl]);
        }
        wg_protect<MT>(acc);wg_fence();
#pragma unroll
        for(int jl=0;jl<2;++jl)wg_atom<MT>(acc,A[z][jl],xs+(size_t)(b+p)*MT*64,2*jh+jl);
        wg_commit();
      }
#pragma unroll
      for(int q=0;q<3;++q)buf[p][q]=ld_w_if(wp+(b+p+2)*step+q*32,b+p+2<nb);
'''+k[b:]
k=k.replace('for(int z=0;z<WB;++z)wg_keep(A[z]);','for(int z=0;z<WB;++z){wg_keep4(A[z][0],hold);wg_keep4(A[z][1],hold);}')
wg=(H/'wg.py').read_text().replace('from coded import CD_CUDA, cd_encode, cd_decode\n','').replace('WG_KERNEL = None','WG_KERNEL = '+repr(k))
wg=wg.replace('bytes=nb*MT*128+((nb+3)/4)*8;', 'bytes=nb*MT*128+((((nb+3)/4)*8+15)&~15)+NW*16;')
body=(H/'coded.py').read_text()+'\n'+wg
(H/'short_body.py').write_text(body)
pre=(H/'lab_body.py').read_text().split('\ndef lab(',1)[0].replace('__CL_VS__','(3,)').replace('__CL_SHAPES__','((8,1,2),)')
lab=(H/'rt2_lab.py.inc').read_text().replace('two 64-row A tiles reuse one CTA TMA X copy vs one tile','K32 fragment groups vs K64 groups, live A operand registers cut 16/32 -> 8/16')
lab=lab.replace("'META RT2 decodes each weight once; X copies per output row halved, register bank larger (sm90 162/165 registers at T32 WB1/2, no stack/spills/C75)'", "'META same 12-bit bank and TMA X layout; one decode per weight; 4 HGMMA per K64 at every T; two K16 ops per commit; shorter operand lifetimes; predicated scratch store after wait enforces distinct A allocation'")
(H/'short_lab.py').write_text('import torch\nimport torch.nn.functional as F\nimport triton\nimport triton.language as tl\n'+pre+'\nSOURCES = '+repr({'one':(H/'wg_body.py').read_text(),'two':body})+'\n'+lab)
s=(H/'test_wg.py').read_text().replace('from coded import cd_module\nfrom wg_frozen import wg_encode,wg_decode,wg_gemv,wg_module','from short_body import cd_module,wg_encode,wg_decode,wg_gemv,wg_module').replace("'wg_local.json'","'short_validation.json'").replace("wg_kernel_sha256=hashlib.sha256((H/'wg_kernel.cu').read_bytes()).hexdigest()", "short_body_sha256=hashlib.sha256((H/'short_body.py').read_bytes()).hexdigest()")
(H/'test_short.py').write_text(s)
print(H/'short_body.py')
