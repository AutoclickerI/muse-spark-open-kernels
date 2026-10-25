"""Two 64-row WGMMA A tiles share each TMA activation copy and CTA, retaining the measured 12-bit codec."""
from pathlib import Path
import runpy
H=Path(__file__).resolve().parent
runpy.run_path(str(H/'build_cpall.py'))
runpy.run_path(str(H/'build_wg.py'))
s=(H/'wg_kernel.cu').read_text()
a=s.index('template<int MT,int WB>\n__global__');head=s[:a];k=s[a:]
k=k.replace('  extern __shared__','  constexpr int RT=2;\n  extern __shared__',1)
k=k.replace('const int r=(blockIdx.x*nw+warp), step=ks*96;', 'const int r=(blockIdx.x*nw+warp)*RT, step=ks*96;')
a=k.index('  const uint4* wp=');b=k.index('  pdl_release();',a)
k=k[:a]+'''  const uint4* wp[RT];float SC[RT];int EA[RT],EB[RT];
#pragma unroll
  for(int rt=0;rt<RT;++rt){
    wp[rt]=Wc+((size_t)(r+rt)*rs+(size_t)kb0*ks)*96+lane;
    SC[rt]=ld_f32(scales+r+rt);
    EA[rt]=ld_i32(eoff+(size_t)(r+rt)*KB+kb0);EB[rt]=ld_i32(eoff+(size_t)(r+rt)*KB+kb1);
  }
  uint4 buf[2][RT][3];
'''+k[b:]
k=k.replace('    for(int q=0;q<3;++q)buf[p][q]=ld_w_if(wp+p*step+q*32,p<nb);', '''    for(int q=0;q<3;++q)
#pragma unroll
      for(int rt=0;rt<RT;++rt)buf[p][rt][q]=ld_w_if(wp[rt]+p*step+q*32,p<nb);''')
k=k.replace('  float acc[MT/2];\n#pragma unroll\n  for(int i=0;i<MT/2;++i)acc[i]=0.f;\n  uint32_t A[WB][4][4];', '''  float ACC[RT][MT/2];
#pragma unroll
  for(int rt=0;rt<RT;++rt)
#pragma unroll
    for(int i=0;i<MT/2;++i)ACC[rt][i]=0.f;
  uint32_t A[WB][RT][4][4];''')
k=k.replace('        wg_keep(A[z]);','        for(int rt=0;rt<RT;++rt)wg_keep(A[z][rt]);')
a=k.index('#pragma unroll\n      for(int j=0;j<4;++j)\n        decode_step');b=k.index('      wg_commit();',a)
k=k[:a]+'''#pragma unroll
      for(int rt=0;rt<RT;++rt)
#pragma unroll
        for(int j=0;j<4;++j)
          decode_step(word(buf[p][rt][j>>1],(j&1)*2),word(buf[p][rt][j>>1],(j&1)*2+1),word(buf[p][rt][2],j),A[z][rt][j]);
#pragma unroll
      for(int rt=0;rt<RT;++rt)
#pragma unroll
        for(int q=0;q<3;++q)buf[p][rt][q]=ld_w_if(wp[rt]+(b+p+2)*step+q*32,b+p+2<nb);
#pragma unroll
      for(int rt=0;rt<RT;++rt)wg_protect<MT>(ACC[rt]);
      wg_fence();
#pragma unroll
      for(int j=0;j<4;++j)
#pragma unroll
        for(int rt=0;rt<RT;++rt)wg_atom<MT>(ACC[rt],A[z][rt][j],xs+(size_t)(b+p)*MT*64,j);
'''+k[b:]
k=k.replace('  for(int z=0;z<WB;++z)wg_keep(A[z]);\n  wg_protect<MT>(acc);','''  for(int z=0;z<WB;++z)
#pragma unroll
    for(int rt=0;rt<RT;++rt)wg_keep(A[z][rt]);
#pragma unroll
  for(int rt=0;rt<RT;++rt){
    float (&acc)[MT/2]=ACC[rt];const float sc=SC[rt];const int ea=EA[rt],eb=EB[rt];
    wg_protect<MT>(acc);''')
k=k.replace('+(size_t)tok*N+r*16+g;','+(size_t)tok*N+(r+rt)*16+g;')
k=k.rstrip()[:-1]+'  }\n}\n'
wg=(H/'wg.py').read_text().replace('from coded import CD_CUDA, cd_encode, cd_decode\n','')
wg=wg.replace('WG_KERNEL = None','WG_KERNEL = '+repr(head+k))
wg=wg.replace('dim3(N/(NW*16),S)','dim3(N/(NW*32),S)').replace('N%(NW*16)','N%(NW*32)')
(H/'rt2_body.py').write_text((H/'coded.py').read_text()+'\n'+wg)
print(H/'rt2_body.py')

old=(H/'wg_body.py').read_text()
pre=(H/'lab_body.py').read_text().split('\ndef lab(',1)[0].replace('__CL_VS__','(3,)').replace('__CL_SHAPES__','((8,1,2),)')
(H/'rt2_lab.py').write_text('import torch\nimport torch.nn.functional as F\nimport triton\nimport triton.language as tl\n'+pre+'\nSOURCES = '+repr({'one':old,'two':(H/'rt2_body.py').read_text()})+'\n'+(H/'rt2_lab.py.inc').read_text())
