"""Move compressed-weight prefetch from thread registers to a double-buffered bulk-copy ring."""
from pathlib import Path
import runpy

H = Path(__file__).resolve().parent
runpy.run_path(str(H / 'build_wg.py'))
k = (H / 'wg_kernel.cu').read_text()
k = k.replace('const int r=(blockIdx.x*nw+warp), step=ks*96;', 'const int r=(blockIdx.x*nw+warp);')
helper = '''
__device__ __forceinline__ void code_wait(uint32_t bar,int phase) {
  uint32_t done;
  do {
    asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p,[%1],%2; selp.u32 %0,1,0,p; }"
      : "=r"(done):"r"(bar),"r"(phase):"memory");
  } while(!done);
}
'''
k = k.replace('template<int MT,int WB>\n__global__', helper + '\ntemplate<int MT,int WB>\n__global__', 1)
a = k.index('  const uint4* wp=');b = k.index('  if(threadIdx.x==0) {', a)
k = k[:a] + '''  const float sc=ld_f32(scales+r);
  const int ea=ld_i32(eoff+(size_t)r*KB+kb0),eb=ld_i32(eoff+(size_t)r*KB+kb1);
  constexpr int CH=4;
  const int nch=(nb+CH-1)/CH;
  const uint32_t bar0=smem_u32(sm+nb*MT*128);
  const int code_off=(nb*MT*128+nch*8+127)&~127;
  const uint32_t cb0=smem_u32(sm+code_off+2*nw*1536);
  auto codes=reinterpret_cast<uint4*>(sm+code_off);
  auto issue=[&](int b) {
    const int slot=b&1;
    mbar_expect(cb0+slot*8,nw*1536);
    const void* src=Wc+((size_t)(blockIdx.x*nw)*rs+(size_t)(kb0+b)*ks)*96;
    bulk_g2s(smem_u32(codes+slot*nw*96),src,nw*1536,cb0+slot*8);
  };
  pdl_release();
  if(!prew)pdl_wait();
''' + k[b:]
k = k.replace('    for(int c=0;c<nch;++c)mbar_init(bar0+c*8,1);', '    for(int c=0;c<nch;++c)mbar_init(bar0+c*8,1);\n    mbar_init(cb0,1);mbar_init(cb0+8,1);')
k = k.replace('  if(prew)pdl_wait();', '  if(threadIdx.x==0){issue(0);issue(1);}\n  if(prew)pdl_wait();', 1)
old = '''#pragma unroll
      for(int j=0;j<4;++j)
        decode_step(word(buf[p][j>>1],(j&1)*2),word(buf[p][j>>1],(j&1)*2+1),word(buf[p][2],j),A[z][j]);
#pragma unroll
      for(int q=0;q<3;++q)buf[p][q]=ld_w_if(wp+(b+p+2)*step+q*32,b+p+2<nb);'''
new = '''      code_wait(cb0+p*8,(b/2)&1);
      const uint4* cp=codes+(p*nw+warp)*96+lane;
      const uint4 lo0=cp[0],lo1=cp[32],hi=cp[64];
#pragma unroll
      for(int j=0;j<4;++j)
        decode_step(word(j<2?lo0:lo1,(j&1)*2),word(j<2?lo0:lo1,(j&1)*2+1),word(hi,j),A[z][j]);
      // No thread can overwrite this slot until every warp has consumed its codes.
      __syncthreads();
      if(threadIdx.x==0&&b+p+2<nb)issue(b+p+2);'''
assert k.count(old) == 1
k = k.replace(old, new)
wg = (H / 'wg.py').read_text().replace('from coded import CD_CUDA, cd_encode, cd_decode\n', '')
wg = wg.replace('WG_KERNEL = None', 'WG_KERNEL = ' + repr(k))
wg = wg.replace('bytes=nb*MT*128+((nb+3)/4)*8;', 'bytes=((nb*MT*128+((nb+3)/4)*8+127)&~127)+2*NW*1536+16;')
wg = wg.replace('if(M<1||M>32||', 'if(rs!=1||M<1||M>32||')
# WB1 serializes (C7513) with this shared-load schedule. Keep it local and
# instantiate only the native-safe two-group schedule in the candidate.
wg = wg.replace('WG_CASE(8,1) WG_CASE(16,1) WG_CASE(32,1) ', '')
body = (H / 'coded.py').read_text() + '\n' + wg
(H / 'bulk_body.py').write_text(body)
test = (H / 'test_wg.py').read_text().replace('from coded import cd_module\nfrom wg_frozen import wg_encode,wg_decode,wg_gemv,wg_module', 'from bulk_body import cd_module,wg_encode,wg_decode,wg_gemv,wg_module')
test = test.replace('for lay in (False,True):', 'for lay in (True,):')
test = test.replace('for WB in (1,2):', 'for WB in (2,):')
# NW is selected after the original shape screen: use the largest supported CTA width.
test = test.replace('smem=K//64//S*MT*128+K//64//S*2', 'smem=((K//64//S*MT*128+K//64//S*2+127)&~127)+2*8*1536+16')
test = test.replace("'wg_local.json'", "'bulk_validation.json'")
test = test.replace("wg_kernel_sha256=hashlib.sha256((H/'wg_kernel.cu').read_bytes()).hexdigest()", "bulk_body_sha256=hashlib.sha256((H/'bulk_body.py').read_bytes()).hexdigest()")
(H / 'test_bulk.py').write_text(test)
pre = (H / 'lab_body.py').read_text().split('\ndef lab(', 1)[0].replace('__CL_VS__', '(3,)').replace('__CL_SHAPES__', '((8,1,2),)')
lab = (H / 'rt2_lab.py.inc').read_text()
lab = lab.replace('two 64-row A tiles reuse one CTA TMA X copy vs one tile', 'bulk-copy double-buffered compressed weights vs register LDG ring')
lab = lab.replace('META RT2 decodes each weight once; X copies per output row halved, register bank larger (sm90 162/165 registers at T32 WB1/2, no stack/spills/C75)', 'META TMA copies contiguous NW*1536B codes into two shared slots; decode once to register A, four HGMMA per K64; CTA barrier protects slot reuse; extra shared bytes=2*NW*1536+padding')
lab = lab.replace('N%(NW*32)', 'N%(NW*16)')
lab = lab.replace('for WB in (1,2):', 'for WB in (2,):')
lab = lab.replace('K//64//S*MT*128+128>maxsmem', '((K//64//S*MT*128+((K//64//S+3)//4)*8+127)&~127)+2*NW*1536+16>maxsmem')
(H / 'bulk_lab.py').write_text('import torch\nimport torch.nn.functional as F\nimport triton\nimport triton.language as tl\n' + pre + '\nSOURCES = ' + repr({'one': (H / 'wg_body.py').read_text(), 'two': body}) + '\n' + lab)
print(H / 'bulk_body.py')
