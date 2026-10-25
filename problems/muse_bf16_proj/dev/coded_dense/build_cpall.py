"""Build one-barrier cooperative activation staging vs chunked bulk copies, especially for o/slice."""
from pathlib import Path
import runpy
H=Path(__file__).resolve().parent
ns=runpy.run_path(str(H/'build_staging.py'))
new=ns['new']
a=new.index('  if (threadIdx.x == 0) {\n    for (int c = 0; c < nch; ++c) mbar_init')
b=new.index('  const __nv_bfloat16* xrow[NT];',a)
cp=r'''
  if (prew) pdl_wait();
  // All CTA threads stage the entire split while the weight preloads are in flight.
  // One wait and CTA barrier, no per-chunk mbarrier initialization or producer warp.
  for (int i=threadIdx.x; i<M*(L/8); i+=blockDim.x) {
    const int row=i/(L/8), col=(i-row*(L/8))*8;
    const uint32_t sa=smem_u32(xs+(size_t)row*RS+col);
    const void* ga=X+(size_t)row*x_stride+k0+col;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(sa), "l"(ga));
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
  for (int i=threadIdx.x; i<L/8; i+=blockDim.x)
    reinterpret_cast<uint4*>(xs+(size_t)M*RS)[i]=make_uint4(0u,0u,0u,0u);
  asm volatile("cp.async.wait_group 0;" ::: "memory");
  __syncthreads();
'''
variant=(new[:a]+cp+new[b:]).replace('    if (b % CB == 0) mbar_wait0(mbar0 + 8 * (b / CB));\n','')
# The arm decodes each weight exactly once for all token tiles.
variants={'cpall':variant}
# This payload reuses the staging lab's paired harness but contains only this new mechanism.
(H/'cpall_lab.py').write_text('import triton\nimport triton.language as tl\n'+new+'\n'+ns['pre']+'\nSOURCES = '+repr(variants)+'\n'+ns['body'].replace('activation lifetimes half-k32/k16, chunk CB2/4/8','all-thread cp.async, single CTA barrier vs per-chunk bulk copies'))
print(H/'cpall_lab.py')
