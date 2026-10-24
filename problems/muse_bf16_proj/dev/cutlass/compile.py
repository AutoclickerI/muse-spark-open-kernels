"""Compile the actual shipped Hopper collectives, without launching on sm_120."""
from pathlib import Path
import os,time
from torch.utils.cpp_extension import load_inline
src=Path(__file__).with_name('kernel.cu').read_text()
os.environ['TORCH_CUDA_ARCH_LIST']='9.0a'
os.environ['MAX_JOBS']='2'
t=time.monotonic()
p=load_inline(name='muse_verify_cutlass_proto',cpp_sources='',cuda_sources=src,
  is_python_module=False,no_implicit_headers=True,
  extra_include_paths=['/opt/sglang/lib/python3.12/site-packages/flashinfer/data/cutlass/include'],
  extra_cuda_cflags=['-O3','-DNDEBUG','-std=c++17','--expt-relaxed-constexpr'],extra_ldflags=['-lcuda'],verbose=True)
print('COMPILED',p,'seconds',time.monotonic()-t)
