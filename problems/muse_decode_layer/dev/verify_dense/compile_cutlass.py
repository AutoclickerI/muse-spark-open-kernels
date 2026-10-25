from pathlib import Path
import os,hashlib
from torch.utils.cpp_extension import load_inline
HERE=Path(__file__).resolve().parent
s=(HERE/'cutlass.cu').read_text();os.environ['TORCH_CUDA_ARCH_LIST']='9.0a'
root='/opt/sglang/lib/python3.12/site-packages/flashinfer/data/cutlass'
path=load_inline(name='muse_vdense_cutlass_'+hashlib.sha256(s.encode()).hexdigest()[:14],cpp_sources='',cuda_sources=s,functions=None,is_python_module=False,no_implicit_headers=True,
 extra_include_paths=[root+'/include',root+'/tools/util/include','/usr/local/cuda/include/cccl'],extra_cuda_cflags=['-O3','-std=c++17','--expt-relaxed-constexpr','--ptxas-options=-v'],verbose=True)
print(path)
