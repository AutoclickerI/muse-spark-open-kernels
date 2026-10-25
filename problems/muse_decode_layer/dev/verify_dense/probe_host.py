from pathlib import Path
import hashlib,time,os
os.environ['TORCH_CUDA_ARCH_LIST']='9.0a'
from torch.utils.cpp_extension import load_inline
h=Path(__file__).resolve().parent;s=(h/'cutlass_host.cpp').read_text();root='/opt/sglang/lib/python3.12/site-packages/flashinfer/data/cutlass';start=time.monotonic()
p=load_inline(name='muse_vdense_host_'+hashlib.sha256(s.encode()).hexdigest()[:12],cpp_sources='',cuda_sources=s,functions=None,is_python_module=False,no_implicit_headers=True,
 extra_include_paths=[root+'/include',root+'/tools/util/include','/usr/local/cuda/include','/usr/local/cuda/include/cccl'],extra_cuda_cflags=['-O2','-std=c++17','--expt-relaxed-constexpr'],extra_ldflags=['-L/usr/local/cuda/lib64','-L/usr/local/cuda/lib64/stubs','-lcudart','-lcuda'],verbose=True)
print(p,'seconds',time.monotonic()-start)
