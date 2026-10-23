import sys,os
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import torch
from torch.utils.cpp_extension import load_inline
os.environ['TORCH_CUDA_ARCH_LIST']='9.0a'
src=Path(__file__).with_name('moe_w4a16_canonical.cu').read_text()
load_inline(name='muse_prefill_canonical_hopper_compile',cpp_sources='void moe_gemm(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t,int64_t);',cuda_sources=src,functions=['moe_gemm'],extra_cuda_cflags=['-O3','-std=c++20','--expt-relaxed-constexpr','-lineinfo','--ptxas-options=-v'],verbose=True)
print('SM90a compile PASS')
