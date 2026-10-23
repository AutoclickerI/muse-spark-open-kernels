import sys,os
sys.path.insert(0,'problems/muse_prefill_layer/dev')
from torch.utils.cpp_extension import load_inline
os.environ['TORCH_CUDA_ARCH_LIST']='9.0a'
from pathlib import Path
load_inline(name='muse_prefill_ss_hopper_compile',cpp_sources='void moe_gemm(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t, int64_t, int64_t);',cuda_sources=Path('problems/muse_prefill_layer/dev/moe_w4a16_ss.cu').read_text(),functions=['moe_gemm'],extra_cuda_cflags=['-O3','-std=c++20','--expt-relaxed-constexpr','-lineinfo','--ptxas-options=-v'],verbose=True)
print('SM90a compile passed')
