from pathlib import Path
import os,importlib.util
os.environ['TORCH_CUDA_ARCH_LIST']='9.0a'
from torch.utils.cpp_extension import load_inline
p=Path('problems/muse_prefill_layer/cand_expert_portfolio.py')
s=importlib.util.spec_from_file_location('portfolio_compile',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
load_inline(name='prefill_portfolio_sm90_compile',cpp_sources='void gemm(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t,int64_t);',cuda_sources=m.GROUPED_SRC,functions=['gemm'],extra_cuda_cflags=['-O3','-std=c++20','-lineinfo','--ptxas-options=-v'],verbose=True)
import triton
from triton.compiler import ASTSource
from triton.backends.compiler import GPUTarget
sig=dict(WB='*u8',WW='*i32',OUT='*i32',COUNTS='*i32',ROWS='constexpr',GATED='constexpr',KB='constexpr')
for gated,rows in ((True,8192),(False,4096)):
 for kb in (1024,2048):
  c=triton.compile(ASTSource(m._canonical_expand_coalesced,signature=sig,constexprs=dict(ROWS=rows,GATED=gated,KB=kb)),target=GPUTarget('cuda',90,32),options=dict(num_warps=8))
  print('SM90 canonical coalesced expand',gated,kb,c.metadata,flush=True)
  (p.parent/'dev'/f'canonical_coalesced_{gated}_{kb}_sm90.cubin').write_bytes(c.asm['cubin'])
print('SM90 portfolio compile PASS')
