import importlib.util
import os
import sys
from pathlib import Path
import hashlib
os.environ['TORCH_CUDA_ARCH_LIST']='9.0a'
os.environ['MAX_JOBS']='2'
from torch.utils.cpp_extension import load_inline
p=Path(__file__).resolve().parent.parent
for filename in sys.argv[1:]:
 spec=importlib.util.spec_from_file_location('source',p/filename)
 m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 if hasattr(m,'GROUPED_SRC'):
  src=m.GROUPED_SRC
  cpp=('void route(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t);\n'
       'void gemm(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t);')
  functions=['route','gemm']
  if 'void configure()' in src:
   cpp=cpp.replace('int64_t,int64_t,int64_t,int64_t);','int64_t,int64_t,int64_t,int64_t,int64_t);')+'\nvoid configure();'
   functions.append('configure')
 else:
  src,cpp=m.CUDA_SRC,m.CPP_SRC
  functions=['max_smem_attr','smem_needed','resident_blocks','run']
 print('SM90a OFFLINE',filename,hashlib.sha256(src.encode()).hexdigest(),flush=True)
 load_inline(name='moe_round2_90a_'+hashlib.sha1(src.encode()).hexdigest()[:12],
             cpp_sources=cpp,cuda_sources=src,functions=functions,
             extra_cuda_cflags=['-O3','-std=c++20','-lineinfo','--ptxas-options=-v'],verbose=True)
 print('OFFLINE COMPILE PASS',filename,flush=True)
