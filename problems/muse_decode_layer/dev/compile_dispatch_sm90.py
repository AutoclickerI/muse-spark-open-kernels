import ast, hashlib, importlib.util, os
from pathlib import Path
from torch.utils.cpp_extension import load_inline
p=Path(__file__).parents[1]/'cand_r3_moe_dispatch.py'
tree=ast.parse(p.read_text())
vals={}
for n in tree.body:
    if isinstance(n,ast.Assign):
        for t in n.targets:
            if isinstance(t,ast.Name) and t.id in {'DP_CUDA_SRC','DP_CPP_SRC','GROUPED_SRC'}:
                vals[t.id]=ast.literal_eval(n.value)
os.environ['TORCH_CUDA_ARCH_LIST']='9.0a'
os.environ['MAX_JOBS']='2'
for name in ('DP','GROUPED'):
    src=vals[name+'_CUDA_SRC'] if name=='DP' else vals['GROUPED_SRC']
    cpp=vals['DP_CPP_SRC'] if name=='DP' else ('void route(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t);\nvoid gemm(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t);')
    functions=['max_smem_attr','smem_needed','resident_blocks','run'] if name=='DP' else ['route','gemm']
    print(name,hashlib.sha256(src.encode()).hexdigest(),flush=True)
    load_inline(name='muse_decode_dispatch_'+name.lower()+'_'+hashlib.sha1(src.encode()).hexdigest()[:12], cpp_sources=cpp,cuda_sources=src,functions=functions,extra_cuda_cflags=['-O3','-std=c++20','-lineinfo','--ptxas-options=-v'],verbose=True)
    print(name,'SM90A COMPILE PASS',flush=True)
