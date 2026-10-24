"""Check the actual V9 device conversion on every finite E4M3 encoding and raw bank preservation."""
from pathlib import Path
import sys
import torch
from torch.utils.cpp_extension import load_inline
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tools'))
import local_eval
mod=local_eval.load_module('best_scales',ROOT/'problems/muse_decode_layer/cand_r3_best.py')
src=mod.V9_CUDA_SRC
helper=src[src.index('__device__ __forceinline__ uint32_t scale_e4m3'):src.index('__device__ __forceinline__ void mma16816')]
cu='#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n'+helper+'''
__global__ void convert(const uint8_t* x,int32_t* y,int n){int i=threadIdx.x;if(i<n)y[i]=(int32_t)scale_e4m3(x[i]);}
torch::Tensor check(torch::Tensor x){auto y=torch::empty({x.numel()},x.options().dtype(at::kInt));
convert<<<1,256,0,at::cuda::getCurrentCUDAStream()>>>(x.data_ptr<uint8_t>(),y.data_ptr<int32_t>(),x.numel());return y;}
'''
ext=load_inline(name='best_scale_check',cpp_sources='torch::Tensor check(torch::Tensor x);',cuda_sources=cu,
                functions=['check'],extra_cuda_cflags=['-O3','-std=c++20'])
raw=torch.arange(256,device='cuda').to(torch.uint8)
sf=raw.view(torch.float8_e4m3fn)
finite=torch.isfinite(sf.float())
expected=(sf.float()*float(2**118)).to(torch.bfloat16).view(torch.int16).to(torch.int64)&65535
expected=(expected|(expected<<16)).to(torch.int32)
actual=ext.check(raw)
assert torch.equal(actual[finite],expected[finite]),(raw[(actual!=expected)&finite],actual[(actual!=expected)&finite],expected[(actual!=expected)&finite])
w=torch.randint(256,(16,2048),device='cuda',dtype=torch.uint8)
scales=raw.repeat(16).view(16,256).contiguous()
scales=torch.where((scales&127)==127,torch.zeros_like(scales),scales)
bank=mod._repack(w,scales.view(mod.FP8),torch.arange(16,device='cuda').view(1,16))
restored=bank[:,32768:].reshape(1,8,8,4,8,2).permute(0,5,2,1,4,3).reshape(16,256)
assert torch.equal(restored,scales)
torch.cuda.synchronize()
print('PASS: all 254 finite E4M3 encodings decode exactly on GPU; bank retains exact stored E4M3 bytes')
