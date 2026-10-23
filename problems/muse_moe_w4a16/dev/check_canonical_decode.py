"""Exhaustive finite E4M3 codes x all E2M1 codes through V6's exact tile reader."""
from pathlib import Path
import importlib.util
import sys
import torch
from torch.utils.cpp_extension import load_inline
p=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(p))
from nvfp4 import dequantize
spec=importlib.util.spec_from_file_location('cand',p/'cand_rs.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
# All finite scale encodings, including subnormals, negative values and both zeros.
sfbytes=torch.arange(256,device='cuda',dtype=torch.int32).to(torch.uint8)
sfbytes[(sfbytes&127)==127]=0
sf=sfbytes[None,:].expand(16,256).contiguous().view(torch.float8_e4m3fn)
codes=(torch.arange(4096,device='cuda')[None,:]+torch.arange(16,device='cuda')[:,None])%16
packed=(codes[:,::2]|(codes[:,1::2]<<4)).to(torch.uint8)
tile=c._repack(packed,sf,torch.arange(16,device='cuda').view(1,16))
src=c.GROUPED_SRC+r'''
__global__ void decode_test(const uint8_t* wt,__nv_bfloat16* out){
 int kg=blockIdx.x,lane=threadIdx.x,r=lane>>2,q=lane&3;
 uint4 w=*(const uint4*)(wt+(kg*32+lane)*16);
 uint16_t sw=*(const uint16_t*)(wt+32768+(kg/8)*512+lane*16+(kg%8)*2);
 uint32_t sr=0x70007000u+(uint32_t)(sw&255)*0x100010u;
 uint32_t su=0x70007000u+(uint32_t)(sw>>8)*0x100010u;
 uint32_t o[16];canonical::dq8(w.x,w.y,o);canonical::dq8(w.z,w.w,o+8);
 for(int s=0;s<4;++s){const uint32_t* os=o+(s>>1)*8+(s&1)*4;
  for(int j=0;j<4;++j){
   uint32_t pair=canonical::hmul2(canonical::hmul2(os[j],(j&1)?su:sr),0x43804380u);
   int row=r+8*(j&1),col=kg*64+q*16+s*4+2*(j>>1);
   *(uint32_t*)(out+row*4096+col)=pair;
  }
 }
}
void decode(torch::Tensor x,torch::Tensor y){decode_test<<<64,32,0,at::cuda::getCurrentCUDAStream()>>>(x.data_ptr<uint8_t>(),(__nv_bfloat16*)y.data_ptr());C10_CUDA_KERNEL_LAUNCH_CHECK();}
'''
mod=load_inline(name='muse_canonical_exact_decode',cpp_sources='void decode(torch::Tensor,torch::Tensor);',cuda_sources=src,functions=['decode'],extra_cuda_cflags=['-O3','-std=c++20'],verbose=False)
out=torch.empty((16,4096),device='cuda',dtype=torch.bfloat16)
mod.decode(tile,out)
expected=dequantize(packed,sf,1.).bfloat16()
assert torch.equal(out,expected),f'{int((out!=expected).sum())} mismatches'
print('PASS: all 16 E2M1 codes x all 254 finite E4M3 encodings, including negative/subnormal/zero scales. Exact BF16 values before FP32 global scale.',flush=True)
