"""Exact timed packet_sv proof: every u8 scale, delta, bit position, raw and compressed."""
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.cpp_extension import load_inline
H=Path(__file__).resolve().parent
source=(H/'pack5_cell.py').read_text()
cu=next(ast.literal_eval(n.value) for n in ast.parse(source).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='P3B_CUDA')
a=cu.index('__device__ __forceinline__ unsigned spread5');z=cu.index('constexpr int BK=',a);helper=cu[a:z]
cpp='#include <pybind11/pybind11.h>\n#include <cstdint>\nvoid run(int64_t,int64_t,int64_t,int64_t);'
probe='#include <cuda_runtime.h>\n#include <cstdint>\n'+helper+'''
__global__ void probe(const unsigned char*in,unsigned*out){
 __shared__ __align__(16) unsigned char stage[144];int t=threadIdx.x;
 for(int i=t;i<144;i+=128)stage[i]=in[blockIdx.x*144+i];__syncthreads();
 if(t<8)((uint4*)(out+blockIdx.x*32))[t]=packet_sv(stage,t);
}
void run(int64_t a,int64_t b,int64_t n,int64_t stream){probe<<<n,128,0,(cudaStream_t)stream>>>((const unsigned char*)a,(unsigned*)b);}
'''
tag=hashlib.sha256(probe.encode()).hexdigest()[:12]
mod=load_inline(name='p3b_packet_proof_'+tag,cpp_sources=cpp,cuda_sources=probe,functions=['run'],no_implicit_headers=True,with_pytorch_error_handling=False,extra_cuda_cflags=['-O3','-std=c++20'])
inputs=[];expected=[]
def add(values,base,raw):
 values=np.asarray(values,dtype=np.uint8);stage=np.zeros(144,dtype=np.uint8)
 if raw:stage[:128]=values
 else:
  delta=(values.astype(np.uint16)-base);bits=((delta[:,None]>>np.arange(5))&1).astype(np.uint8).reshape(-1)
  stage[:80]=np.packbits(bits,bitorder='little')
 stage[128]=base;stage[129]=int(raw);inputs.append(stage)
 # Independent whole-word PERMW inverse target, unrelated to bitfield extraction implementation.
 words=values.copy().view('<u4').astype(np.uint64)
 expected.append((((words<<4)|(words>>28))&0xffffffff).astype(np.uint32))
for base in range(225):
 for phase in range(32):add(base+(np.arange(128)*13+phase)%32,base,False)
for code in range(256):add(np.full(128,code,dtype=np.uint8),0,True)
add(np.arange(128,dtype=np.uint8),0,True);add(np.arange(128,256,dtype=np.uint8),0,True)
x=torch.from_numpy(np.stack(inputs)).cuda();want=torch.from_numpy(np.stack(expected)).cuda();got=torch.empty_like(want)
mod.run(x.data_ptr(),got.data_ptr(),len(inputs),torch.cuda.current_stream().cuda_stream);torch.cuda.synchronize()
assert torch.equal(got,want)
result=dict(cuda_sha256=hashlib.sha256(cu.encode()).hexdigest(),probe_sha256=hashlib.sha256(probe.encode()).hexdigest(),groups=len(inputs),compressed_groups=225*32,raw_groups=258,exact_scale_bytes=len(inputs)*128,all_uint8_codes=True,all_delta_codes=True,every_packed_alignment=True,exact=True,device='GPU0 sm120',limitation='Exact same timed decode helper; portable GPU execution plus native static compilation, not native H200 raw-branch execution.')
(H/'packet_decoder_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
