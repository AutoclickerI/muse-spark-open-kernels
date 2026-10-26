"""Run actual cancellation helper and packet decoder on zero/extreme and raw-fallback inputs."""
import ast,hashlib,json,linecache,sys
from pathlib import Path
import torch
from torch.utils.cpp_extension import load_inline
D=Path(__file__).resolve().parent;R=D.parents[4]
cs=(D/'cancel_cell.py').read_text();n={'torch':torch};exec(compile(cs,'<cancel extreme cell>','exec'),n)
cuda=n['RVC_CUDA'];a=cuda.index('__device__ __forceinline__ float cancel_norm');b=cuda.index('\nstruct PrefetchState',a);helper=cuda[a:b];cancel_sha=hashlib.sha256(helper.encode()).hexdigest()
source='#include <cuda_runtime.h>\n#include <cstdint>\nconstexpr int HD=4096;\n'+helper+'''
__global__ void check(const float* meta,const float* sums,float* output){int e=threadIdx.x;if(e<32)output[e]=cancel_norm(meta,e,sums[e]);}
void run(int64_t m,int64_t s,int64_t o,int64_t st){check<<<1,32,0,(cudaStream_t)st>>>((float*)m,(float*)s,(float*)o);}
'''
arch=''.join(map(str,torch.cuda.get_device_capability()));mod=load_inline(name='p3a_cancel_ext_'+hashlib.sha256(source.encode()).hexdigest()[:12],cpp_sources='#include <cstdint>\n#include <pybind11/pybind11.h>\nvoid run(int64_t,int64_t,int64_t,int64_t);',cuda_sources=source,functions=['run'],extra_cuda_cflags=['-O3','-arch=sm_'+arch],no_implicit_headers=True,with_pytorch_error_handling=False)
scale=torch.tensor([0.,-0.,1.,-1.,1e-40,-1e-40,1e-30,-1e-30,1e-20,-1e-20,1e-10,-1e-10,1e10,-1e10,1e20,-1e20,1e30,-1e30,1e38,-1e38,1e-12,-1e-12,1e-6,-1e-6,3.,-3.,1e-25,-1e-25,1e25,-1e25,1e-35,-1e-35],device='cuda',dtype=torch.float32)
a=scale.double()*256;meta=torch.cat((scale,(1e-8/(a*a)).float(),a.sign().float()));out=torch.empty_like(scale);maxerr=0.
for value in (0.,1e-30,.5,1.,1e30):
 sums=torch.full_like(scale,value);mod.run(meta.data_ptr(),sums.data_ptr(),out.data_ptr(),torch.cuda.current_stream().cuda_stream);torch.cuda.synchronize()
 want=torch.where((a==0)|(sums==0),0.,a.sign()*torch.rsqrt(sums.double()/4096+1e-8/(a*a))).float()
 assert bool(torch.isfinite(out).all()),(value,out)
 err=float(((out-want).abs()/want.abs().clamp_min(1e-38)).max());assert err<2e-6,(value,err,out,want);maxerr=max(maxerr,err)
# Execute the exact shipped packet_tile/packet_scale helper on actual raw-fallback serialization.
ps=(D/'pack5_cell.py').read_text();f='<packet extreme cell>';linecache.cache[f]=(len(ps),None,ps.splitlines(True),f);q={'torch':torch};exec(compile(ps,f,'exec'),q)
helper=(D/'pack5_cuda.py.inc').read_text();source2='#include <cuda_runtime.h>\n#include <cstdint>\n'+helper+'''
__global__ void decode(const uint8_t* bank,uint8_t* output,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n*4096){int bytes;const uint8_t* t=packet_tile(bank,i/4096,bytes);output[i]=packet_scale(t,i%4096);}}
void run(int64_t b,int64_t o,int64_t n,int64_t st){decode<<<(n*4096+255)/256,256,0,(cudaStream_t)st>>>((uint8_t*)b,(uint8_t*)o,n);}
'''
mod2=load_inline(name='p3a_packet_ext_'+hashlib.sha256(source2.encode()).hexdigest()[:12],cpp_sources='#include <cstdint>\n#include <pybind11/pybind11.h>\nvoid run(int64_t,int64_t,int64_t,int64_t);',cuda_sources=source2,functions=['run'],extra_cuda_cflags=['-O3','-arch=sm_'+arch],no_implicit_headers=True,with_pytorch_error_handling=False)
raw=torch.randint(0,256,(17,36864),device='cuda',dtype=torch.uint8);raw[:8,32768:]=100+raw[:8,32768:]%32
bank=q['pack5_bank'](raw);got=torch.empty((17,4096),device='cuda',dtype=torch.uint8);mod2.run(bank.data_ptr(),got.data_ptr(),17,torch.cuda.current_stream().cuda_stream);torch.cuda.synchronize();assert torch.equal(got,raw[:,32768:])
result=dict(cancel_helper_sha256=cancel_sha,cancel_cases=160,cancel_worst_relative=maxerr,packet_mixed_raw_compressed_tiles=17,packet_bytes_exact=True)
(D/'extreme_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True)
