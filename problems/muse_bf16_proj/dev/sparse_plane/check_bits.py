"""Native reader bitwise round trip over enumerated/random NF codewords.
Out-of-window high bytes stay in the original escape list; this reader transforms
only low-byte/index/sign fields, and proves its emitted NF stream is identical.
"""
import runpy,torch,json
from pathlib import Path
H=Path(__file__).resolve().parent;p=runpy.run_path(str(H/'adapter.py'));rows=[]
for vector in (False,True):
 reader=p['SPARSE_CUDA']
 if vector:
  a=reader.index(' uint32_t cc[4]={};');b=reader.index(' out[2]=make_int4(cc[0],cc[1],cc[2],cc[3]);',a)+len(' out[2]=make_int4(cc[0],cc[1],cc[2],cc[3]);')
  reader=reader[:a]+p['SPARSE_VECTOR']+reader[b:]
 source='#include <cuda_runtime.h>\n#include <stdint.h>\n'+reader+'''
__global__ void sp_debug(const int4*w,int4*o,int n,int rot){int i=blockIdx.x,l=threadIdx.x;int4 v[3];sp_read(w,i,n,l,v,rot,false);for(int j=0;j<3;j++)o[(i*3+j)*32+l]=v[j];}
void run(int64_t w,int64_t o,int n,int rot){sp_debug<<<n,32>>>((int4*)w,(int4*)o,n,rot);}
'''
 m=p['load_inline'](name='sp_bits_'+p['hashlib'].sha256(source.encode()).hexdigest()[:12],cpp_sources='#include <pybind11/pybind11.h>\n#include <cstdint>\nvoid run(int64_t,int64_t,int,int);',cuda_sources=source,functions=['run'],extra_cuda_cflags=['-O3'],no_implicit_headers=True,with_pytorch_error_handling=False)
 for seed in (61791,14357,92003):
  g=torch.Generator(device='cuda').manual_seed(seed);n=128
  raw=torch.randint(0,256,(n,3,32,16),device='cuda',dtype=torch.uint8,generator=g)
  vals=torch.arange(65536,device='cuda',dtype=torch.int32).reshape(64,32,32)
  lo=(vals&255).to(torch.uint8);high=vals>>8
  raw[:64,:2]=lo.reshape(64,32,2,16).permute(0,2,1,3)
  nib=torch.arange(0,32,4,device='cuda');pos=torch.tensor((3,19,7,23,11,27,15,31),device='cuda')
  code=(((high&7).long().reshape(64,32,4,8)<<nib)|((high>>7).long().reshape(64,32,4,8)<<pos)).sum(-1)
  raw[:64,2]=code.to(torch.int32).contiguous().view(torch.uint8).reshape(64,32,16)
  wp=raw.flatten();plan=dict(wp=wp,cfg=(1,1,n,1),esc=torch.zeros((1,34),device='cuda',dtype=torch.int32),u01=0)
  packed,stats=p['sparse_pack'](plan,'v18');assert torch.equal(packed['esc'],plan['esc']);out=torch.empty_like(wp)
  graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph):m.run(packed['wp'].data_ptr(),out.data_ptr(),n,stats['rotation'])
  graph.replay();torch.cuda.synchronize();assert torch.equal(out,wp),int((out!=wp).sum())
  rows.append(dict(vector=vector,seed=seed,enumerated_inputs=65536,random_stream_bytes=wp.numel(),exact=True,stats=stats));print('PASS',vector,seed,flush=True)
(H/'bit_validation.json').write_text(json.dumps(dict(passed=True,rows=rows),indent=2)+'\n')
