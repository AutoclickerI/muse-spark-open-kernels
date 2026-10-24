"""Execute the route cell's exact producer and compare with the shared ABI writer."""
from pathlib import Path
import ctypes,hashlib,json,sys
import torch
from torch.utils.cpp_extension import load_inline
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'tools'));import local_eval
m=local_eval.load_module('route_meta_reference',ROOT/'problems/muse_decode_layer/cand_r3_best7_scale.py')
header=(HERE.parent/'nt1_branches/route_meta.h').read_text().replace('#pragma once','')
golden=(HERE.parent/'nt1_branches/route_meta.cu.inc').read_text()
source='#include <cuda_runtime.h>\n#include <cuda_bf16.h>\n#include <math_constants.h>\n'+header+'\n'+(HERE/'prologue.cuh').read_text()+'\n'+(HERE/'emit_meta.cuh').read_text()+'\n'+golden+r'''
__global__ void fused_probe(RouteGlue rg,int M,__nv_bfloat16* x){
 extern __shared__ int scratch[];
 int* ids=scratch;float* weights=(float*)(ids+8*M);float* inv=weights+8*M;
 int* meta_scratch=(int*)(inv+M);
 rg_prepare(rg,M,ids,weights,inv);
 rg_emit_meta(rg.meta,M,*rg.local_start,ids,meta_scratch);
 for(int i=threadIdx.x;i<M*4096;i+=256)x[i]=rg_value(rg,i/4096,i%4096,inv[i/4096]);
}
extern "C" int probe(const void** p,int M,int S,void* stream){
 RouteGlue rg={(const float*)p[0],(const float*)p[1],(const float*)p[2],S,(int64_t)M*4608,4608,(int*)p[3],(int*)p[4],(float*)p[5],(int*)p[6],(MoERouteMeta*)p[7]};
 fused_probe<<<1,256,4*(8*M*2+M+320),(cudaStream_t)stream>>>(rg,M,(__nv_bfloat16*)p[8]);return (int)cudaGetLastError();
}
extern "C" int compare(const void* ids,int M,int start,void* out,void* stream){
 make_route_meta<<<1,256,0,(cudaStream_t)stream>>>((const int*)ids,M,start,(MoERouteMeta*)out);return (int)cudaGetLastError();
}
'''
path=load_inline(name='route_meta_probe_'+hashlib.sha256(source.encode()).hexdigest()[:16],cpp_sources='',cuda_sources=source,functions=None,
    is_python_module=False,no_implicit_headers=True,extra_cuda_cflags=['-O3','-std=c++17','-lineinfo'],verbose=False)
lib=ctypes.CDLL(path);lib.probe.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_void_p]
lib.compare.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p]
g=torch.Generator(device='cuda').manual_seed(72815);rows=[];worst=0.
for M in (1,2,7,8,16,31,32):
 for S in (1,2,8,16):
  p=torch.randn((S,M,4608),device='cuda',generator=g)
  bias=torch.randn(256,device='cuda',generator=g)*.055;pre=1+torch.randn(4096,device='cuda',generator=g)*.1
  ids=torch.empty((M,8),device='cuda',dtype=torch.int32);lids=torch.empty_like(ids);wts=torch.empty((M,8),device='cuda',dtype=torch.float32)
  local=torch.tensor([0],device='cuda',dtype=torch.int32);out=torch.empty(736,device='cuda',dtype=torch.int32)
  x=torch.empty((M,4096),device='cuda',dtype=torch.bfloat16)
  tensors=[p,bias,pre,ids,lids,wts,local,out,x];ptrs=(ctypes.c_void_p*9)(*(v.data_ptr() for v in tensors))
  def fn():
   e=lib.probe(ptrs,M,S,torch.cuda.current_stream().cuda_stream)
   if e:raise RuntimeError(e)
  fn();graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph):fn()
  for draw,start in enumerate((0,96,224)):
   p.normal_(generator=g)
   if draw==2:p.zero_()
   local.fill_(start);out.fill_(-123456);graph.replay()
   ref=m.route_and_norm((p,S),dict(router_bias=bias,pre_expert_w=pre),dict(local_start=local))
   assert torch.equal(ids,ref[1]) and torch.equal(lids,ref[3]),(M,S,draw,'route')
   want=torch.zeros_like(out)
   e=lib.compare(ids.data_ptr(),M,start,want.data_ptr(),torch.cuda.current_stream().cuda_stream)
   if e:raise RuntimeError(e)
   # Compare every defined field; both buffers have zeroed tail padding.
   assert torch.equal(out,want),(M,S,draw,'ABI mismatch',torch.where(out!=want)[0].tolist())
   error=float((x.float()-ref[0].float()).norm()/ref[0].float().norm().clamp_min(1e-12));worst=max(worst,error);assert error<.001
   # All-remote and zero inputs must rewrite the previous nonempty descriptor.
   bias.fill_(-1.);bias[(start+64)%256:(start+64)%256+8]=10.
  rows.append(dict(tokens=M,splits=S,draws=3,route_flips=0))
result=dict(checks=len(rows)*3,route_flips=0,abi_bytes=2944,version=1,max_norm_l2=worst,rows=rows,
   source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (HERE.parent/'nt1_branches/route_meta.h',HERE.parent/'nt1_branches/route_meta.cu.inc',HERE/'prologue.cuh',HERE/'emit_meta.cuh')})
(HERE/'metadata_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('rows','source_hashes')}))
