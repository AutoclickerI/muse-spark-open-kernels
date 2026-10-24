"""Patch only the NT1 input-routing prologue; retain its packed expert body."""
from pathlib import Path
import ast
import hashlib
import json
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
a=ast.parse((ROOT/'problems/muse_decode_layer/cand_r3_best7_scale.py').read_text())
values={}
for node in a.body:
    if isinstance(node,ast.Assign):
        for target in node.targets:
            if isinstance(target,ast.Name) and target.id in ('V9_CUDA_SRC','V9_CPP_SRC'):
                values[target.id]=ast.literal_eval(node.value)
src=values['V9_CUDA_SRC'];start=src.index('__device__ __forceinline__ uint32_t scale_e4m3');end=src.index('\n__device__ __forceinline__ void mma16816',start)
src=src[:start]+'''__device__ __forceinline__ uint32_t scale_e4m3(uint32_t code){
 const uint32_t b=0x7000u+(code<<4);return b|(b<<16);}
'''+src[end:]
src=src.replace('#include <stdint.h>','#include <stdint.h>\n#include <math_constants.h>')
pos=src.index('template <int NT>\n__global__')
header=(HERE.parent/'nt1_branches/route_meta.h').read_text().replace('#pragma once','')
src=src[:pos]+header+'\n'+(HERE/'prologue.cuh').read_text()+'\n'+(HERE/'emit_meta.cuh').read_text()+'\n'+src[pos:]
src=src.replace('int M, int local_start, int nstage) {','int M, int local_start, int nstage, RouteGlue rg) {')
needle='  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;'
pos=src.index(needle,src.index('__global__ void __launch_bounds__'))
src=src[:pos]+'''  int* private_ids=reinterpret_cast<int*>(pair_of+npairs);
  float* private_tw=reinterpret_cast<float*>(private_ids+npairs);
  float* private_inv=private_tw+npairs;
  rg_prepare(rg,M,private_ids,private_tw,private_inv);
  if(blockIdx.x==0 && M<=32)rg_emit_meta(rg.meta,M,*rg.local_start,private_ids,reinterpret_cast<int*>(rt));
  __syncthreads();
  ids=private_ids;tw=private_tw;
'''+src[pos:]
src=src.replace('__ldg(ids + i)','ids[i]')
# rg private IDs are already local (-1 for nonlocal), so the body's local_start is zero.
# Every original expert scalar/MMA operation is retained; only BF16 input loads change.
old='''              else { v0 = __ldg(p); v1 = __ldg(p + 1); }'''
new='''              else {
                const int token=pair_of[off+idx]>>3;
                __nv_bfloat16 vv[16];
                #pragma unroll
                for(int z=0;z<16;++z)vv[z]=rg_value(rg,token,warp*512+g*64+q*16+z,private_inv[token]);
                v0=*reinterpret_cast<uint4*>(vv);v1=*reinterpret_cast<uint4*>(vv+8);
              }'''
assert old in src
src=src.replace(old,new)
src=src.replace('4 * npairs;','4 * npairs + 8*npairs + 4*M;')
# smem layout: original pairs, private int ids, FP32 weights and FP32 row inverse norms.
old='''         int64_t grid) {'''
new='''         int64_t grid, torch::Tensor pr, torch::Tensor bias, torch::Tensor pre,
         torch::Tensor out_ids, torch::Tensor out_lids, torch::Tensor out_wts, torch::Tensor lstart, int64_t splits) {'''
assert old in src;src=src.replace(old,new)
needle='  const int M = static_cast<int>(x.size(0));'
src=src.replace(needle,needle+'''
  RouteGlue rg={pr.data_ptr<float>(),bias.data_ptr<float>(),pre.data_ptr<float>(),(int)splits,
                pr.stride(0),pr.stride(1),out_ids.data_ptr<int>(),out_lids.data_ptr<int>(),out_wts.data_ptr<float>(),lstart.data_ptr<int>(),nullptr};''')
src=src.replace('static_cast<int>(nstage))','static_cast<int>(nstage), rg)')
cpp=values['V9_CPP_SRC'].replace('int64_t grid);','''int64_t grid, torch::Tensor pr, torch::Tensor bias, torch::Tensor pre,
         torch::Tensor out_ids, torch::Tensor out_lids, torch::Tensor out_wts, torch::Tensor lstart, int64_t splits);''')
# A raw-pointer ABI avoids PyTorch headers in the CUDA translation unit.
src=src[:src.index('int64_t max_smem_attr()')]
src=src.replace('#include <torch/extension.h>', '#include <cuda_runtime.h>').replace('#include <ATen/cuda/CUDAContext.h>', '')
src += r'''extern "C" int rg_max_smem(){int dev=0,n=0;cudaGetDevice(&dev);cudaDeviceGetAttribute(&n,cudaDevAttrMaxSharedMemoryPerBlockOptin,dev);cudaFuncAttributes attr={};auto e=cudaFuncGetAttributes(&attr,moe_fused<1>);if(e!=cudaSuccess)return -(int)e;n-=attr.sharedSizeBytes;e=cudaFuncSetAttribute(moe_fused<1>,cudaFuncAttributeMaxDynamicSharedMemorySize,n);return e==cudaSuccess?n:-(int)e;}
extern "C" int rg_smem(int stages,int m){return smem_bytes(1,stages,m);}
extern "C" int rg_resident(int stages,int m){int n=0;cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,moe_fused<1>,NTHR,smem_bytes(1,stages,m));return n;}
extern "C" int rg_run(const void** p,int M,int S,int stages,int grid,int64_t ps,int64_t rs,void* stream){
 RouteGlue rg={(const float*)p[14],(const float*)p[15],(const float*)p[16],S,ps,rs,(int*)p[17],(int*)p[18],(float*)p[19],(int*)p[20],(MoERouteMeta*)p[21]};
 moe_fused<1><<<grid,NTHR,smem_bytes(1,stages,M),(cudaStream_t)stream>>>((const __nv_bfloat16*)p[0],(const int*)p[1],(const float*)p[2],(const __nv_bfloat16*)p[3],
 (const uint8_t*)p[4],(const uint8_t*)p[5],(const float*)p[6],(const float*)p[7],(__nv_bfloat16*)p[8],(float*)p[9],(float*)p[10],(int*)p[11],(__nv_bfloat16*)p[12],M,0,stages,rg);
 return (int)cudaGetLastError();
}
'''
(HERE/'route_moe.cu').write_text(src)
(HERE/'route_moe.cpp').write_text('')
print('native',hashlib.sha256(src.encode()).hexdigest())
