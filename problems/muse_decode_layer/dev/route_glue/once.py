"""Build the per-row cooperative routing producer without duplicating RMS work."""
from pathlib import Path
HERE=Path(__file__).resolve().parent
s=(HERE/'route_moe.cu').read_text()
s=s.replace('#include <cuda_runtime.h>','#include <cuda_runtime.h>\n#include <cooperative_groups.h>',1)
s=s.replace('MoERouteMeta* meta;','MoERouteMeta* meta;float* global_inv;')
row=(HERE/'prologue.cuh').read_text();row=row[row.index('__device__ __noinline__ void rg_prepare'):row.index('\n__device__ __forceinline__ __nv_bfloat16 rg_value')]
row=row.replace('rg_prepare(', 'rg_prepare_row(').replace('if(blockIdx.x==0){','{')
pos=s.index('// MoERouteMeta v1');s=s[:pos]+row+'\n'+s[pos:]
old='''  rg_prepare(rg,M,private_ids,private_tw,private_inv);
  if(blockIdx.x==0 && M<=32)rg_emit_meta(rg.meta,M,*rg.local_start,private_ids,reinterpret_cast<int*>(rt));
  __syncthreads();'''
new='''  // One CTA owns each token's route and pre-expert RMS. A cooperative
  // grid barrier publishes it before all expert workers copy the summary.
  for(int tok=blockIdx.x;tok<M;tok+=gridDim.x){
    RouteGlue row=rg;row.pr+=int64_t(tok)*rg.rs;
    row.out_ids+=tok*8;row.out_lids+=tok*8;row.out_wts+=tok*8;
    rg_prepare_row(row,1,private_ids,private_tw,private_inv);
    if(threadIdx.x==0)rg.global_inv[tok]=private_inv[0];
    __syncthreads();
  }
  cooperative_groups::this_grid().sync();
  for(int i=threadIdx.x;i<M*8;i+=NTHR){private_ids[i]=rg.out_lids[i];private_tw[i]=rg.out_wts[i];}
  for(int i=threadIdx.x;i<M;i+=NTHR)private_inv[i]=rg.global_inv[i];
  __syncthreads();
  if(blockIdx.x==0 && M<=32)rg_emit_meta(rg.meta,M,*rg.local_start,private_ids,reinterpret_cast<int*>(rt));
  __syncthreads();'''
assert old in s;s=s.replace(old,new)
s=s.replace('(MoERouteMeta*)p[21]};','(MoERouteMeta*)p[21],(float*)p[0]};')
a=s.index(' moe_fused<1><<<grid,NTHR',s.index('extern "C" int rg_run'))
s=s[:a]+''' const __nv_bfloat16* x=(const __nv_bfloat16*)p[0];const int* ids=(const int*)p[1];const float* tw=(const float*)p[2];
 const __nv_bfloat16* pnw=(const __nv_bfloat16*)p[3];const uint8_t* w13=(const uint8_t*)p[4];const uint8_t* w2=(const uint8_t*)p[5];
 const float* gs13=(const float*)p[6];const float* gs2=(const float*)p[7];__nv_bfloat16* inter=(__nv_bfloat16*)p[8];
 float* y=(float*)p[9];float* ss=(float*)p[10];int* cnt=(int*)p[11];__nv_bfloat16* out=(__nv_bfloat16*)p[12];int zero=0;
 void* args[]={&x,&ids,&tw,&pnw,&w13,&w2,&gs13,&gs2,&inter,&y,&ss,&cnt,&out,&M,&zero,&stages,&rg};
 return cudaLaunchCooperativeKernel((const void*)moe_fused<1>,dim3(grid),dim3(NTHR),args,smem_bytes(1,stages,M),(cudaStream_t)stream);
}
'''
(HERE/'route_once.cu').write_text(s)
