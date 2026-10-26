"""Local-only stand-ins for FA3 main and one-rank DSM; never shipped."""
import hashlib,types
import torch
from torch.utils.cpp_extension import load_inline
import local_n128

def native(module,source):
 m=types.ModuleType('chain_local');m.__dict__.update(module.__dict__);m.VQ8_CUDA=source;m.VQ8_CPP=module.VQ8_CPP
 original=m.load_inline
 def load(*args,**kw):
  cu=kw.get('cuda_sources','')
  if isinstance(cu,str) and 'local_vt' in cu:
   cu=cu.replace('(int64_t)blockIdx.x * ST * 16384','((int64_t)blockIdx.y*gridDim.x+blockIdx.x) * ST * 16384')
   cu=cu.replace('cfg.numAttrs=2;','cfg.numAttrs=pdl?1:0;')
   kw['cuda_sources']=cu
  return original(*args,**kw)
 m.load_inline=load;local_n128.install(m);n=m._vq8_module();n._p5_cluster_control='cfg.gridDim = dim3(p.min_tiles,p.NS);' in source and '__syncthreads();cluster_reduce(p,sm);' not in source;return n

CONVERT=r'''
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
__global__ void conv(const float*op,const float*mp,const float*lp,const float*vs,float*oa,float*la,int B,int L,int S,int LB){
 int i=blockIdx.x*blockDim.x+threadIdx.x;int total=S*16*B*L*64;if(i>=total)return;
 int d=i%64,t=(i/64)%(B*L),h=(i/(64*B*L))%16,s=i/(64*B*L*16),b=t/L,j=t%L;
 int row=((b*(L/LB)+j/LB)*S+s)*(LB*16)+(j%LB)*16+h;
 float den=lp[row],v=op[(int64_t)row*128+(h/8)*64+d];oa[i]=den? v/den*(*vs):0.f;
 if(d==0)la[i/64]=den?mp[row]*0.6931471805599453f+logf(den):-INFINITY;
}
void fa_convert(int64_t op,int64_t mp,int64_t lp,int64_t vs,int64_t oa,int64_t la,int64_t B,int64_t L,int64_t S,int64_t LB,int64_t stream){
 int n=S*16*B*L*64;conv<<<(n+255)/256,256,0,(cudaStream_t)stream>>>((const float*)op,(const float*)mp,(const float*)lp,(const float*)vs,(float*)oa,(float*)la,B,L,S,LB);
}
'''
_cache=[]
def fa_main(base,q,data,S,oa,la):
 if not _cache:
  _cache.append(load_inline(name='p5_chain_convert_'+hashlib.sha256(CONVERT.encode()).hexdigest()[:12],cpp_sources='void fa_convert(int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);',cuda_sources=CONVERT,functions=['fa_convert'],extra_cuda_cflags=['-O3']))
 c=data['config'];B,L=c['batch'],c['q_len'];LB=min(L,4);span=(c['window']+LB-1) if c['window'] else data['page_table'].shape[1]*c['page_size']
 chunk=((span+S-1)//S+63)//64*64;rows=B*(L//LB)*S*(LB*16)
 op=torch.empty(rows*128,device='cuda');mp=torch.empty(rows,device='cuda');lp=torch.empty_like(mp)
 base._launch(base._attn_split_kernel,(S,B,L//LB),q.to(torch.bfloat16),data['k_cache'],data['v_cache'],data['page_table'],data['seq_lens'],data['k_scale'],op,mp,lp,data['page_table'].stride(0),c['sm_scale']*c['qk_scale_factor'],S,L=L,LB=LB,HQ=16,D=64,PAGE=c['page_size'],WINDOW=c['window'],SPLIT=chunk,BLOCK_N=64,num_warps=8 if LB*16>32 else 4,num_stages=2)
 _cache[0].fa_convert(op.data_ptr(),mp.data_ptr(),lp.data_ptr(),data['v_scale'].data_ptr(),oa.data_ptr(),la.data_ptr(),B,L,S,LB,torch.cuda.current_stream().cuda_stream)
