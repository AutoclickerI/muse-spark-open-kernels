CUDA=_embedded("_HY_SOURCES/lee_hybrid/CUDA.cu")
# Derived from Lee v56 dispatch, Apache-2.0; changed to the owned S19 attention ABI.
import hashlib
import torch
from torch.utils.cpp_extension import load_inline
_MOD=None;_BUF={};_INV={};_EMPTY=None
_LOCAL=False
DECL='void lee_attention(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t,double,double,double,double,int64_t);'
def setup(device):
 global _MOD,_EMPTY
 if _MOD is None:
  flags=['-O3','-lineinfo','-DRSW=4','-DPRS=4','-DDRS=4','-DRBF=5','-gencode=arch=compute_90a,code=sm_90a']
  if _LOCAL:flags=['-O3','-lineinfo','-DRSW=2','-DPRS=2','-DDRS=2','-DRBF=2','-DWGEMU=1','-DCLUS=0']
  _MOD=load_inline(name='p5b_hybrid_'+hashlib.sha256(CUDA.encode()).hexdigest()[:12]+('_local' if _LOCAL else '_90a'),cpp_sources=DECL,cuda_sources=CUDA,functions=['lee_attention'],extra_cuda_cflags=flags)
  _EMPTY=torch.empty(0,device=device)
 for theta in (0.,500000.):
  key=str(device),theta
  if key not in _INV:_INV[key]=(1./((theta or 1.)**(torch.arange(0,64,2,device=device).float()/64))).contiguous()

def attention_pair(pair,data,arm='F'):
 p,s=pair;c=data['config'];B,L=c['batch'],c['q_len'];assert B==32 and L==1 and c['window'] and arm=='F'
 dev=p.device;setup(dev);nl=1 if _LOCAL else 2;sms=torch.cuda.get_device_properties(dev).multi_processor_count
 span=min(data['page_table'].shape[1],c['window']+L);ns=max(nl,min((span+63)//64,nl*(sms//B)));ns-=ns%nl
 key=(1,B,L,ns)
 if key not in _BUF:_BUF[key]=(torch.empty(B*ns*16*64,device=dev),torch.empty(B*ns*16*2,device=dev),torch.zeros(2*B,device=dev,dtype=torch.int32))
 wo,wml,cnt=_BUF[key];out=torch.empty((B,L,1024),device=dev,dtype=torch.bfloat16)
 _MOD.lee_attention(p,data['cache_loc'],data['k_cache'],data['v_cache'],data['page_table'],data['seq_lens'],wo,wml,cnt,out,_INV[str(dev),float(c['rope_theta'])],_EMPTY,1,c['window'],ns,64,c['rope_theta'],c['qk_scale_factor'],c['sm_scale'],c['kv_scale'],2)
 return out
