CUDA=_embedded("_PUBLIC_SOURCES/lee/CUDA.cu")
# Derived from Lee v56 dispatch, Apache-2.0; changed to the owned S19 attention ABI.
import hashlib
import torch
from torch.utils.cpp_extension import load_inline
_MOD=None;_BUF={};_INV={};_EMPTY=None
_LOCAL=False
VCFG={(1,True):(-1,128,8,8),(1,False):(-2,304,128,68),(16,True):(-2,128,8,32836),(32,True):(-2,256,8,32836),(16,False):(-2,256,8,324),(32,False):(-2,256,8,324)}
DECL='void lee_attention(torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,torch::Tensor,int64_t,int64_t,int64_t,int64_t,double,double,double,double,int64_t);'
def setup(device):
 global _MOD,_EMPTY
 if _MOD is None:
  flags=['-O3','-lineinfo','-DRSW=4','-DPRS=4','-DDRS=4','-DRBF=5','-gencode=arch=compute_90a,code=sm_90a']
  if _LOCAL:flags=['-O3','-lineinfo','-DRSW=2','-DPRS=2','-DDRS=2','-DRBF=2','-DWGEMU=1','-DCLUS=0']
  _MOD=load_inline(name='p5b_lee_'+hashlib.sha256(CUDA.encode()).hexdigest()[:12]+('_local' if _LOCAL else '_90a'),cpp_sources=DECL,cuda_sources=CUDA,functions=['lee_attention'],extra_cuda_cflags=flags)
  _EMPTY=torch.empty(0,device=device)
 for theta in (0.,500000.):
  key=str(device),theta
  if key not in _INV:_INV[key]=(1./((theta or 1.)**(torch.arange(0,64,2,device=device).float()/64))).contiguous()
def attention(qkv,data):
 c=data['config'];B,L=c['batch'],c['q_len'];sl=bool(c['window']);dev=qkv.device;sms=torch.cuda.get_device_properties(dev).multi_processor_count
 setup(dev);span=min(data['page_table'].shape[1],c['window']+L) if sl else data['page_table'].shape[1]
 if L==1 and sl and B in (16,32):
  kind=1;nl=1 if _LOCAL else 2;ns=max(nl,min((span+63)//64,nl*(sms//B)));ns-=ns%nl;x=64;cap=2
 elif L==1:
  kind=0;ns=max(1,min(132//B,(span+127)//128));x=0;cap=16
  if ns<=17 and B*min(ns,16)<=64:ns=1<<(min(ns,16).bit_length()-1)
 else:
  var,kps,cap,x=VCFG[B,sl];kind=2 if var==-1 else 3
  ns=max(1,min((span+kps-1)//kps,264//B))
  if kind==3:
   nl=1 if _LOCAL else 2;ns=max(nl,min(ns+(ns&1) if nl==2 else ns,nl*(sms//B)))
   if x&256:ns=min(64,2*((264+B-1)//B))
 key=(kind,B,L,ns)
 if key not in _BUF:
  rows=16 if L==1 else 128
  _BUF[key]=(torch.empty(B*ns*rows*64,device=dev),torch.empty(B*ns*rows*2,device=dev),torch.zeros(2*B,device=dev,dtype=torch.int32))
 wo,wml,cnt=_BUF[key];out=torch.empty((B,L,1024),device=dev,dtype=torch.bfloat16)
 _MOD.lee_attention(qkv.reshape(B,L,2304),data['cache_loc'],data['k_cache'],data['v_cache'],data['page_table'],data['seq_lens'],wo,wml,cnt,out,_INV[str(dev),float(c['rope_theta'])],_EMPTY,kind,c['window'],ns,x,c['rope_theta'],c['qk_scale_factor'],c['sm_scale'],c['kv_scale'],cap)
 return out
