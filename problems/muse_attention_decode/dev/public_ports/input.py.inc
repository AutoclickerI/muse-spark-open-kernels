# S19 producer ABI: FP32 split planes or BF16;1280 qkv-only/2304 qkv+gate.
# A +inf logit supplies exactly sigmoid=1 when S19 applies its gate afterwards.
import torch
import triton
import triton.language as tl
@triton.jit
def _pack_kernel(P,O,ss:tl.constexpr,rs:tl.constexpr,S:tl.constexpr,W:tl.constexpr,T:tl.constexpr,OW:tl.constexpr,BLOCK:tl.constexpr):
 x=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK);t=x//OW;c=x%OW;ok=(t<T)&(c<W)
 v=tl.full((BLOCK,),0.,tl.float32)
 for s in tl.static_range(S):v+=tl.load(P+s*ss+t*rs+c,ok,other=0).to(tl.float32)
 v=tl.where(c<W,v,float('inf'))
 tl.store(O+x,v.to(tl.bfloat16),x<T*OW)
def raw(qkv,data):
 p,s=qkv;c=data['config'];t=c['batch']*c['q_len'];w=p.shape[-1]
 full=(2*c['num_q_heads']+2*c['num_kv_heads'])*c['head_dim']
 assert w in ((c['num_q_heads']+2*c['num_kv_heads'])*c['head_dim'],full) and p.stride(-1)==1
 if w==full and p.dtype==torch.bfloat16 and s==1:return p[0].reshape(c['batch'],c['q_len'],full)
 o=torch.empty((t,full),device=p.device,dtype=torch.bfloat16)
 _pack_kernel[(triton.cdiv(t*full,512),)](p,o,p.stride(0),p.stride(-2),s,w,t,full,512)
 return o.reshape(c['batch'],c['q_len'],full)
