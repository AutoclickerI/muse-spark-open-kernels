"""Offline sm90 Gluon compilation of the actual selected dispatch, no Hopper execution."""
import hashlib,importlib.util,json
from pathlib import Path
import torch,triton
from triton.backends.compiler import GPUTarget
from triton.experimental.gluon._runtime import GluonASTSource
H=Path(__file__).resolve().parent;o=H/'out';sp=importlib.util.spec_from_file_location('op_compile',o/'operator.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
DT={torch.float32:'fp32',torch.bfloat16:'bf16',torch.float16:'fp16',torch.float8_e4m3fn:'fp8e4nv',torch.int32:'i32',torch.int64:'i64'};rows=[]
class Compile:
 def __init__(self,fn):self.fn=fn
 def __getitem__(self,grid):
  def run(*args,**kw):
   fn=self.fn;values=dict(zip(fn.arg_names,args));values.update({k:v for k,v in kw.items() if k in fn.arg_names});sig={};c={};at={}
   for i,n in enumerate(fn.arg_names):
    v=values[n]
    if i in fn.constexprs:c[n]=v
    else:
     sig[n]='*'+DT[v.dtype] if isinstance(v,torch.Tensor) else 'fp32' if isinstance(v,float) else 'i32'
     if isinstance(v,torch.Tensor) or isinstance(v,int) and v%16==0:at[i,]=[['tt.divisibility',16]]
   opts={k:v for k,v in kw.items() if k not in fn.arg_names};src=GluonASTSource(fn,sig,c,at);k=triton.compile(src,target=GPUTarget('cuda',90,32),options=opts)
   stem=fn.__name__+str(len(rows));(o/(stem+'.ptx')).write_text(k.asm['ptx']);(o/(stem+'.cubin')).write_bytes(k.asm['cubin'])
   row=dict(kernel=fn.__name__,grid=grid,constants=c,options=opts,registers=getattr(k.metadata,"n_regs",None),spills=getattr(k.metadata,"n_spills",None),shared=k.metadata.shared,sha256=hashlib.sha256(k.asm['cubin']).hexdigest(),wgmma=k.asm['ptx'].count('wgmma.mma_async'))
   rows.append(row);print(row,flush=True)
  return run
m._g_full_decode=Compile(m._g_full_decode);m._g_full_verify=Compile(m._g_full_verify);m._HOPPER_FORCE=True
for B,L in ((16,1),(32,1),(16,8),(32,8)):
 c=dict(batch=B,q_len=L,num_q_heads=16,num_kv_heads=2,head_dim=64,page_size=1,window=0,rope_theta=0.,sm_scale=.125,qk_scale_factor=5.473,kv_scale=1.)
 f=lambda *shape,dtype=torch.float32:torch.empty(shape,device='cuda',dtype=dtype)
 d=dict(config=c,qkv=f(B,L,2304,dtype=torch.bfloat16),cache_loc=f(B*L,dtype=torch.int64),seq_lens=f(B,dtype=torch.int32),k_cache=f(4,1,2,64,dtype=torch.float8_e4m3fn),v_cache=f(4,1,2,64,dtype=torch.float8_e4m3fn),page_table=f(B,80008,dtype=torch.int32),k_scale=f(),v_scale=f())
 m.attention(d['qkv'],d)
(o/'operator_native.json').write_text(json.dumps(rows,indent=2)+'\n')
