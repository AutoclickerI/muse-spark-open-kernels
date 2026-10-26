"""Cross-compile every reviewed dispatch for sm90 without executing donor code."""
import importlib.util,hashlib,json,sys,inspect
from pathlib import Path
import torch,triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.runtime.jit import mangle_type
H=Path(__file__).resolve().parent;spec=importlib.util.spec_from_file_location('public_native_compile',H/'out/source.py');m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);rows=[];cache={}
class Compiled:
 function=None
 def __init__(self,k):self.kernel=k;self.packed_metadata=(k.metadata.num_warps,1,k.metadata.shared)
 def _init_handles(self):pass
 def launch_metadata(self,*args):return None
 def run(self,*args):pass
class Cross:
 def __init__(self,fn):self.fn=fn
 def go(self,args,kw):
  fn=self.fn;bound=inspect.signature(fn.fn).bind_partial(*args,**{k:z for k,z in kw.items() if k in fn.arg_names});bound.apply_defaults();v=bound.arguments;sig={};const={};attrs={}
  for i,n in enumerate(fn.arg_names):
   x=v[n]
   if i in fn.constexprs:const[n]=x
   else:
    sig[n]=mangle_type(x)
    if isinstance(x,torch.Tensor) or isinstance(x,int) and x%16==0:attrs[(i,)]=[['tt.divisibility',16]]
  opts={k:x for k,x in kw.items() if k not in fn.arg_names and k!='grid'};key=str((fn.__name__,sig,const,opts))
  if key in cache:return cache[key]
  k=triton.compile(ASTSource(fn,sig,const,attrs),target=GPUTarget('cuda',90,32),options=opts);tag=hashlib.sha256(key.encode()).hexdigest()[:12];(H/'out'/('native_'+tag+'.ptx')).write_text(k.asm['ptx']);(H/'out'/('native_'+tag+'.cubin')).write_bytes(k.asm['cubin']);rows.append(dict(kernel=fn.__name__,constants=const,shared=k.metadata.shared,wgmma=k.asm['ptx'].count('wgmma.mma_async'),cubin_sha256=hashlib.sha256(k.asm['cubin']).hexdigest()));cache[key]=Compiled(k);return cache[key]
 def __getitem__(self,grid):return lambda *a,**kw:self.go(a,kw)
 def warmup(self,*a,**kw):return self.go(a,kw)
for name,value in list(vars(m).items()):
 if name.startswith(('_cbf_','_pp_coded')) and hasattr(value,'arg_names'):setattr(m,name,Cross(value))
roles={r:torch.empty((n,k),device='cuda',dtype=torch.bfloat16) for r,n,k in [('qkv_gate',2304,8192),('qkv3',1280,8192),('gate',1024,8192),('o',8192,1024),('pool_router',4608,8192),('unpooler',8192,4096),('unpooler_slice',1024,4096)]};bank=dict(roles=roles,cluster={},tiled={},coded={})
for r,w in roles.items():
 n,k=w.shape
 for bk in (128,256):bank['tiled'][r,bk]=torch.empty((n*k//bk,bk),device='cuda',dtype=torch.bfloat16)
 bank['coded'][r]=dict(words=torch.empty((n,k//8,3),device='cuda',dtype=torch.int32),base=torch.empty(n//16,device='cuda',dtype=torch.int32),offset=torch.empty(n*k//128+1,device='cuda',dtype=torch.int32),values=torch.empty(1,device='cuda',dtype=torch.int16))
for t in (1,8,16,32,128,256):
 for r in ('qkv_gate','qkv3','gate','o','pool_router','unpooler_slice' if t<=8 else 'unpooler'):
  x=torch.empty((t,roles[r].shape[1]),device='cuda',dtype=torch.bfloat16)
  for p in ('O','P','C'):
   print('COMPILE',t,r,p,flush=True);m.pp_project(x,r,bank,p)
(H/'native_validation.json').write_text(json.dumps(dict(passed=all(r['shared']<=227328 for r in rows),all_launch_shared_admitted=all(r['shared']<=227328 for r in rows),units=rows,limits='Actual sm90 compile only; no H200 math executed locally'),indent=2)+'\n');print('PASS',len(rows),'units')
