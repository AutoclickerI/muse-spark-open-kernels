"""Explicit local glue:sm120 executes producer/consumer math but lacks setmaxnreg andthe full shared budget."""
def install(ns):
 import re
 calls=[]
 for name,src in list(ns.items()):
  if name.startswith('RVC_OV_') and name.endswith('_CUDA'):
   ns[name]=re.sub(r'asm volatile\("setmaxnreg\.(?:inc|dec)\.sync\.aligned\.u32 \d+;":::"memory"\);','asm volatile("":::"memory");',src)
 for name,fn in list(ns.items()):
  if name.startswith('rvc_ov_') and name.endswith(('0_module','1_module')) and any(k in name for k in ('cache','ws8')):
   def get(_fn=fn,_name=name):
    mod=_fn()
    class Proxy:
     def __getattr__(self,k):return getattr(mod,k)
     def smem_needed(self,nt,st,m):return mod.smem_needed(nt,1,m)
     def epi_info(self,f,st,m):return mod.epi_info(f,1,m)
     def run(self,v,m,start,st,g,stream,f):
      calls.append(dict(module=_name,requested_stages=st,executed_stages=1,M=m,register_reassignment='local nop'))
      return mod.run(v,m,start,1,g,stream,f)
    return Proxy()
   ns[name]=get
 return calls
