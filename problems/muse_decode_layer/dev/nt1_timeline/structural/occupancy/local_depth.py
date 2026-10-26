"""Test-only sm120 depth3/4 ABI stand-in; native math executes with depth2."""
def install(ns):
 calls=[]
 for name,fn in list(ns.items()):
  if name.startswith('rvc_occ_') and name.endswith(('0_module','1_module')):
   def get(_fn=fn,_name=name):
    mod=_fn()
    class Proxy:
     def __getattr__(self,k):return getattr(mod,k)
     def smem_needed(self,nt,st,m):return mod.smem_needed(nt,min(st,2),m)
     def epi_info(self,f,st,m):return mod.epi_info(f,min(st,2),m)
     def run(self,v,m,start,st,g,stream,f):
      if st>2:calls.append(dict(module=_name,requested_depth=st,executed_depth=2,M=m,kind='fused'))
      return mod.run(v,m,start,min(st,2),g,stream,f)
    return Proxy()
   ns[name]=get
 return calls
