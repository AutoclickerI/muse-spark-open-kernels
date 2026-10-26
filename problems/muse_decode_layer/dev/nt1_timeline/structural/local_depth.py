"""Test-only sm120 depth3/4 ABI stand-in; native math executes with depth2."""
def install(ns):
 calls=[]
 for name,fn in list(ns.items()):
  if name.startswith('rvc_struct_') and name.endswith('_module') and name.rsplit('_',1)[0].endswith(('fused0','fused1','split0','split1')):
   def get(_fn=fn,_name=name):
    mod=_fn()
    class Proxy:
     def __getattr__(self,k):return getattr(mod,k)
     def smem_needed(self,nt,st,m):return mod.smem_needed(nt,min(st,2),m)
     def epi_info(self,f,st,m):return mod.epi_info(f,min(st,2),m)
     def structural_info(self,mode,st,m):return mod.structural_info(mode,min(st,2),m)
     def run(self,v,m,start,st,g,stream,f):
      if st>2:calls.append(dict(module=_name,requested_depth=st,executed_depth=2,M=m,kind='fused'))
      return mod.run(v,m,start,min(st,2),g,stream,f)
     def structural_run(self,v,m,start,st,g,stream,f):
      if st>2:calls.append(dict(module=_name,requested_depth=st,executed_depth=2,M=m,kind='split'))
      return mod.structural_run(v,m,start,min(st,2),g,stream,f)
    return Proxy()
   ns[name]=get
 return calls
