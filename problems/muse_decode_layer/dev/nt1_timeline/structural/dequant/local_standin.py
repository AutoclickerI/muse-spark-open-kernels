"""Only for sm120:32KiB lane table,explicit128-byte address math,one raw stage."""
def install(ns):
 calls=[]
 for name,src in list(ns.items()):
  if name in ('RVC_DQ0_CUDA','RVC_DQ1_CUDA'):
   before='const uint32_t ptr=__byte_perm(z,tableptr,0x7604u+((k&3)<<4))+(tableptr&0xff00u);'
   assert before in src
   src=src.replace('__align__(256) uint32_t dqtable[256*64]','__align__(256) uint32_t dqtable[256*32]').replace('i<256*64','i<256*32').replace('uint32_t u=i>>6','uint32_t u=i>>5')
   src=src.replace(before,'const uint32_t ix=__byte_perm(z,0u,0x4440u+(k&3));const uint32_t ptr=tableptr+ix*128u;')
   ns[name]=src
 for name,fn in list(ns.items()):
  if name in ('rvc_dq0_module','rvc_dq1_module'):
   def get(_fn=fn,_name=name):
    mod=_fn()
    class Proxy:
     def __getattr__(self,k):return getattr(mod,k)
     def smem_needed(self,nt,st,m):return mod.smem_needed(nt,1,m)
     def epi_info(self,f,st,m):return mod.epi_info(f,1,m)
     def run(self,v,m,start,st,g,stream,f):
      calls.append(dict(module=_name,requested_stages=st,executed_stages=1,table_bytes=32768,M=m))
      return mod.run(v,m,start,1,g,stream,f)
    return Proxy()
   ns[name]=get
 return calls
