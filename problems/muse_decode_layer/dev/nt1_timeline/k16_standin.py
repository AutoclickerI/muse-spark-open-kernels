"""Test-only sm120 stand-in: run K16 math with one raw stage; the shipped H200 cell uses two."""
def install(namespace):
 if namespace['torch'].cuda.get_device_capability()==(9,0):raise RuntimeError('stand-in is local-only')
 native=namespace['pk16_module']();calls=[]
 class Adapter:
  smem_optin=native.smem_optin
  def __getattr__(self,k):return getattr(native,k)
  def smem_needed(self,nt,stages,m):
   assert stages==2;return native.smem_needed(nt,1,m)
  def epi_info(self,flags,stages,m):
   assert stages==2;return native.epi_info(flags,1,m)
  def run(self,ptrs,m,start,stages,grid,stream,flags):
   assert stages==2 and len(ptrs)==(14 if flags&256 else 13);calls.append((m,flags));return native.run(ptrs,m,start,1,grid,stream,flags)
 a=Adapter();namespace['pk16_module']=lambda:a
 namespace['pk16_experts'].__globals__['pk16_module']=namespace['pk16_module']
 return calls
