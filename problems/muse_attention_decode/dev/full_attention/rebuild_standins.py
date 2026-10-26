"""Local N128 arithmetic/launch stand-in carrying each structural VQ8 edit."""
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3]

def install(m):
 import sys
 sys.path.insert(0,str(R/'problems/muse_decode_layer/dev/s15'))
 import local_standins
 result=local_standins.install(m)
 if m.torch.cuda.get_device_capability()==(9,0):return result
 # Preserve VDSK's M/N/K tile and split reduction instead of rejecting the
 # Hopper8-stage172KiB plan and silently taking the generic local projection.
 # Only the prefetch depth changes to fit GPU0's99KiB shared capacity.
 kernel=m._vdsk_gemm
 if not getattr(kernel,'_p5_depth_cap',False):
  class DepthCap:
   _p5_depth_cap=True
   def __getattr__(self,name):return getattr(kernel,name)
   def cap(self,kw):
    kw=dict(kw)
    if kw.get('NS',0)>3:kw['NS']=3;kw['num_stages']=3
    return kw
   def warmup(self,*args,**kw):return kernel.warmup(*args,**self.cap(kw))
   def __getitem__(self,grid):return lambda *args,**kw:kernel[grid](*args,**self.cap(kw))
  m._vdsk_gemm=DepthCap()
 result['vdsk']='same tiles/splits/BF16 dot; cap pipeline depth3 to fit sm120 shared memory'
 import local_n128
 result.update(local_n128.install(m))
 result.pop('regmax',None)
 print('P5_LOCAL_STANDINS',result,flush=True)
 return result
