"""Explicit CACHE1 transport stand-in for sm120's smaller per-CTA shared limit.

Only the private BF16 activation bank moves to preallocated global scratch. Raw TMA,
all math, route/readiness, launch glue and producer ordering still execute natively.
The H200 candidate/payload never contains this installation.
"""
import hashlib,json
from pathlib import Path
D=Path(__file__).resolve().parent
def cache_global(cu):
 old='  uint32_t* frag=reinterpret_cast<uint32_t*>((reinterpret_cast<uintptr_t>(pair_of+npairs)+127ull)&~127ull);'
 assert cu.count(old)==1
 out=cu.replace(old,'  uint32_t* frag=reinterpret_cast<uint32_t*>(trace)+(size_t)c*NT*16384;')
 old='4 * npairs + 128 + static_cast<size_t>(nt)*65536;';assert out.count(old)==1;out=out.replace(old,'4 * npairs;')
 old='ld.volatile.shared.v4.u32 {%0,%1,%2,%3},[%4];';assert out.count(old)==1;out=out.replace(old,'ld.volatile.global.v4.u32 {%0,%1,%2,%3},[%4];')
 old='"r"(smem_u32(p)):"memory");return v;}';assert out.count(old)==1;out=out.replace(old,'"l"(p):"memory");return v;}')
 return out

def install_cache(ns):
 records=[]
 for prefix in ('rvc_nfidx0','rvc_nbase1','rvc_nbase2',*[f'b22_cache1{i}' for i in range(3)]):
  key=prefix.upper()+'_CUDA'
  if key not in ns or 'bc_ld4' not in ns[key] or 'nstage=1;' not in ns[key]:continue
  before=ns[key];ns[key]=cache_global(before);original=ns[prefix+'_module'];cache={}
  def module(_original=original,_cache=cache):
   native=_original()
   if id(native) not in _cache:
    scratch=ns['torch'].empty(264*2*16384,device='cuda',dtype=ns['torch'].int32)
    class Wrapped:
     def __getattr__(self,name):return getattr(native,name)
     def run(self,v,*args):return native.run(list(v[:13])+[scratch.data_ptr()],*args)
    _cache[id(native)]=Wrapped()
   return _cache[id(native)]
  ns[prefix+'_module']=module
  records.append(dict(prefix=prefix,original_cuda_sha256=hashlib.sha256(before.encode()).hexdigest(),standin_cuda_sha256=hashlib.sha256(ns[key].encode()).hexdigest(),scope='Only CACHE1 BF16 cache storage and its LDS address space;all arithmetic/raw copies/ordering native'))
 return records
