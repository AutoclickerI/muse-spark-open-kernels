# Owned ABI/dispatch. Embedded public kernels retain complete MIT/Apache notices.
import linecache as _pv_lc, types as _pv_types
_PV_MODULES={}
def _pv_module(name):
 if name not in _PV_MODULES:
  s=_PV_SOURCES[name];f='<p5b-vector-'+name+'>';_pv_lc.cache[f]=(len(s),None,s.splitlines(True),f)
  m=_pv_types.ModuleType('p5b_vector_'+name);m.__file__=f;exec(compile(s,f,'exec'),m.__dict__);_PV_MODULES[name]=m
 return _PV_MODULES[name]
_pv_original_attention=_best7_attention
def _best7_attention(qkv,data,state,plan):
 c=data['config'];key=c['batch'],c['q_len'],bool(c['window']);arm=_PV_SELECTION.get(key)
 valid=c['num_q_heads']==16 and c['num_kv_heads']==2 and c['head_dim']==64 and c['page_size']==1
 if not arm or not valid:return _pv_original_attention(qkv,data,state,plan)
 if key in ((1,1,True),(1,1,False)):
  # Native modules were replaced in this namespace; native dispatch is unchanged.
  return _pv_original_attention(qkv,data,state,plan)
 op=key==(1,8,True)
 pair=qkv
 # D isolates PDL at the existing packing boundary. The accepted two direct
 # cells already omit packing and must stay direct in this control arm.
 if arm=='D' and key not in ((32,1,True),(1,8,True)):
  x=_pv_module('pack').raw(qkv,data);pair=x.reshape(1,c['batch']*c['q_len'],2304),1
 y=_pv_module(('op_' if op else 'lee_')+arm).attention_pair(pair,data)
 return y.reshape(c['batch']*c['q_len'],1024)
def _pv_build():
 for name in sorted(_PV_SOURCES):
  m=_pv_module(name)
  if hasattr(m,'setup'):m.setup(torch.device('cuda'))
_S19_BUILDS.append(_pv_build)
