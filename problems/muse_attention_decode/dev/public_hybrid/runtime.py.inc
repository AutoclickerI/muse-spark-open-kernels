# Hybrid direct-partial attention. License notices apply to embedded public code.
import linecache as _hy_linecache,types as _hy_types
_HY_MODULES={}
def _hy_module(name):
 if name not in _HY_MODULES:
  s=_HY_SOURCES[name];f='<p5b-hybrid-'+name+'>';_hy_linecache.cache[f]=(len(s),None,s.splitlines(True),f)
  m=_hy_types.ModuleType('p5b_hybrid_'+name);m.__file__=f;exec(compile(s,f,'exec'),m.__dict__);_HY_MODULES[name]=m
 return _HY_MODULES[name]
_hy_original_attention=_best7_attention
def _best7_attention(qkv,data,state,plan):
 c=data['config'];arm=_HY_SELECTION.get((c['batch'],c['q_len'],bool(c['window'])))
 valid=c['num_q_heads']==16 and c['num_kv_heads']==2 and c['head_dim']==64 and c['page_size']==1
 if arm is None or not valid:return _hy_original_attention(qkv,data,state,plan)
 if c['window'] and (arm=='O' or c['batch']==1 and arm!='J'):y=_hy_module('operator_direct').attention_pair(qkv,data)
 else:y=_hy_module('lee_hybrid').attention_pair(qkv,data,arm)
 return y.reshape(c['batch']*c['q_len'],1024)
def _hy_build():
 for name in sorted(_HY_SOURCES):_hy_module(name).setup(torch.device('cuda'))
# S21/S22 builds extensions in its prepare pool before synthetic dispatch warmup.
_S19_BUILDS.append(_hy_build)
