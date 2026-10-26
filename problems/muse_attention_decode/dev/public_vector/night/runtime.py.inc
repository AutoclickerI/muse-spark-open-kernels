# Owned ABI and PDL plumbing; the embedded public source retains its notices.
import linecache as _night_lc,types as _night_types
_NIGHT_MODULES={}
def _night_module(name):
 if name not in _NIGHT_MODULES:
  s=_NIGHT_SOURCES[name];f='<p5b-night-'+name+'>';_night_lc.cache[f]=(len(s),None,s.splitlines(True),f)
  m=_night_types.ModuleType('p5b_night_'+name);m.__file__=f;exec(compile(s,f,'exec'),m.__dict__);_NIGHT_MODULES[name]=m
 return _NIGHT_MODULES[name]
_night_original_attention=_best7_attention
def _best7_attention(qkv,data,state,plan):
 c=data['config'];key=c['batch'],c['q_len'],bool(c['window']);arm=_NIGHT_SELECTION.get(key)
 valid=c['num_q_heads']==16 and c['num_kv_heads']==2 and c['head_dim']==64 and c['page_size']==1
 if not arm or not valid or key in ((1,1,True),(1,1,False)):return _night_original_attention(qkv,data,state,plan)
 if arm in ('STAGE','LANE1','SOFTMAX'):y=_night_module('lee_stage').attention_pair(qkv,data,arm)
 else:
  pair=qkv
  if arm=='D' and key not in ((32,1,True),(1,8,True)):
   x=_night_module('pack').raw(qkv,data);pair=x.reshape(1,c['batch']*c['q_len'],2304),1
  y=_night_module(('op_' if key==(1,8,True) else 'lee_')+arm).attention_pair(pair,data)
 return y.reshape(c['batch']*c['q_len'],1024)
def _night_setup():
 for name in sorted(_NIGHT_SOURCES):
  m=_night_module(name)
  if hasattr(m,'setup'):m.setup(torch.device('cuda'))
_S19_BUILDS.append(_night_setup)
_night_previous_delta=_s16_prepare_delta
def _s16_prepare_delta(weights,state):
 state=_night_previous_delta(weights,state);_night_setup();return state
