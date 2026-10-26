# Licensed public attention behind the unchanged S19 producer ABI. Only shape
# policy and packing are ours; all value-dependent work remains in this call.
import linecache as _public_linecache,types as _public_types
_PUBLIC_MODULES={}
def _public_module(name):
 if name not in _PUBLIC_MODULES:
  src=_PUBLIC_SOURCES[name];f='<p5b-public-'+name+'>';_public_linecache.cache[f]=(len(src),None,src.splitlines(True),f)
  m=_public_types.ModuleType('p5b_public_'+name);m.__file__=f;exec(compile(src,f,'exec'),m.__dict__);_PUBLIC_MODULES[name]=m
 return _PUBLIC_MODULES[name]
_public_original_attention=_best7_attention
def _best7_attention(qkv,data,state,plan):
 c=data['config'];arm=_PUBLIC_SELECTION.get((c['batch'],c['q_len'],bool(c['window'])))
 valid=c['num_q_heads']==16 and c['num_kv_heads']==2 and c['head_dim']==64 and c['page_size']==1
 if arm is None or not valid:return _public_original_attention(qkv,data,state,plan)
 packed=_public_module('input').raw(qkv,data)
 return _public_module(arm).attention(packed,data).reshape(c['batch']*c['q_len'],1024)
_public_original_prepare=prepare
def prepare(weights):
 # CUDA compilation is independent of the layer's synthetic prepare tuning.
 # Overlap it to keep the true300s prepare budget visible in P4's confirmation.
 from concurrent.futures import ThreadPoolExecutor
 with ThreadPoolExecutor(max_workers=2) as pool:
  futures=[pool.submit(_public_module(n).setup,torch.device('cuda')) for n in sorted(set(_PUBLIC_SELECTION.values()))]
  state=_public_original_prepare(weights)
  for f in futures:f.result()
 return state
