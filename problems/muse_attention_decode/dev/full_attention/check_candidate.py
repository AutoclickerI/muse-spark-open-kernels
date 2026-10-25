"""Exact whole-layer normal/forced screens; require both changed attention paths in captured calls."""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3]
sys.path.insert(0,str(R/'tools'));import local_eval
sys.path.insert(0,str(R/'problems/muse_decode_layer/dev/s12'));import compile_cache
compile_cache.install(local_eval)
cand=H/'s13_ldsv128.py';forced='--forced' in sys.argv
out=H/'candidate_check'/('forced' if forced else 'normal');out.mkdir(parents=True,exist_ok=True)
original=local_eval.load_module;counts={}
def loader(name,path):
 m=original(name,path)
 if name=='submission':
  fn=m._vq8_attention
  def attention(qkv,data,setting):
   cfg=data['config'];key=str((cfg['batch'],cfg['q_len'],bool(cfg['window'])))
   assert setting==(3,1),(key,setting)
   row=counts.setdefault(key,dict(eager=0,captured=0))
   row['captured' if m.torch.cuda.is_current_stream_capturing() else 'eager']+=1
   return fn(qkv,data,setting)
  m._vq8_attention=attention
 return m
local_eval.load_module=loader
spec=importlib.util.spec_from_file_location('full_candidate_audit',R/'problems/muse_decode_layer/dev/s9/audit.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
sys.argv=['audit.py','--candidate',str(cand),'--out',str(out)]+(['--forced'] if forced else [])
try:audit.main()
except SystemExit as exc:
 if exc.code==0:
  assert set(counts)=={'(16, 8, False)','(32, 8, False)'} and all(x['captured']>0 for x in counts.values()),counts
  (out/'attention_calls.json').write_text(json.dumps(dict(sha256=hashlib.sha256(cand.read_bytes()).hexdigest(),calls=counts),indent=2)+'\n')
 raise
