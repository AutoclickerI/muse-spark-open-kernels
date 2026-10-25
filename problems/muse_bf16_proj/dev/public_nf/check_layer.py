"""Normal/forced full local_eval, preserving the established Hopper stand-ins."""
import hashlib,importlib.util,json,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3]
sys.path[:0]=[str(R/'tools'),str(R/'problems/muse_decode_layer/dev/s12')]
import local_eval,compile_cache
compile_cache.install(local_eval)
forced='--forced' in sys.argv
out=H/('forced' if forced else 'normal');out.mkdir(exist_ok=True)
loader=local_eval.load_module;counts={}
def wrapped(name,path):
 m=loader(name,path)
 if name!='submission':return m
 if forced:
  p=R/'problems/muse_decode_layer/dev/best7/force.py'
  sp=importlib.util.spec_from_file_location('nf_layer_force',p);f=importlib.util.module_from_spec(sp);sp.loader.exec_module(f);f.install(m,False,False)
  m.NF_PORT_FORCE=True
 original=m.nf_project
 def counted(x,name,state):
  key=str((x.shape[0],name));r=counts.setdefault(key,dict(eager=0,captured=0,native=(x.shape[0],name) in state['plans']))
  r['captured' if m.torch.cuda.is_current_stream_capturing() else 'eager']+=1
  return original(x,name,state)
 m.nf_project=counted
 return m
local_eval.load_module=wrapped
sys.argv=['local_eval.py',str(H/'cand_nf_s15.py'),'--no-baseline','--reps','4','--inputs','3','--json',str(out/'local.json')]
try:local_eval.main()
finally:(out/'nf_calls.json').write_text(json.dumps(counts,indent=2)+'\n')
