import importlib.util,json,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];sys.path[:0]=[str(R/'tools'),str(H),str(R/'problems/muse_decode_layer/dev/p12_components')]
import local_eval,static_cache,local
static_cache.install(local_eval);forced='--forced' in sys.argv;loader=local_eval.load_module;calls={}
def load(name,path):
    m=loader(name,path)
    if name=='submission':calls.update(local.install(m,forced))
    return m
local_eval.load_module=load;out=H/'screen'/('forced' if forced else 'normal');out.mkdir(parents=True,exist_ok=True)
sys.argv=['local_eval.py',str(H/'candidate.py'),'--no-baseline','--reps','2','--inputs','3','--json',str(out/'result.json')]
local_eval.main();(out/'calls.json').write_text(json.dumps(calls,indent=2)+'\n')
