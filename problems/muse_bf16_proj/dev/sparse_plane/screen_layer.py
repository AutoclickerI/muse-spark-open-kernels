import importlib.util,sys,json
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];sys.path[:0]=[str(R/'tools')]
import local_eval
def module(name,path):
    sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
cache=module('p10_pool_static',R/'problems/muse_decode_layer/dev/p12_components/static_cache.py');cache.install(local_eval)
stand=module('p10_pool_stand',R/'problems/muse_decode_layer/dev/s15/local_standins.py');forced='--forced' in sys.argv
loader=local_eval.load_module;calls={}
def load(name,path):
    m=loader(name,path)
    if name=='submission':
        if forced:module('p10_pool_force',R/'problems/muse_decode_layer/dev/best7/force.py').install(m,False,False)
        stand.install(m);fn=m._project_variant
        def project(x,w,name,state,*a,**kw):
            if name=='pool_router' and x.shape[0] in m.P10_POOL_TOKENS:
                label='graph' if m.torch.cuda.is_current_stream_capturing() else 'eager';calls[label]=calls.get(label,0)+1
            return fn(x,w,name,state,*a,**kw)
        m._project_variant=project
    return m
local_eval.load_module=load
out=H/'screen_layer'/('forced' if forced else 'normal');out.mkdir(parents=True,exist_ok=True)
sys.argv=['local_eval.py',str(H/'cand_s15_pool_raw.py'),'--no-baseline','--reps','2','--inputs','3','--json',str(out/'result.json')]
try:local_eval.main()
finally:(out/'calls.json').write_text(json.dumps(calls,indent=2)+'\n')
