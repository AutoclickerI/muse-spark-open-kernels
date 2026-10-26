"""Run P4-facing exact layer ABI under official TASK with only DSM stood in."""
import importlib.util,json,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];sys.path[:0]=[str(R/'tools')]
import local_eval
def module(name,path):
    sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
stand=module('sgq_current_screen',R/'problems/muse_decode_layer/dev/s18/screen.py')
forced='--forced' in sys.argv;loader=local_eval.load_module;seen={}
def load(name,path):
    m=loader(name,path)
    if name!='submission':return m
    stand.install(m,forced);m.SGQ_LOCAL_REDUCE=True;original=m.sgq_project
    def project(x,n,arm,bank,query=0):
        key=str((x.shape[0],n,arm));seen[key]=seen.get(key,0)+1
        return original(x,n,arm,bank,query)
    m.sgq_project=project;return m
local_eval.load_module=load;folder=H/'screen'/('layer_qkv_forced' if forced else 'layer_qkv_normal');folder.mkdir(parents=True,exist_ok=True)
sys.argv=['local_eval.py',str(H/'layer_qkv_candidate.py'),'--no-baseline','--reps','2','--inputs','3','--json',str(folder/'result.json')]
try:local_eval.main()
finally:(folder/'calls.json').write_text(json.dumps(seen,indent=2)+'\n')
