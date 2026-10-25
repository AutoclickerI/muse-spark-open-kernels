"""Exercise whole-layer pairing glue locally with explicit Hopper stand-ins."""
import importlib.util,runpy,time,json,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3]
sys.path[:0]=[str(R/'tools'),str(R/'problems/muse_decode_layer/dev/best7')]
import local_eval
sp=importlib.util.spec_from_file_location('nf_layer_force',R/'problems/muse_decode_layer/dev/best7/force.py')
force=importlib.util.module_from_spec(sp);sp.loader.exec_module(force)
s=runpy.run_path(str(H/'s15_layer_lab.py'))
g=s['lab'].__globals__
g['NF_LAYER_FORCE']=lambda m:force.install(m,False,False)
# Screen all proposed families, not a performance admission. One timing draw locally
# keeps 16 independent fresh inputs and every full-layer graph/gate under test.
p=runpy.run_path(str(H/'port_body.py'))
if '--all' in sys.argv:
 g['NF_LAYER_ROWS']=set(p['NF_SMALL'])|p['NF_WIDE']
 g['NF_LAYER_ROWS']-={(1,'qkv_gate'),(1,'qkv3'),(1,'pool_router')}
a=s['ab_time'];g['ab_time']=lambda graphs,rounds:a(graphs,1)
g['lab_cold_l2']=lambda:None
lines=s['lab'](time.monotonic()+1200,None)
for line in lines:print(line,flush=True)
(H/'layer_lab_validation.json').write_text(json.dumps(dict(passed=not any(l.startswith(('FAIL','SKIP')) for l in lines),lines=lines),indent=2)+'\n')
assert not any(l.startswith(('FAIL','SKIP')) for l in lines)
