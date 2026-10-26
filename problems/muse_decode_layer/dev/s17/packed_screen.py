"""Exercise packed entry points against every official TASK case."""
import importlib.util,json,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3]
sys.path[:0]=[str(R/'tools'),str(H.parent/'s16')]
import local_eval
spec=importlib.util.spec_from_file_location('s17_screen',H/'screen.py');s=importlib.util.module_from_spec(spec);spec.loader.exec_module(s)
forced='--forced' in sys.argv;late='--late' in sys.argv;manifest=json.loads((H/'generated'/('lab_late_manifest.json' if late else 'lab_manifest.json')).read_text());original=local_eval.load_module

def loader(name,path):
    m=original(name,path)
    if name=='submission':s.install(m,forced)
    return m
local_eval.load_module=loader
mode=('late_' if late else '')+('forced' if forced else 'normal');out=H/'screen'/('packed_'+mode);out.mkdir(parents=True,exist_ok=True)
sys.argv=['local_eval.py',str(R/manifest['shipped_source']),'--tests-only','--json',str(out/'local.json')]
local_eval.main()
