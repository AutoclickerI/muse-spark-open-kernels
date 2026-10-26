"""Normal/forced whole-layer screens, real probability math and explicit DSM substitutes."""
import argparse,importlib.util,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];sys.path.insert(0,str(R/'tools'))
import local_eval
spec=importlib.util.spec_from_file_location('p5_window_local_host',H.parent/'window_rebuild/local_host.py');local_host=importlib.util.module_from_spec(spec);spec.loader.exec_module(local_host)
import structural
p=argparse.ArgumentParser(add_help=False);p.add_argument('--out',type=Path);a,_=p.parse_known_args()
if a.out:a.out.mkdir(parents=True,exist_ok=True)
original=local_eval.load_module
def load(name,path):
    m=original(name,path)
    if name=='submission':
        print(local_host.install(m),flush=True);m.so_LOCAL=True;m.so_SOURCES={n:structural.source(n[0],True) for n in m.so_SOURCES}
        print('Screen only: decode C1; verify two real WGs/nonspinning global combine. Probability exponent approximation executes as shipped.',flush=True)
    return m
local_eval.load_module=load
spec=importlib.util.spec_from_file_location('p5_so_audit',R/'problems/muse_decode_layer/dev/s9/audit.py');audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit);audit.main()
