"""Normal/forced full-layer TASK and graph screens, using two real local WGs."""
import argparse,importlib.util,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3]
sys.path.insert(0,str(R/'tools'))
import local_eval,kernel,local_host
p=argparse.ArgumentParser(add_help=False);p.add_argument('--out',type=Path);a,_=p.parse_known_args()
if a.out:a.out.mkdir(parents=True,exist_ok=True)
original=local_eval.load_module
def load(name,path):
    m=original(name,path)
    if name=='submission':
        print(local_host.install(m),flush=True)
        m.wr_KERNELS={n:kernel.source(c,local=True,stream=n.startswith('STREAM')) for n,c in m.wr_TOPOLOGY.items() if n in m.wr_KERNELS}
        print('Screen only: both real WGs execute; global last-WG publication replaces native DSM. ResidentC2 uses C4 capacity; streaming slices are unchanged.',flush=True)
    return m
local_eval.load_module=load
spec=importlib.util.spec_from_file_location('window_audit',R/'problems/muse_decode_layer/dev/s9/audit.py');audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit);audit.main()
