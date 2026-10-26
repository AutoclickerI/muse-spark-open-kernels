"""Normal/forced whole-layer local_eval for the exact default P5 round2 patch."""
import argparse,importlib.util,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];sys.path.insert(0,str(R/'tools'))
parser=argparse.ArgumentParser(add_help=False);parser.add_argument('--out',type=Path)
args,_=parser.parse_known_args()
if args.out:args.out.mkdir(parents=True,exist_ok=True)
import local_eval
original=local_eval.load_module

def load(name,path):
    m=original(name,path)
    if name=='submission':
        m._ap_fa3_scheduler_metadata=lambda *a,**kw:None
        original_attention=m.r2_attention
        def attention(data,qkv=None,kind='last',qt=1,ns=8):
            if kind in ('dsm','dsm_old'):ns=1
            return original_attention(data,qkv,kind,qt,ns)
        m.r2_attention=attention
        print('LOCAL ONLY: native DSM gangs use the genuine C1 launch; CUDA math and layer glue execute',flush=True)
    return m
local_eval.load_module=load
spec=importlib.util.spec_from_file_location('r2audit',R/'problems/muse_decode_layer/dev/s9/audit.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.main()
