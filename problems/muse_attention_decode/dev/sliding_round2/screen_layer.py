"""Normal/forced whole-layer local_eval for the exact default P5 round2 patch."""
import importlib.util,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];sys.path.insert(0,str(R/'tools'))
import local_eval
original=local_eval.load_module

def load(name,path):
    m=original(name,path)
    if name=='submission':m._ap_fa3_scheduler_metadata=lambda *a,**kw:None
    return m
local_eval.load_module=load
spec=importlib.util.spec_from_file_location('r2audit',R/'problems/muse_decode_layer/dev/s9/audit.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.main()
