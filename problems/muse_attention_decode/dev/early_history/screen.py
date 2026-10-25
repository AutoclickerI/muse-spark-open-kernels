"""Normal and forced-Hopper whole-layer glue through local_eval, actual compact tile math on sm120."""
import importlib.util,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT/'tools'))
import local_eval
original=local_eval.load_module

def loader(name,path):
    m=original(name,path)
    if name=='submission':m._ap_fa3_scheduler_metadata=lambda *a,**kw:None
    return m
local_eval.load_module=loader
spec=importlib.util.spec_from_file_location('p5_early_audit',ROOT/'problems/muse_decode_layer/dev/s9/audit.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.main()
