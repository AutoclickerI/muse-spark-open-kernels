"""Official local screen of exact P4 patch, with declared Hopper body stand-ins."""
import importlib.util,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3]
def load(name,path):
 sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
local=load('p10_patch_local',H/'screen.py');local.s19=load('p10_patch_s21',R/'problems/muse_decode_layer/dev/s21/screen.py')
def install(m,forced=False):local.install(m);return []
local.s19.prior.s19.old.install=install;local.s19.prior.s19.old.main()
