"""Normal/forced local screen of an S22-based P10 candidate: S22's hybrid stand-ins under the night stand-ins.

Same chain as candidate_screen.py, with S22's screen (S21 install + direct-input/Lee hybrid stand-ins) in
place of S21's as the public-port screen's base. Usage: tools/img python screen_s22.py FILE --out DIR [--forced]
"""
import importlib.util,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[4]
def load(name,path):
 sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sys.modules[name]=m;sp.loader.exec_module(m);return m
local=load('ng_candidate_standins',H/'local_standins.py')
s22=load('ng_s22_screen',R/'problems/muse_decode_layer/dev/s22/screen.py');local.prior.s19=s22
local.prior.calls=local.calls  # port and night stand-ins record into one list, dumped to calls.json at exit
def install(m,forced=False):local.install(m,forced);return local.calls
s22.prior.prior.s19.old.install=install;s22.prior.prior.s19.old.main()
