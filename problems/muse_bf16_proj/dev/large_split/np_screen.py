"""Official local layer screen on the minimal optional integration patch."""
import importlib.util,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3]
def load(name,path):
 sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
s19=load('np_s19_screen',R/'problems/muse_decode_layer/dev/s19/screen.py');portable=load('np_portable',H.parent/'dense_all/portable.py')
def install(m,forced=False):
 records=s19.install(m,forced);m.NP_LOCAL_MM=portable.mm;return records
s19.old.install=install;s19.old.main()
