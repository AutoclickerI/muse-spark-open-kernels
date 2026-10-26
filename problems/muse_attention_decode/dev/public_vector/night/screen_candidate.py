"""P4's 13 TASK/12 production graph screen for each exact packed candidate."""
import importlib.util,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[4]
def load(path,name):
 sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
scr=load(R/'problems/muse_decode_layer/dev/s22/screen.py','night_candidate_s22')
stand=load(H.parents[1]/'public_ports/local_standins.py','night_candidate_public')
original=scr.install
def install(m,forced=False):
 records=original(m,forced)
 if hasattr(m,'_night_module'):
  get=m._night_module
  def module(name):
   out=get(name)
   if name.startswith('op_'):stand.operator(out,forced)
   elif name.startswith('lee_'):stand.lee(out)
   return out
  m._night_module=module
 return records
scr.prior.prior.s19.old.install=install
scr.prior.prior.s19.old.main()
