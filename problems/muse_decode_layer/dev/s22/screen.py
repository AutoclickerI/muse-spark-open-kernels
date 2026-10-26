"""S22 local native glue, retaining S21's explicit Hopper arithmetic stand-ins."""
import importlib.util
from pathlib import Path
H=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('s22_prior_screen',H.parent/'s21/screen.py');prior=importlib.util.module_from_spec(sp);sp.loader.exec_module(prior)
def install(m,forced=False):
 records=prior.install(m,forced)
 if hasattr(m,'_hy_module'):
  original=m._hy_module
  def module(name):
   port=original(name)
   if not getattr(port,'_S22_LOCAL_INSTALLED',False):
    if name=='operator_direct':prior.public_standins.operator(port,forced)
    else:prior.public_standins.lee(port)
    port._S22_LOCAL_INSTALLED=True
   return port
  m._hy_module=module
 return records
if __name__=='__main__':prior.prior.s19.old.install=install;prior.prior.s19.old.main()
