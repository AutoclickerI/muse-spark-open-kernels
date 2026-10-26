import importlib.util,sys
from pathlib import Path
from local_support import install_cache
D=Path(__file__).resolve().parent;R=D.parents[5];p=R/'problems/muse_decode_layer/dev/s22/screen.py';sp=importlib.util.spec_from_file_location('builds22_screen',p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);old=m.prior.prior.s19.old;original_install=m.install
def install(module,forced=False):
 records=original_install(module,forced);records['CACHE1_STANDIN']=install_cache(vars(module))
 for prefix in ('rvc_nfidx0','rvc_nbase1','rvc_nbase2'):
  original=getattr(module,prefix+'_module');cache={}
  def native_module(_original=original,_cache=cache,_prefix=prefix):
   native=_original()
   if id(native) not in _cache:
    class Wrapped:
     def __getattr__(self,name):return getattr(native,name)
     def run(self,*args):
      row=records.setdefault('NATIVE_'+_prefix,dict(eager=0,captured=0));row['captured' if module.torch.cuda.is_current_stream_capturing() else 'eager']+=1;return native.run(*args)
    _cache[id(native)]=Wrapped()
   return _cache[id(native)]
  setattr(module,prefix+'_module',native_module)
 for name in ('rvc_nfidx0_experts','rvc_nbase1_experts','rvc_nbase2_experts'):
  fn=getattr(module,name)
  def call(*a,_fn=fn,_name=name,**kw):
   row=records.setdefault('P3A_'+_name,dict(eager=0,captured=0));row['captured' if module.torch.cuda.is_current_stream_capturing() else 'eager']+=1;return _fn(*a,**kw)
  setattr(module,name,call)
 return records
old.install=install;original_main=old.local_eval.main
def main():
 args=sys.argv;args[args.index('--reps')+1]='2';args[args.index('--inputs')+1]='1';return original_main()
old.local_eval.main=main
if __name__=='__main__':old.main()
