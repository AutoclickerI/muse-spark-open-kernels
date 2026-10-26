import importlib.util,sys
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[6];p=R/'problems/muse_decode_layer/dev/s19/screen.py';sp=importlib.util.spec_from_file_location('early_s19',p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);install=m.old.install

def wrapped(module,forced=False):
 counts=install(module,forced)
 for name in ('rvc_nbase1_experts','rvc_nbase2_experts'):
  fn=getattr(module,name)
  def call(*a,_fn=fn,_name=name,**kw):
   row=counts.setdefault('EARLY_'+_name,dict(eager=0,captured=0));row['captured' if module.torch.cuda.is_current_stream_capturing() else 'eager']+=1;return _fn(*a,**kw)
  setattr(module,name,call)
 return counts
m.old.install=wrapped;original=m.old.local_eval.main

def run():
 args=sys.argv;args[args.index('--reps')+1]='2';args[args.index('--inputs')+1]='1';return original()
m.old.local_eval.main=run;m.old.main()
