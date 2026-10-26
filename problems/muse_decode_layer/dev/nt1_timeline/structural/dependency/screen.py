"""S19 whole-layer smoke with its established Hopper stand-ins;changed slice math stays native."""
import importlib.util,sys
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[5]
p=R/'problems/muse_decode_layer/dev/s19/screen.py';sp=importlib.util.spec_from_file_location('dep_s19_screen',p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);install=m.old.install

def wrapped(module,forced=False):
 counts=install(module,forced);name='dep_experts' if hasattr(module,'dep_experts') else 'dpx_experts';fn=getattr(module,name)
 def call(*a,**kw):
  row=counts.setdefault(name,dict(eager=0,captured=0));row['captured' if module.torch.cuda.is_current_stream_capturing() else 'eager']+=1;return fn(*a,**kw)
 setattr(module,name,call)
 return counts
m.old.install=wrapped
original=m.old.local_eval.main

def run():
 # Keep13TASK and12 shapes;two replays/oneinput are enough for the local glue smoke.
 args=sys.argv
 args[args.index('--reps')+1]='2';args[args.index('--inputs')+1]='1'
 return original()
m.old.local_eval.main=run
m.old.main()
