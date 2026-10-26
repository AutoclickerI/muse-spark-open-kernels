"""Real Gluon and repacking checks; Hopper phase host glue uses tiny stand-ins."""
import importlib.util,json,sys,time
from pathlib import Path
import torch
H=Path(__file__).resolve().parent;R=H.parents[3];sys.path.insert(0,str(R/'tools'));import local_eval
sp=importlib.util.spec_from_file_location('public_moe',H/'out/public_moe.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
ck=m.module('localTASK',m.BUNDLE['checker']);op=m.module('localO',m.BUNDLE['operator']);lee=m.module('localL',m.BUNDLE['lee'])
w=ck.generate_weights(183528937);st=op.prepare(w);sl=lee.prepare_lee(w);checks=[]
# Two real X emulation configs use the donor's mma.sync path. This validates
# layouts and phase dependencies, while sm90a compilation covers actual WGMMA.
mods={i:m.lee_module('X',flags,emulate=True) for i,flags in ((1,('-DRPW=2','-DTPX=16','-DNSL=2')),(2,('-DNCW=16','-DTPX=16','-DNSL=2')))}
for M,dist in ((1,'normal'),(4,'normal'),(8,'normal'),(16,'normal'),(32,'normal'),(33,'normal'),(128,'normal'),(128,'skewed'),(256,'normal'),(257,'normal')):
 d=ck.generate_input(bs=M,sharding='ep8',seed=1899321+M,dist=dist);ref=ck.ref_kernel(d,w);plans={'O':op.operator_plan(d,st)}
 for i,mod in mods.items():
  plans['X'+str(i)]=lee.lee_plan(d,sl,mod);plans['S'+str(i)]=lee.lee_plan(d,sl,mod,split=True)
 for name,p in plans.items():
  y=m.run_plan(p);torch.cuda.synchronize();ok,msg=ck.compare(y,ref,ck.REL_L2_TOL);assert ok,(M,dist,name,'eager',msg)
  g,y=m.capture(lambda p=p:m.run_plan(p));g.replay();g.replay();torch.cuda.synchronize();ok,msg=ck.compare(y,ref,ck.REL_L2_TOL);assert ok,(M,dist,name,'graph',msg);del g
  checks.append([M,dist,name]);print('PASS',M,dist,name,flush=True)
 plans=d=ref=None;torch.cuda.empty_cache()
# Small-M F host glue is checked with a stand-in (real sm90a source compiled).
class F:
 def port_f(self,ptr,M,lo,stream):
  assert len(ptr)==12 and M<=32 and lo==0 and isinstance(stream,int);self.called=True
fake=F();p=lee.lee_plan(ck.generate_input(bs=8,sharding='ep8',seed=188),sl,fake,'F');m.run_plan(p);assert fake.called
(H/'local_validation.json').write_text(json.dumps(dict(checks=checks,passed=True,Hopper='native sources compiled sm90a; X emulated mma.sync; F host glue stand-in; actual H200 still required'),indent=2)+'\n')
