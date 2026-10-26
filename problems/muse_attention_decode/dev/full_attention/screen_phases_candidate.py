"""Screen final layer-patch attention calls through local_eval, without unrelated prepare."""
import argparse,hashlib,importlib.util,json,linecache,sys,types,gc
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];o=H/'phases';sys.path[:0]=[str(R/'tools'),str(H),str(R/'problems/muse_decode_layer/dev/s15')]
import local_eval as ev
ap=argparse.ArgumentParser();ap.add_argument('--forced',action='store_true');ap.add_argument('--stem',default='candidate');a=ap.parse_args();mode='forced' if a.forced else 'normal'
s=(o/(a.stem+'.py')).read_text();original=ev.load_module;holder={}
# Packaging sees the true final layer candidate, with only an attention header
# and adapter for this component screen. No local import enters the artifact.
adapter=(H/'attention_adapter.py.inc').read_text()
setup="\ndef prepare(weights):\n    if '_p5_phase_setup' in globals():_p5_phase_setup(torch.device('cuda'))\n    return None\n"
preview=o/(a.stem+'_attention.py');preview.write_text(s.replace('# okc: problem=muse_decode_layer','# okc: problem=muse_attention_decode',1)+adapter+setup)
def loader(name,path):
 m=original(name,path)
 if name=='reference':m.CACHE_CHUNK=8192
 if name=='submission':
  saved={k:getattr(m,k) for k in ('_attn_port_hopper','_fa3_on_device','ATTN_BACKEND')};oldprepare=m.prepare
  spec=importlib.util.spec_from_file_location('phase_force',R/'problems/muse_decode_layer/dev/verify_best.py');helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper);helper.install(m,False);m.prepare=oldprepare
  if not a.forced:
   for k,v in saved.items():setattr(m,k,v)
  import rebuild_standins,local_n128
  rebuild_standins.install(m)
  oldmod=getattr(m,'_p5_phase_mod',None);cache={}
  if oldmod:
   def mod(role):
    if not role.startswith('E'):return oldmod(role)
    if role not in cache:
     n=types.ModuleType('phase_E_local');n.__dict__.update(m.__dict__);n.VQ8_CUDA=m._P5_PHASE_SOURCES[role][0];n.VQ8_CPP=m._P5_PHASE_SOURCES[role][1];local_n128.install(n);cache[role]=n._vq8_module()
    return cache[role]
   m._p5_phase_mod=mod
  m.prepare(None) # attention host has no weights: explicitly run the same shape-only component setup
  holder['module']=m
 return m
ev.load_module=loader
oldtests=ev.tests_of
ev.tests_of=lambda md:oldtests(md)+[dict(batch=b,q_len=l,ctx=80000,layer='full',heads='tp8',page=1,dist='coherent',spread='agentic' if b>1 else 'fixed') for b in (1,16,32) for l in (1,8)]
def run(problem,side,cases,seed0):
 rows=[]
 for i,c in enumerate(cases):
  d=problem.make_input(c,seed0+i);row=dict(case=ev.describe(c))
  try:
   y=side.call(d);ev.torch.cuda.synchronize();ok,msg=problem.check(d,y);row.update(eager_ok=ok,eager_msg=msg)
   g,y=ev.capture(side.call,d);y.fill_(float('nan'));g.replay();g.replay();ev.torch.cuda.synchronize();ok,msg=problem.check(d,y);row.update(replay_ok=ok,ok=row['eager_ok'] and ok,msg=msg);del g,y
  except Exception as e:row.update(ok=False,msg=type(e).__name__+':'+str(e))
  print('PHASE_CANDIDATE',mode,row,flush=True);rows.append(row);del d;gc.collect();ev.torch.cuda.empty_cache()
 return rows
ev.run_tests=run
sys.argv=['local_eval.py',str(preview),'--tests-only','--json',str(o/(a.stem+'_'+mode+'_screen.json'))];ev.main()
