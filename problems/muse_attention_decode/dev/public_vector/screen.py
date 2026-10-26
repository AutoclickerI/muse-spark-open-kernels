"""local_eval driver for every production graph arm, with explicit sm120 stand-ins."""
import argparse,hashlib,importlib.util,json,linecache,sys,time,types
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];O=H/'out';F=H.parent/'full_attention'
sys.path[:0]=[str(R/'tools'),str(F),str(R/'problems/muse_decode_layer/dev/s15')]
import local_eval as ev
ap=argparse.ArgumentParser();ap.add_argument('--forced',action='store_true');args=ap.parse_args()
mode='forced' if args.forced else 'normal'
def load(p,name):
 sp=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
scr=load(R/'problems/muse_decode_layer/dev/s21/screen.py','pv_s21_screen')
stand=load(H.parent/'public_ports/local_standins.py','pv_public_standins')
import rebuild_standins
def install(base,old,modules,native):
 for m in (base,old):
  scr.install(m,args.forced);rebuild_standins.install(m)
  if hasattr(m,'_hy_module'):
   for n in m._HY_SOURCES:
    x=m._hy_module(n);stand.operator(x,args.forced) if n=='operator_direct' else stand.lee(x)
  if hasattr(m,'_p5b_merge_setup'):m._p5b_merge_setup()
 for n,m in modules.items():
  if n.startswith('op_'):stand.operator(m,args.forced)
  elif n.startswith('lee_'):stand.lee(m)
 # Install native glue stand-ins on a namespace proxy; functions continue to
 # see the original dictionary, updated atomically from this proxy's attributes.
 proxy=types.SimpleNamespace(**native);scr.install(proxy,args.forced);rebuild_standins.install(proxy)
 native.clear();native.update(vars(proxy))
preview=O/('screen_'+mode+'.py')
source='# okc: problem=muse_decode_layer\n'+(O/'in_graph.py').read_text()+'''
def prepare(weights):return None
def kernel_fn(data,state):raise RuntimeError('batch screen must call lab')
'''
preview.write_text(source)
oldload=ev.load_module;holder={}
def loader(name,path):
 m=oldload(name,path)
 if name=='reference':
  # The payload generates its own untouched TASK weights; no outer duplicate.
  m.generate_weights=lambda *a,**k:None
 if name=='submission':
  m.PV_LOCAL_INSTALL=install;m.PV_PROFILE=False;m.SEEDS=m.SEEDS[:2];m.ROUNDS=2;m.CTX=257
  m.PV_ARCHIVE=lambda x:(O/(mode+'_raw.json')).write_text(json.dumps(x))
  holder['m']=m
 return m
ev.load_module=loader
def run_tests(problem,side,tests,seed0):
 m=holder['m'];res=m.lab(time.monotonic()+7200,None,None)['packet']
 assert res['complete'] and not res['failures'],res['failures']
 rows=[dict(case=cell,ok=all(n==len(m.SEEDS) for n in counts.values()),TASK=counts) for cell,counts in res['checks'].items()]
 (H/(mode+'_screen.json')).write_text(json.dumps(dict(mode=mode,payload_sha256=hashlib.sha256((O/'in_graph.py').read_bytes()).hexdigest(),passed=res['complete'],checks=res['checks'],TASK=sum(sum(x.values()) for x in res['checks'].values()),fixture_ctx=257,seeds=m.SEEDS,native_claim=False),indent=2)+'\n')
 print('PUBLIC_VECTOR_SCREEN',mode,rows,flush=True);return rows
ev.run_tests=run_tests
sys.argv=['local_eval.py',str(preview),'--tests-only','--json',str(O/(mode+'_local_eval.json'))]
ev.main()
