"""Exact P4-protocol batch in local_eval, with explicit Hopper arithmetic glue."""
import argparse,base64,hashlib,importlib.util,json,linecache,lzma,sys,time,types
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[4];O=H/'out';F=H.parents[1]/'full_attention'
sys.path[:0]=[str(R/'tools'),str(F),str(R/'problems/muse_decode_layer/dev/s15')]
import local_eval as ev
ap=argparse.ArgumentParser();ap.add_argument('--forced',action='store_true');ap.add_argument('--full-protocol',action='store_true');ap.add_argument('--ctx',type=int,default=257);ap.add_argument('--name',default='p5b_night');a=ap.parse_args();mode=a.name+'_'+('forced' if a.forced else 'normal')+('_protocol' if a.full_protocol else '')
def load(path,name):
 sp=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
scr=load(R/'problems/muse_decode_layer/dev/s22/screen.py','p5b_night_s22_standins');stand=load(H.parents[1]/'public_ports/local_standins.py','p5b_night_public_standins')
import rebuild_standins
def module(name,m):
 if name.startswith('op_'):stand.operator(m,a.forced)
 elif name.startswith('lee_'):stand.lee(m)
def install(m):
 scr.install(m,a.forced);rebuild_standins.install(m);m._NIGHT_LOCAL_MODULE=module
 if hasattr(m,'_night_module'):
  original=m._night_module
  def get(name):
   out=original(name);module(name,out);return out
  m._night_module=get
 if hasattr(m,'_p5b_merge_setup'):m._p5b_merge_setup()
 # Native d1 last-CTA uses the existing pointer/glue stand-in. Its vector
 # readers and exact sm90a source were checked separately in public_vector.
payload=(O/(a.name+'.py')).read_text();preview=O/('screen_'+mode+'.py')
preview.write_text('# okc: problem=muse_decode_layer\n'+payload+"\ndef prepare(weights):return None\ndef kernel_fn(data,state):raise RuntimeError('batch screen must call lab')\n")
holder={};oldload=ev.load_module
flush=[]
def cold():
 if not flush:flush.append(ev.torch.empty(1<<30,device='cuda',dtype=ev.torch.uint8))
 flush[0].view(ev.torch.int64).sum()
def loader(name,path):
 m=oldload(name,path)
 if name=='reference':m.generate_weights=lambda *a,**kw:None
 if name=='submission':
  g=m._run.__globals__;g['S16_LOCAL_INSTALL']=install;g['lab_cold_l2']=cold
  if not a.full_protocol:
   g['AB_SEEDS']=g['AB_SEEDS'][:2];g['S21_PROTOCOL']=dict(g['S21_PROTOCOL'],seed_count=2,rounds=2)
  # Short fixture contexts only in the local rehearsal; native payload is exact.
  original_host=g['s19_host']
  def host():
   base=original_host();orig=base['_ap_generate_input']
   base['_ap_fa3_scheduler_metadata']=lambda *a,**kw:None
   def make(*args,**kw):kw['ctx']=a.ctx;return orig(*args,**kw)
   base['_ap_generate_input']=make
   return base
  g['s19_host']=host
  g['S21_LOCAL_ARCHIVE']=lambda raw:(O/(mode+'_raw.json')).write_text(json.dumps(raw))
  holder['m']=m
 return m
ev.load_module=loader
def tests(problem,side,specs,seed0):
 lines=holder['m'].lab(time.monotonic()+7200,None,None)
 encoded=next(x.split(' ',1)[1] for x in lines if x.startswith('P5N_XZ85 '))
 envelope=json.loads(lzma.decompress(base64.b85decode(encoded)))
 codec=load(R/'problems/muse_decode_layer/dev/s21/evidence.py','p5b_night_codec');raw=codec.s21_decode_samples(codec.s21_unpack_evidence(envelope))
 assert raw['complete'] and not raw['failures'],raw['failures']
 from lab_packet import encode_packet,paired_evidence
 wire=encode_packet('p5b_night',hashlib.sha256(payload.encode()).hexdigest()[:16],dict(lines=lines,complete=None,format='legacy-lines'))+'\n'
 assert len(wire.encode())<=11700,len(wire.encode())
 tail=(wire+'P'*4000)[-16000:];got=paired_evidence(tail,'P5N');assert got==envelope
 rows=[dict(case=label,ok=True,arms=list(v['r'])) for label,v in raw['shapes'].items()]
 (H/(mode+'_screen.json')).write_text(json.dumps(dict(mode=mode,payload_sha256=hashlib.sha256(payload.encode()).hexdigest(),TASK=raw['task_checks'],nearby_TASK=raw['nearby_task_checks'],expected_TASK=raw['expected_task_checks'],complete=raw['complete'],failures=raw['failures'],native_claim=False,seed_count=len(raw['seed_ids']),rounds=raw['measurement_protocol']['rounds'],fixture_ctx=a.ctx,wire_bytes=len(wire.encode()),tail_with_4000_suffix_pass=True),indent=2)+'\n')
 print('NIGHT_SCREEN',mode,raw['task_checks'],raw['nearby_task_checks'],rows,flush=True);return rows
ev.run_tests=tests
sys.argv=['local_eval.py',str(preview),'--tests-only','--json',str(O/(mode+'_local_eval.json'))];ev.main()
