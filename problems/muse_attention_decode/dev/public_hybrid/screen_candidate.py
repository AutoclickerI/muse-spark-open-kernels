"""Screen exact appended S21/S22 hybrid runtime, not an independent reimplementation."""
import argparse,gc,hashlib,importlib.util,json,sys
from pathlib import Path
H=Path(__file__).resolve().parent;P=H.parent/'public_ports';R=H.parents[3];F=H.parent/'full_attention';sys.path[:0]=[str(R/'tools'),str(F),str(R/'problems/muse_decode_layer/dev/s15')]
import local_eval as ev
ap=argparse.ArgumentParser();ap.add_argument('--forced',action='store_true');a=ap.parse_args();mode='forced' if a.forced else 'normal';o=H/'out';raw=(o/'candidate.py').read_text()
adapter=(F/'attention_adapter.py.inc').read_text().replace("partial_port_rows=ns['_PP_ROWS'],vq8_rows=ns['_VQ8_ROWS'],attention_plan={})","partial_port_rows=ns['_PP_ROWS'],vq8_rows=ns['_VQ8_ROWS'],attention_plan={},r2_rows=ns['_R2_SELECTION'],wr_rows=ns['_WINDOW_SELECTION'])")
preview=o/('candidate_'+mode+'_attention.py');preview.write_text(raw.replace('# okc: problem=muse_decode_layer','# okc: problem=muse_attention_decode',1)+adapter)
old=ev.load_module;holder={}
def loader(name,path):
 m=old(name,path)
 if name=='reference':m.CACHE_CHUNK=8192;m._fa3_scheduler_metadata=lambda *a,**kw:None
 if name=='submission':
  spec=importlib.util.spec_from_file_location('hybrid_s21_standins',R/'problems/muse_decode_layer/dev/s21/screen.py');helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper);helper.install(m,a.forced)
  import rebuild_standins
  rebuild_standins.install(m)
  spec=importlib.util.spec_from_file_location('public_candidate_standins',P/'local_standins.py');v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
  # Existing S21 modules remain its own cells; only local native glue is replaced.
  for n in sorted(set(m._PUBLIC_SELECTION.values())):
   z=m._public_module(n);v.operator(z,a.forced) if n=='operator' else v.lee(z);z.setup('cuda')
  for n in sorted(m._HY_SOURCES):
   z=m._hy_module(n);v.operator(z,a.forced) if n=='operator_direct' else v.lee(z);z.setup('cuda')
  if hasattr(m,'_p5b_merge_setup'):m._p5b_merge_setup()
  holder['m']=m
 return m
ev.load_module=loader
orig=ev.tests_of;ev.tests_of=lambda md:orig(md)+[dict(batch=b,q_len=l,ctx=257,page=1,heads='tp8',layer='sliding' if w else 'full',dist='edges') for l in (1,8) for b in (1,16,32) for w in (2048,0)]
def tests(problem,side,specs,seed0):
 m=holder['m'];rows=[]
 abi=json.loads((P/'receipt_data.json').read_text())['ABI']
 for i,spec in enumerate(specs):
  d=problem.make_input(spec,seed0+i);c=d['config'];b,l,w=c['batch'],c['q_len'],c['window'];row=dict(case=ev.describe(spec))
  try:
   PP=d['qkv'].reshape(1,b*l,-1);S=1
   if i>=16:
    typ,S,width=abi[f'{b}x{l}'+('s' if w else 'f')]
    if typ=='f':
     PP=ev.torch.zeros((S,b*l,width),device='cuda');PP[0].copy_(d['qkv'][...,:width].reshape(b*l,width).float())
     if width==1280:d['qkv'][...,1280:]=float('inf')
   if c['num_q_heads']!=16:fn=lambda:m.attention_partial_port((PP,S),d,m._PP_FAST).reshape(b,l,-1)
   else:
    state=dict(inv_freq=m._inv_freq(c['rope_theta'],'cuda'),cluster_rows=m._CLUSTER_ROWS,partial_port_rows=m._PP_ROWS,vq8_rows=m._VQ8_ROWS,attention_plan={},r2_rows=m._R2_SELECTION,wr_rows=m._WINDOW_SELECTION)
    st=m.BEST7_SETTINGS.get((b,l,w),m.BEST7_SETTINGS['other']);glue=st[0][0] if isinstance(st[0],tuple) else st[0];dd=dict(d,config=dict(c,tokens=b*l));fn=lambda:m._best7_attention((PP,S),dd,state,dict(glue=glue,attention=st[1])).reshape(b,l,-1)
   y=fn();ev.torch.cuda.synchronize();ok,msg=problem.check(d,y);row.update(eager_ok=ok,eager_msg=msg)
   stream=ev.torch.cuda.Stream();stream.wait_stream(ev.torch.cuda.current_stream())
   with ev.torch.cuda.stream(stream):y=fn()
   ev.torch.cuda.current_stream().wait_stream(stream);g=ev.torch.cuda.CUDAGraph()
   with ev.torch.cuda.graph(g):y=fn()
   y.fill_(float('nan'));g.replay();g.replay();ev.torch.cuda.synchronize();ok,msg=problem.check(d,y);row.update(ok=ok and row['eager_ok'],replay_ok=ok,msg=msg);del g,y
  except Exception as e:row.update(ok=False,msg=type(e).__name__+':'+str(e))
  print('HYBRID_CANDIDATE',mode,row,flush=True);rows.append(row);d=PP=None;gc.collect();ev.torch.cuda.empty_cache()
 return rows
ev.run_tests=tests;sys.argv=['local_eval.py',str(preview),'--tests-only','--json',str(H/('candidate_'+mode+'_screen.json'))];ev.main()
