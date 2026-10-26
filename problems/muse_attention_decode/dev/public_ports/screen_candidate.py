"""Screen the exact selected layer wrapper as attention, with genuine producer ABI."""
import argparse,gc,hashlib,importlib.util,json,sys,types
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];F=H.parent/'full_attention';sys.path[:0]=[str(R/'tools'),str(F),str(R/'problems/muse_decode_layer/dev/s15')]
import local_eval as ev
ap=argparse.ArgumentParser();ap.add_argument('--forced',action='store_true');ap.add_argument('--stem',default='candidate');a=ap.parse_args();mode='forced' if a.forced else 'normal';o=H/'out';raw=(o/(a.stem+'.py')).read_text()
# Component tests use identical selected runtime module creation/wrapping and
# input pack. Full layer prepare/G0 remains P4's confirmation on held-out data.
adapter=(F/'attention_adapter.py.inc').read_text().replace("partial_port_rows=ns['_PP_ROWS'],vq8_rows=ns['_VQ8_ROWS'],attention_plan={})","partial_port_rows=ns['_PP_ROWS'],vq8_rows=ns['_VQ8_ROWS'],attention_plan={},r2_rows=ns['_R2_SELECTION'],wr_rows=ns['_WINDOW_SELECTION'])")
s=raw.replace('# okc: problem=muse_decode_layer','# okc: problem=muse_attention_decode',1)+adapter
preview=o/(a.stem+'_'+mode+'_attention.py');preview.write_text(s);old=ev.load_module;holder={}
def loader(name,path):
 m=old(name,path)
 if name=='reference':m.CACHE_CHUNK=8192;m._fa3_scheduler_metadata=lambda *a,**kw:None
 if name=='submission':
  saved={k:getattr(m,k) for k in ('_attn_port_hopper','_fa3_on_device','ATTN_BACKEND')}
  spec=importlib.util.spec_from_file_location('public_verify',R/'problems/muse_decode_layer/dev/verify_best.py');v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v);v.install(m,False)
  if not a.forced:
   for k,z in saved.items():setattr(m,k,z)
  import rebuild_standins
  rebuild_standins.install(m)
  spec=importlib.util.spec_from_file_location('public_candidate_standins',H/'local_standins.py');v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
  for port in set(m._PUBLIC_SELECTION.values()):
   n=m._public_module(port);v.operator(n,a.forced) if port=='operator' else v.lee(n);n.setup('cuda')
  holder['m']=m
 return m
ev.load_module=loader
origtests=ev.tests_of
ev.tests_of=lambda md:origtests(md)+[dict(batch=b,q_len=l,ctx=257,page=1,heads='tp8',layer='sliding' if w else 'full',dist='edges') for l in (1,8) for b in (1,16,32) for w in (2048,0)]
def tests(problem,side,specs,seed0):
 m=holder['m'];rows=[]
 for i,spec in enumerate(specs):
  data=problem.make_input(spec,seed0+i);c=data['config'];b,l,w=c['batch'],c['q_len'],c['window'];row=dict(case=ev.describe(spec))
  try:
   # Additional12 fixtures exercise the observed S19 split/dtype/width; official
   # tests preserve their board ABI to verify uncovered dispatch as well.
   P=data['qkv'].reshape(1,b*l,-1);S=1
   if i>=16:
    abi={(1,1,2048):('f',1,1280),(1,1,0):('f',1,1280),(16,1,2048):('f',10,1280),(16,1,0):('f',7,2304),(32,1,2048):('f',1,2304),(32,1,0):('f',1,1280),(1,8,2048):('f',1,2304),(1,8,0):('f',1,2304),(16,8,2048):('f',3,2304),(16,8,0):('f',3,2304),(32,8,2048):('b',1,2304),(32,8,0):('b',1,2304)}
    typ,S,width=abi[b,l,w]
    if typ=='f':
     P=ev.torch.zeros((S,b*l,width),device='cuda');P[0].copy_(data['qkv'][...,:width].reshape(b*l,width).float())
     if width==1280:data['qkv'][...,1280:]=float('inf')
   if c['num_q_heads']!=16:fn=lambda:m.attention_partial_port((P,S),data,m._PP_FAST).reshape(b,l,-1)
   else:
    state=dict(inv_freq=m._inv_freq(c['rope_theta'],'cuda'),cluster_rows=m._CLUSTER_ROWS,partial_port_rows=m._PP_ROWS,vq8_rows=m._VQ8_ROWS,attention_plan={},r2_rows=m._R2_SELECTION,wr_rows=m._WINDOW_SELECTION)
    setting=m.BEST7_SETTINGS.get((b,l,w),m.BEST7_SETTINGS['other']);glue=setting[0][0] if isinstance(setting[0],tuple) else setting[0];d=dict(data,config=dict(c,tokens=b*l));fn=lambda:m._best7_attention((P,S),d,state,dict(glue=glue,attention=setting[1])).reshape(b,l,-1)
   y=fn();ev.torch.cuda.synchronize();ok,msg=problem.check(data,y);row.update(eager_ok=ok,eager_msg=msg)
   stream=ev.torch.cuda.Stream();stream.wait_stream(ev.torch.cuda.current_stream())
   with ev.torch.cuda.stream(stream):y=fn()
   ev.torch.cuda.current_stream().wait_stream(stream);g=ev.torch.cuda.CUDAGraph()
   with ev.torch.cuda.graph(g):y=fn()
   y.fill_(float('nan'));g.replay();g.replay();ev.torch.cuda.synchronize();ok,msg=problem.check(data,y);row.update(ok=ok and row['eager_ok'],replay_ok=ok,msg=msg);del g,y
  except Exception as e:row.update(ok=False,msg=type(e).__name__+':'+str(e))
  print('PUBLIC_CANDIDATE',mode,row,flush=True);rows.append(row);data=P=None;gc.collect();ev.torch.cuda.empty_cache()
 return rows
ev.run_tests=tests
sys.argv=['local_eval.py',str(preview),'--tests-only','--json',str(H/(a.stem+'_'+mode+'_screen.json'))];ev.main()
