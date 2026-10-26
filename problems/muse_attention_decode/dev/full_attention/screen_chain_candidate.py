"""Exact selected S19 patch attention dispatch through normal/forced local_eval."""
import argparse,importlib.util,json,sys,types,gc
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];o=H/'chain';sys.path[:0]=[str(R/'tools'),str(H),str(R/'problems/muse_decode_layer/dev/s15')]
import local_eval as ev
ap=argparse.ArgumentParser();ap.add_argument('--forced',action='store_true');ap.add_argument('--stem',default='candidate');a=ap.parse_args();mode='forced' if a.forced else 'normal';holder={}
s=(o/(a.stem+'.py')).read_text();adapter=(H/'attention_adapter.py.inc').read_text().replace("partial_port_rows=ns['_PP_ROWS'],vq8_rows=ns['_VQ8_ROWS'],attention_plan={})","partial_port_rows=ns['_PP_ROWS'],vq8_rows=ns['_VQ8_ROWS'],attention_plan={},r2_rows=ns['_R2_SELECTION'],wr_rows=ns['_WINDOW_SELECTION'])")
setup="\ndef prepare(weights):\n    if '_p5b_sched_init' in globals():globals()['_p5b_sched_init'](torch.device('cuda'))\n    if '_p5b_merge_setup' in globals():globals()['_p5b_merge_setup']()\n    return None\n"
preview=o/(a.stem+'_attention.py');preview.write_text(s.replace('# okc: problem=muse_decode_layer','# okc: problem=muse_attention_decode',1)+adapter+setup)
original=ev.load_module

def loader(name,path):
 m=original(name,path)
 if name=='reference':m.CACHE_CHUNK=8192
 if name=='submission':
  saved={k:getattr(m,k) for k in ('_attn_port_hopper','_fa3_on_device','ATTN_BACKEND')};old=m.prepare
  spec=importlib.util.spec_from_file_location('chain_force',R/'problems/muse_decode_layer/dev/verify_best.py');helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper);helper.install(m,False);m.prepare=old
  if not a.forced:
   for k,v in saved.items():setattr(m,k,v)
  import rebuild_standins,chain_standins
  rebuild_standins.install(m)
  if hasattr(m,'_p5b_merge_setup'):
   def merge_setup():
    from torch.utils.cpp_extension import load_inline
    import hashlib
    src=m._P5B_REDUCE_SRC
    m._P5B_REDUCE=load_inline(name='p5_chain_cand_reduce_'+hashlib.sha256(src.encode()).hexdigest()[:12],cpp_sources='void row_forward(int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);',cuda_sources=src,functions=['row_forward'],extra_cuda_cflags=['-O3'])
   m._p5b_merge_setup=merge_setup
   if hasattr(m,'_p5b_merge_main'):
    def main(q,d,ns,fo,lse,oa,la,meta):
     chain_standins.fa_main(m,q,d,ns,oa,la)
     if ns==1:fo.copy_(oa[0].permute(1,0,2).to(ev.torch.bfloat16))
    m._p5b_merge_main=main
  if hasattr(m,'_P5B_VQ_SOURCES'):
   cache={}
   def vmod(role):
    if role not in cache:cache[role]=chain_standins.native(m,m._P5B_VQ_SOURCES[role])
    native=cache[role]
    class Wrapped:
     def vq8_forward(self,*args):
      args=list(args);args[22]=2
      if role in ('F','G'):args[20]=1;args[21]=args[11]*(args[14]//2)
      return native.vq8_forward(*args)
    return Wrapped()
   m._p5b_vq_module=vmod
  m.prepare(None);holder['module']=m
 return m

ev.load_module=loader
oldtests=ev.tests_of
ev.tests_of=lambda md:oldtests(md)+[dict(batch=b,q_len=l,ctx=80000,layer=layer,heads='tp8',page=1,dist='coherent',spread='agentic' if b>1 else 'fixed') for b in (1,16,32) for l in (1,8) for layer in ('sliding','full')]
def run(problem,side,cases,seed0):
 rows=[]
 for i,c in enumerate(cases):
  d=problem.make_input(c,seed0+i);row=dict(case=ev.describe(c))
  try:
   y=side.call(d);ev.torch.cuda.synchronize();ok,msg=problem.check(d,y);row.update(eager_ok=ok,eager_msg=msg)
   g,y=ev.capture(side.call,d);y.fill_(float('nan'));g.replay();g.replay();ev.torch.cuda.synchronize();ok,msg=problem.check(d,y);row.update(ok=row['eager_ok'] and ok,msg=msg);del g,y
  except Exception as e:row.update(ok=False,msg=type(e).__name__+':'+str(e))
  print('CHAIN_CANDIDATE',mode,row,flush=True);rows.append(row);del d;gc.collect();ev.torch.cuda.empty_cache()
 return rows

ev.run_tests=run
sys.argv=['local_eval.py',str(preview),'--tests-only','--json',str(o/(a.stem+'_'+mode+'_screen.json'))];ev.main()
