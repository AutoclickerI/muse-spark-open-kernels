"""Decode-once correctness, replay, fixed-work control, and local/forced geometry validation."""
import hashlib,json,linecache,runpy,sys
from pathlib import Path
import torch
DEV=Path(__file__).resolve().parent;ROOT=DEV.parents[3]
src=(ROOT/'prof/lab/moe_nt1_timeline.py').read_text();n={};name='<reuse fixture>'
linecache.cache[name]=(len(src),None,src.splitlines(True),name);exec(compile(src,name,'exec'),n)
c=runpy.run_path(str(DEV/'reused_cell.py'));gen=torch.Generator(device='cuda').manual_seed(34211)
w=n['rg_generate_weights'](5353);w['post_norm_w']=w['post_expert_norm'];old=n['nt1_banks'](w)
st={k:old[k] for k in ('gs13','gs2','pnw','sms')};st['banks']={'lop3':old['bank']};c['reused_bind'](st)
native=c['reused_module']()
class TraceStageStandin:
 optin=native.optin
 def reused_info(self,mode,trace,stages,m):return native.reused_info(mode,trace,1 if mode==2 else stages,m)
 def reused_run(self,ptrs,m,start,stages,grid,stream,mode,trace):return native.reused_run(ptrs,m,start,1 if mode==2 else stages,grid,stream,mode,trace)
c['reused_experts'].__globals__['_REUSED_MODULE'][0]=TraceStageStandin()
print('LOCAL: mode2 uses1 raw stage to fit local shared; mode0 uses native2 stages',flush=True)
print('GEOM',[(r,t,c['reused_geometry'](st,32,r,t)) for r in (0,2) for t in (0,1)],flush=True)
checks=0;worst=0.;vsinc=0.
for m in (1,8,9,16,17,24,25,31,32):
 for seed in (0,1):
  d=n['fx_make_fixture'](w,m,777+97*seed+m)
  fixtures={'real':d,'zero':dict(d,hidden_states=torch.zeros_like(d['hidden_states']))}
  fixtures['none']=dict(d,topk_ids=torch.full_like(d['topk_ids'],-1))
  one=torch.full_like(d['topk_ids'],-1);one[:,0]=31
  fixtures['one']=dict(d,topk_ids=one)
  fixtures['all8']=dict(d,topk_ids=torch.arange(8,device='cuda',dtype=torch.int32).expand(m,8).contiguous())
  fixtures['uniform']=dict(d,topk_ids=torch.stack([torch.randperm(32,device='cuda',generator=gen)[:8] for _ in range(m)]).int())
  for label,d in fixtures.items():
   want=n['nt1_fp32'](d,w);inc=n['nt1_call'](d,old,'INC');torch.cuda.synchronize()
   for mode in (0,2):
    for trace in (False,True):
     grid=c['reused_geometry'](st,m,mode,trace)[0];stamps=torch.zeros((grid,1040),device='cuda',dtype=torch.int64) if trace else None
     fn=lambda:c['reused_experts'](d,st,mode,trace,stamps)
     eager=fn();g=torch.cuda.CUDAGraph()
     with torch.cuda.graph(g):out=fn()
     for rep in range(3):
      out.fill_(float('nan'));g.replay();torch.cuda.synchronize()
      ok,err,row=n['task_check'](out,want)
      rel=float((out.float()-inc.float()).norm()/inc.float().norm().clamp_min(1e-15))
      assert ok and rel<2e-3,(m,seed,label,mode,trace,err,row,rel)
      if label in ('none','zero'):assert not bool(out.any()),(m,label,mode,trace)
      if trace:assert bool((stamps[:,5]>stamps[:,0]).all())
      checks+=1;worst=max(worst,err);vsinc=max(vsinc,rel)
   print('PASS',m,seed,label,flush=True)
# Fresh activations AND routes on captured graphs (the intermediate is read across CTAs within one call).
for m in (8,16,32):
 d=n['fx_make_fixture'](w,m,9971);graphs={}
 for mode in (0,2):
  c['reused_experts'](d,st,mode);g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):out=c['reused_experts'](d,st,mode)
  graphs[mode]=(g,out)
 for draw in range(4):
  fresh=n['fx_make_fixture'](w,m,9981+31*draw)
  for k in ('hidden_states','topk_ids','topk_weights'):d[k].copy_(fresh[k])
  inc=n['nt1_call'](d,old,'INC');want=n['nt1_fp32'](d,w)
  for mode,(g,out) in graphs.items():
   g.replay();torch.cuda.synchronize();ok,err,row=n['task_check'](out,want)
   rel=float((out.float()-inc.float()).norm()/inc.float().norm().clamp_min(1e-15))
   assert ok and rel<2e-3,(m,draw,mode,'fresh',err,row,rel)
   checks+=1;worst=max(worst,err);vsinc=max(vsinc,rel)
# Force the sm90 geometry/glue while the actual native kernel remains a portable sm120 stand-in.
# The runtime always takes a CUDA stream, capped residency, pointer ABI and PDL launch in both architectures.
mod=c['reused_module']();calls=[]
class Standin:
 optin=mod.optin
 def reused_info(self,mode,trace,stages,m):
  info=mod.reused_info(mode,trace,stages,m);assert stages==2;return info
 def reused_run(self,ptrs,m,start,stages,grid,stream,mode,trace):
  assert len(ptrs)==14 and stages==2 and grid<=st['sms']*mod.reused_info(mode,trace,stages,m)[0]
  assert stream==torch.cuda.current_stream().cuda_stream;calls.append((m,mode,trace))
  mod.reused_run(ptrs,m,start,stages,grid,stream,mode,trace)
c['reused_experts'].__globals__['_REUSED_MODULE'][0]=Standin()
for m in (8,16,32):
 d=n['fx_make_fixture'](w,m,8951)
 for mode in (0,2):
  got=c['reused_experts'](d,st,mode);torch.cuda.synchronize();ok,_,_=n['task_check'](got,n['nt1_fp32'](d,w));assert ok
c['reused_experts'].__globals__['_REUSED_MODULE'][0]=mod
result=dict(mode2_local_stages=1,checks=checks,forced_glue=len(calls),worst_task_l2=worst,max_rel_vs_inc=vsinc,
 cell_sha256=hashlib.sha256((DEV/'reused_cell.py').read_bytes()).hexdigest())
(DEV/'reused_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True)
