"""WG correctness, replay, fixed-work control, and local/forced geometry validation."""
import hashlib,json,linecache,runpy,sys
from pathlib import Path
import torch
DEV=Path(__file__).resolve().parent;ROOT=DEV.parents[3]
src=(ROOT/'prof/lab/moe_nt1_timeline.py').read_text();n={};name='<reuse fixture>'
linecache.cache[name]=(len(src),None,src.splitlines(True),name);exec(compile(src,name,'exec'),n)
c=runpy.run_path(str(DEV/'wg_cell.py'));gen=torch.Generator(device='cuda').manual_seed(34211)
w=n['rg_generate_weights'](5353);w['post_norm_w']=w['post_expert_norm'];old=n['nt1_banks'](w)
st={k:old[k] for k in ('gs13','gs2','pnw','sms')};st['banks']={'lop3':old['bank']};c['wg_bind'](st)
print('GEOM',[(s,c['wg_geometry'](st,32,s,4)) for s in (4,8)],flush=True)
for name in ('w13p','w2p'):
 src=st['banks']['lop3'][name];tiles=src.shape[1]
 back=st['wg_scales'][name].view(32,tiles,8,8,32,2).permute(0,1,2,4,3,5).contiguous().view(32,tiles,4096)
 assert torch.equal(back,src[:,:,32768:]),('lossless scale bytes',name)
checks=0;worst=0.;vsinc=0.
for m in (1,8,9,16,17,24,31,32):
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
   for mode in (4,8):
    for trace in (False,):
     grid=c['wg_geometry'](st,m,mode,4)[0];stamps=torch.zeros((grid,16),device='cuda',dtype=torch.int64) if trace else None
     fn=lambda:c['wg_experts'](d,st,mode,4)
     eager=fn();g=torch.cuda.CUDAGraph()
     with torch.cuda.graph(g):out=fn()
     for rep in range(3):
      out.fill_(float('nan'));g.replay();torch.cuda.synchronize()
      ok,err,row=n['task_check'](out,want)
      rel=float((out.float()-inc.float()).norm()/inc.float().norm().clamp_min(1e-15))
      assert ok and rel<2e-3,(m,seed,label,mode,trace,err,row,rel)
      if label in ('none','zero'):assert not bool(out.any()),(m,label,mode,trace)
      if trace:assert bool((stamps[:,9]>stamps[:,0]).all())
      checks+=1;worst=max(worst,err);vsinc=max(vsinc,rel)
   print('PASS',m,seed,label,flush=True)
# Fresh activations AND routes on captured graphs (the intermediate is read across CTAs within one call).
for m in (8,16,32):
 d=n['fx_make_fixture'](w,m,9971);graphs={}
 for mode in (4,8):
  c['wg_experts'](d,st,mode);g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):out=c['wg_experts'](d,st,mode)
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
# Force the Hopper launch ABI with native local stand-in kernels and capped resident grids.
mod=c['wg_module']();calls=[]
class Standin:
 optin=mod.optin
 def wg_info(self,stages,m):return mod.wg_info(stages,m)
 def wg_run(self,ptrs,m,start,stages,grid,stream):
  assert len(ptrs)==15 and stages in (4,8) and grid<=st['sms']*mod.wg_info(stages,m)[0]
  assert stream==torch.cuda.current_stream().cuda_stream;calls.append((m,stages))
  mod.wg_run(ptrs,m,start,stages,grid,stream)
c['wg_experts'].__globals__['_WG_MODULE'][0]=Standin()
for m in (8,16,32):
 d=n['fx_make_fixture'](w,m,8951)
 for mode in (4,8):
  got=c['wg_experts'](d,st,mode);torch.cuda.synchronize();ok,_,_=n['task_check'](got,n['nt1_fp32'](d,w));assert ok
c['wg_experts'].__globals__['_WG_MODULE'][0]=mod
result=dict(checks=checks,forced_glue=len(calls),worst_task_l2=worst,max_rel_vs_inc=vsinc,
 cell_sha256=hashlib.sha256((DEV/'wg_cell.py').read_bytes()).hexdigest())
(DEV/'wg_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True)
