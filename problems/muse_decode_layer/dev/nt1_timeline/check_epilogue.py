"""Validate changed reduction ownership, delayed atomics, producer handoff and replay epochs."""
import hashlib,json,linecache,runpy
from pathlib import Path
import torch
DEV=Path(__file__).resolve().parent;ROOT=DEV.parents[3]
src=(ROOT/'prof/lab/moe_nt1_reuse.py').read_text();name='<epilogue fixtures>';linecache.cache[name]=(len(src),None,src.splitlines(True),name)
n={'__name__':'epilogue_check'};exec(compile(src,name,'exec'),n)
w=n['rg_generate_weights'](5353);w['post_norm_w']=w['post_expert_norm'];old=n['nt1_banks'](w)
st={k:old[k] for k in ('gs13','gs2','pnw','sms')};st['banks']={'lop3':old['bank']}
cells={tag:runpy.run_path(str(DEV/(tag+'_cell.py'))) for tag in ('epib','epiv','epip','epin')}
for tag,c in cells.items():
 c[tag+'_bind'](st,{1:95,8:69,16:325});print('GEOM',tag,[c[tag+'_module']().epi_info(f,2,32) for f in (69,95,325)],flush=True)
result={tag:dict(checks=0,worst_task_l2=0.,max_rel_vs_inc=0.) for tag in cells};gen=torch.Generator(device='cuda').manual_seed(5642)
for m in (1,8,9,16,17,24,25,31,32):
 for seed in (0,1):
  d=n['fx_make_fixture'](w,m,999+m+97*seed);fixtures={'real':d,'zero':dict(d,hidden_states=torch.zeros_like(d['hidden_states']))}
  fixtures['none']=dict(d,topk_ids=torch.full_like(d['topk_ids'],-1))
  one=torch.full_like(d['topk_ids'],-1);one[:,0]=31;fixtures['one']=dict(d,topk_ids=one)
  fixtures['all8']=dict(d,topk_ids=torch.arange(8,device='cuda',dtype=torch.int32).expand(m,8).contiguous())
  fixtures['uniform']=dict(d,topk_ids=torch.stack([torch.randperm(32,device='cuda',generator=gen)[:8] for _ in range(m)]).int())
  for label,d in fixtures.items():
   want=n['nt1_fp32'](d,w);inc=n['nt1_call'](d,old,'INC')
   for tag,c in cells.items():
    for f in ((95,69,325) if m==1 else (69,325)):
     grid=min(264,st['sms']*c[tag+'_module']().epi_info(f,2,m)[0]);stamps=torch.zeros((grid,64),device='cuda',dtype=torch.int64) if f&256 else None
     fn=lambda:c[tag+'_experts'](d,st,f,stamps=stamps)
     fn();g=torch.cuda.CUDAGraph()
     with torch.cuda.graph(g):out=fn()
     for rep in range(2):
      out.fill_(float('nan'));g.replay();torch.cuda.synchronize();ok,err,row=n['task_check'](out,want)
      rel=float((out.float()-inc.float()).norm()/inc.float().norm().clamp_min(1e-15));assert ok and rel<2e-3,(tag,m,seed,label,f,err,row,rel)
      if label in ('zero','none'):assert not bool(out.any())
      if stamps is not None:assert bool((stamps[:,8]>stamps[:,0]).all())
      r=result[tag];r['checks']+=1;r['worst_task_l2']=max(r['worst_task_l2'],err);r['max_rel_vs_inc']=max(r['max_rel_vs_inc'],rel)
   print('PASS',m,seed,label,flush=True)
# Captures must consume changed routes, activations and expert global starts, not stale sums or norm ownership.
for start in (0,96,224):
 for m in (8,16,32):
  d=n['fx_make_fixture'](w,m,9961);d['config']=dict(d['config'],local_start=start);d['topk_ids'].add_(start);graphs={}
  for tag,c in cells.items():
   for f in (69,325):
    grid=min(264,st['sms']*c[tag+'_module']().epi_info(f,2,m)[0]);stamps=torch.zeros((grid,64),device='cuda',dtype=torch.int64) if f&256 else None
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):out=c[tag+'_experts'](d,st,f,stamps=stamps)
    graphs[tag,f]=(g,out)
  for draw in range(4):
   fresh=n['fx_make_fixture'](w,m,9971+draw*31);fresh['topk_ids'].add_(start)
   for k in ('hidden_states','topk_ids','topk_weights'):d[k].copy_(fresh[k])
   want=n['nt1_fp32'](d,w);inc=n['nt1_call'](d,old,'INC')
   for (tag,f),(g,out) in graphs.items():
    g.replay();torch.cuda.synchronize();ok,err,row=n['task_check'](out,want);rel=float((out.float()-inc.float()).norm()/inc.float().norm().clamp_min(1e-15))
    assert ok and rel<2e-3,(tag,m,start,draw,f,err,row,rel);result[tag]['checks']+=1
# Force the same pointer/stream/PDL ABI through the local native stand-in and cap by actual residency.
for tag,c in cells.items():
 native=c[tag+'_module']();calls=[]
 class Standin:
  smem_optin=native.smem_optin
  def __getattr__(self,k):return getattr(native,k)
  def run(self,ptrs,m,start,stages,grid,stream,flags):
   assert len(ptrs)==(14 if flags&256 else 13) and stages==2
   assert grid<=st['sms']*native.epi_info(flags,stages,m)[0] and stream==torch.cuda.current_stream().cuda_stream
   calls.append((m,flags));native.run(ptrs,m,start,stages,grid,stream,flags)
 c[tag+'_experts'].__globals__['_'+tag.upper()+'_MODULE'][0]=Standin()
 for m in (8,16,32):
  d=n['fx_make_fixture'](w,m,9981)
  for f in (69,325):
   grid=min(264,st['sms']*native.epi_info(f,2,m)[0]);ts=torch.zeros((grid,64),device='cuda',dtype=torch.int64) if f&256 else None
   out=c[tag+'_experts'](d,st,f,stamps=ts);torch.cuda.synchronize();assert n['task_check'](out,n['nt1_fp32'](d,w))[0]
 c[tag+'_experts'].__globals__['_'+tag.upper()+'_MODULE'][0]=native
 result[tag].update(forced_glue=len(calls),cell_sha256=hashlib.sha256((DEV/(tag+'_cell.py')).read_bytes()).hexdigest())
(DEV/'epilogue_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True)
