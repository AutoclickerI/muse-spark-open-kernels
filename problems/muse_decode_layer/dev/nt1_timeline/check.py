"""Local admission of the small-M lab: every variant arm's math (TASK.md gate and
closeness to INC) on real-router fixtures and edge routes, fresh graph replays, and forced Hopper glue."""
from pathlib import Path
import hashlib,json,runpy,sys
import torch
ROOT=Path(__file__).resolve().parents[4];DEV=Path(__file__).resolve().parent
n=runpy.run_path(str(ROOT/'prof/lab/moe_nt1_timeline.py'));g=n['var_call'].__globals__
F=dict((k,n[k]) for k in ('F_Z','F_SLOT','F_ZOUT','F_RSQ','F_NREG','F_PDL','F_LAST','F_TRACE','F_DYN'))
gen=torch.Generator(device='cuda').manual_seed(67182)

w=n['rg_generate_weights'](5353);w['post_norm_w']=w['post_expert_norm'];state=n['nt1_banks'](w)
arms=n['SMALLM_ARMS'];checks=0;worst={};vsinc={}
# every E arm plus every R arm's flags on the edge fixtures; the lab's ECAP is F0 math on capped routes
ecks={m:{**{k:v for k,v in a['E'].items() if v is not None},**{'E'+k:v for k,v in a['R'].items() if v is not None}} for m,a in arms.items()}
TRACED={1:(0,95,1119),16:(0,93,69),32:(0,93,69)}
def rel(a,b):return float((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-12))
def edge_fixtures(m,seed):
 d=n['fx_make_fixture'](w,m,9000+m+97*seed);yield 'real',d
 if m==1:
  for name,ids in (('none',[-1]*8),('all8',list(range(3,27,3))),('one',[-1,-1,-1,-1,-1,-1,-1,31]),('two_hi',[31,-1,0,-1,-1,-1,-1,-1])):
   e=dict(d);e['topk_ids']=torch.tensor([ids],device='cuda',dtype=torch.int32);yield name,e
 else:
  e=dict(d);ids=d['topk_ids'].clone();ids[::2]=-1;e['topk_ids']=ids;yield 'halfremote',e
  e=dict(d);e['topk_ids']=torch.randint(0,32,d['topk_ids'].shape,device='cuda',generator=gen,dtype=torch.int32)
  # distinct ids per token
  e['topk_ids']=torch.stack([torch.randperm(32,device='cuda',generator=gen)[:8] for _ in range(m)]).int();yield 'uniform',e
  e=dict(d);e['hidden_states']=torch.zeros_like(d['hidden_states']);yield 'zero_x',e
for seed in (0,1):
 for m in (1,8,16,32):
  for name,d in edge_fixtures(m,seed):
   want=n['nt1_fp32'](d,w);inc=n['nt1_call'](d,state,'INC');torch.cuda.synchronize()
   for label,flags in ecks[m].items():
    for trace in ((False,True) if flags in TRACED.get(m,()) else (False,)):
     f=flags|(F['F_TRACE'] if trace else 0)
     grid=min(264,state['sms']*n['var_module']().resident_blocks(1,2,m))
     stamps=torch.zeros((grid,64),device='cuda',dtype=torch.int64) if trace else None
     fn=lambda:n['var_call'](d,state,f,stamps=stamps)
     eager=fn();graph=torch.cuda.CUDAGraph()
     with torch.cuda.graph(graph):out=fn()
     for rep in range(3):
      out.fill_(float('nan'));graph.replay();torch.cuda.synchronize()
      if name=='zero_x' or name=='none':
       assert torch.equal(out,torch.zeros_like(out)),(label,m,name)
       continue
      ok,err,row=n['task_check'](out,want);assert ok,(label,m,name,err,row)
      worst[label]=max(worst.get(label,0.),err)
      r=rel(out,inc);vsinc[label]=max(vsinc.get(label,0.),r);assert r<2e-3,(label,m,name,'vs INC',r)
      checks+=1
     if trace:assert int((stamps[:,0]>0).sum())==grid and int((stamps[:,9]>0).sum())==grid,(label,'stamps')
 # router arms on the real router fixture, incl. PDL, with fresh inputs on the same graph
 for m in (1,8,16,32):
  pr,rs=n['routed_fixture'](w,m,9000+seed);base=n['routed_experts'](pr,rs,state,'INC');torch.cuda.synchronize()
  for label,flags in arms[m]['R'].items():
   if flags is None:continue
   fn=lambda:n['routed_var'](pr,rs,state,flags)
   fn();graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(graph):out=fn()
   for draw in range(3):
    if draw:
     pr2,rs2=n['routed_fixture'](w,m,10000+seed+draw);pr[0].copy_(pr2[0]);rs['start'].copy_(rs2['start'])
    graph.replay();ref=n['routed_experts'](pr,rs,state,'INC');torch.cuda.synchronize()
    assert torch.equal(out[2],ref[2]),(label,m,'routing differs from INC router')
    d=dict(hidden_states=ref[1],topk_ids=ref[4],topk_weights=ref[3],config={'local_start':0})
    ok,err,row=n['task_check'](out[0],n['nt1_fp32'](d,w));assert ok,(label,m,err,row)
    r=rel(out[0],ref[0]);vsinc[label]=max(vsinc.get(label,0.),r);assert r<2e-3,(label,m,'vs INC',r)
    worst[label]=max(worst.get(label,0.),err);checks+=1
   pr,rs=n['routed_fixture'](w,m,9000+seed)
print('MATH',checks,'worst',{k:round(v,5) for k,v in worst.items()},'vsINC',{k:f'{v:.2e}' for k,v in vsinc.items()},flush=True)
# Forced Hopper glue: the H200 path differs only in -gencode and the resident grid (2 CTAs/SM): run the
# variant module through the Hopper branch with a portable stand-in and a 2x grid request clipped by residency.
class Standin:
 def __init__(self,real):self.real=real;self.smem_optin=real.smem_optin
 def smem_needed(self,*a):return self.real.smem_needed(*a)
 def resident_blocks(self,*a):return self.real.resident_blocks(*a)
 def run(self,ptrs,m,start,stages,grid,stream,flags):
  assert stages==2 and grid<=state['sms']*self.real.resident_blocks(1,2,m) and stream==torch.cuda.current_stream().cuda_stream
  assert len(ptrs)==(14 if flags&F['F_TRACE'] else 13)
  self.real.run(ptrs,m,start,stages,grid,stream,flags)
real=n['var_module']();g['NT1_HOPPER']=True;g['VAR_MODULE'][0]=Standin(real);forced=0
for m in (1,8):
 d=n['fx_make_fixture'](w,m,123);inc=n['nt1_call'](d,state,'INC')
 for label,flags in ecks[m].items():
  out=n['var_call'](d,state,flags);torch.cuda.synchronize();assert rel(out,inc)<2e-3,(label,'forced');forced+=1
g['VAR_MODULE'][0]=real;g['NT1_HOPPER']=False
result=dict(math_checks=checks,forced_hopper_glue=forced,worst_task_l2=worst,max_rel_vs_inc=vsinc,
  architecture=torch.cuda.get_device_capability(),payload_sha256=hashlib.sha256((ROOT/'prof/lab/moe_nt1_timeline.py').read_bytes()).hexdigest())
(DEV/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
