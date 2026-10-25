"""Native math, descriptor ABI and forced Hopper launch-glue admission."""
from pathlib import Path
import hashlib,json,runpy,sys
import torch
from torch.utils.cpp_extension import load_inline
ROOT=Path(__file__).resolve().parents[4];DEV=Path(__file__).resolve().parent
n=runpy.run_path(str(ROOT/'prof/lab/moe_nt1_timeline.py'))
g=n['nt1_module'].__globals__
gen=torch.Generator(device='cuda').manual_seed(67182)
# Compile the exact native v1 writer as the independent metadata oracle.
header=(DEV.parent/'nt1_branches/route_meta.h').read_text().replace('#pragma once','')
source='#include <cuda_runtime.h>\n'+header+'\n'+(DEV.parent/'nt1_branches/route_meta.cu.inc').read_text()+'''
void golden(int64_t ids,int64_t M,int64_t start,int64_t out,int64_t stream){make_route_meta<<<1,256,0,(cudaStream_t)stream>>>((int*)ids,M,start,(MoERouteMeta*)out);}
'''
mod=load_inline(name='timeline_abi_'+hashlib.sha256(source.encode()).hexdigest()[:12],cpp_sources='#include <pybind11/pybind11.h>\n#include <stdint.h>\nvoid golden(int64_t,int64_t,int64_t,int64_t,int64_t);',cuda_sources=source,functions=['golden'],with_pytorch_error_handling=False,no_implicit_headers=True,verbose=False)
abi=0;flips=0;worstx=0.
for m in (1,2,7,8,16,31,32):
 for splits in (1,2,8,16):
  p=torch.randn((splits,m,4608),device='cuda',generator=gen);rs=dict(bias=torch.randn(256,device='cuda',generator=gen)*.055,pre=1+torch.randn(4096,device='cuda',generator=gen)*.1,start=torch.zeros(1,device='cuda',dtype=torch.int32),arrive=torch.zeros(1,device='cuda',dtype=torch.int32))
  n['route_call']((p,splits),rs,True)
  graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph):b=n['route_call']((p,splits),rs,True)
  for draw,start in enumerate((0,96,224)):
   p.normal_(generator=gen);rs['start'].fill_(start)
   if draw==1:p.zero_()
   if draw==2:rs['bias'].fill_(-1);rs['bias'][(start+64)%256:(start+64)%256+8]=10
   b[-1].fill_(-123456);graph.replay();a=n['route_call']((p,splits),rs,False)
   want=torch.zeros_like(b[-1]);mod.golden(b[2].data_ptr(),m,start,want.data_ptr(),torch.cuda.current_stream().cuda_stream)
   torch.cuda.synchronize()
   assert torch.equal(b[-1],want),(m,splits,start,'metadata',torch.where(b[-1]!=want)[0].tolist())
   assert not int(rs['arrive']),('arrive counter',m,splits)
   assert torch.equal(a[1],b[1]) and torch.equal(a[3],b[3]),(m,splits,'same router changed')
   assert torch.equal(a[0],b[0]) and torch.equal(a[2],b[2]),(m,splits,'norm/weights')
   x,score,bias=n['route_reference']((p,splits),rs)
   flips+=n['route_gate'](b[1],bias,start)
   err=n['nt1_error'](b[0],x)[0];worstx=max(worstx,err);assert err<.004
   abi+=1
print('ABI',abi,'flips',flips,'norm',worstx,flush=True)
w=n['rg_generate_weights'](5353);w['post_norm_w']=w['post_expert_norm'];state=n['nt1_banks'](w)
checks=0;maxerr=0.;maxrow=0.;same=True
for seed in (0,1):
 for m in (1,8,16,32):
  pr,rs=n['routed_fixture'](w,m,9000+seed);a=n['routed_experts'](pr,rs,state,'INC',False)
  d=dict(hidden_states=a[1],topk_ids=a[4],topk_weights=a[3],config={'local_start':0})
  want=n['nt1_fp32'](d,w)
  for kind,emit in (('INC',False),('Z',False),('META',True),('METAZ',True)):
   fn=lambda:n['routed_experts'](pr,rs,state,kind,emit)
   fn();graph=torch.cuda.CUDAGraph()
   with torch.cuda.graph(graph):out=fn()
   graph.replay();torch.cuda.synchronize();err,row=n['nt1_error'](out[0],want);maxerr=max(maxerr,err);maxrow=max(maxrow,row)
   assert err<.01 and row<.05,(kind,m,err,row)
   assert torch.equal(out[0],a[0]),(kind,m,'incumbent math')
   # Same graph, new input values and routes; the old zero state cannot survive.
   pr2,rs2=n['routed_fixture'](w,m,10000+seed);pr[0].copy_(pr2[0]);rs['start'].copy_(rs2['start'])
   graph.replay();fresh=n['routed_experts'](pr,rs,state,'INC',False);torch.cuda.synchronize()
   assert torch.equal(out[0],fresh[0]),(kind,m,'fresh replay')
   pr,rs=n['routed_fixture'](w,m,9000+seed)
   checks+=1
# Force each Hopper branch in the Python glue with an ABI-accurate portable stand-in.
class Standin:
 def __init__(self,real):self.real=real;self.smem_optin=real.smem_optin
 def smem_needed(self,*a):return self.real.smem_needed(*a)
 def resident_blocks(self,*a):return self.real.resident_blocks(*a)
 def run(self,ptrs,m,start,stages,grid,stream):
  assert st_agree==stages==2 and grid<=state['sms'] and stream==torch.cuda.current_stream().cuda_stream
  assert len(ptrs)==(14 if active in ('META','METAZ','TRACE','TRACEZ') else 13)
  self.real.run(ptrs,m,start,stages,grid,stream)
for kind in n['NT1_SOURCES']:n['nt1_module'](kind)
old=g['NT1_MODULES'];g['NT1_HOPPER']=True;g['NT1_MODULES']={k:Standin(v) for k,v in old.items()};st_agree=2
for active in n['NT1_SOURCES']:
 pr,rs=n['routed_fixture'](w,8,123);x,ids,tw,lids,meta=n['route_call'](pr,rs,True)
 d=dict(hidden_states=x,topk_ids=lids,topk_weights=tw,config={'local_start':0})
 out=n['nt1_call'](d,state,active,metadata=meta);torch.cuda.synchronize()
 assert torch.equal(out,n['nt1_call'](d,state,'INC'))
g['NT1_MODULES']=old;g['NT1_HOPPER']=False
result=dict(abi_checks=abi,math_graph_checks=checks,forced_hopper_glue=list(n['NT1_SOURCES']),same_incumbent=True,max_fp32_l2=maxerr,max_fp32_row=maxrow,router_flips=flips,route_norm_l2=worstx,architecture=torch.cuda.get_device_capability(),payload_sha256=hashlib.sha256((ROOT/'prof/lab/moe_nt1_timeline.py').read_bytes()).hexdigest())
(DEV/'validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
