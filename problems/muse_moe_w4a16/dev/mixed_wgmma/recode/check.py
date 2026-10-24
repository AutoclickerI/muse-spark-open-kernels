"""Exhaustive finite-scale bank equivalence, then fresh graph and allocator-lifetime gates."""
import ast,json,sys
from pathlib import Path
import torch
H=Path(__file__).resolve().parent;P=H.parent;R=P.parents[3];sys.path.insert(0,str(P));sys.path.insert(0,str(H));sys.path.insert(0,str(R/'prof/lab'))
from recoded import rc_prepare,rc_plan,rc_repack,RC_SIGNS,RC_ZEROS,n1_prepare,n1_call
from nt1_pack import n1_encode_scales
from engine import mw_repack_planes
from validate import make_weights
from moe_w2fix_m256 import dc_oracles,dc_fixtures,rg_generate_weights

def exhaustive():
 sf=torch.arange(256,device='cuda',dtype=torch.uint8).view(1,1,256).expand(1,16,256).clone();sf[(sf&127)==127]=0
 q=(torch.arange(16,device='cuda',dtype=torch.uint8)*17).view(1,16,1).expand(1,16,2048).contiguous();u,neg,zero=n1_encode_scales(sf.view(torch.float8_e4m3fn))
 expect_q=torch.where(zero[...,None],0,q.reshape(1,16,256,8)^(neg.to(torch.uint8)[...,None]*136)).reshape_as(q)
 raw=mw_repack_planes(q);qr=torch.empty_like(raw);sp=sf.reshape(1,1,2,8,128,2).permute(0,1,4,3,5,2).contiguous();sr=torch.empty_like(sp)
 rc_repack[((raw.numel()//2+255)//256,)](raw,sp,qr,sr,raw.numel()//2,RC_SIGNS,RC_ZEROS)
 wantq=mw_repack_planes(expect_q);wants=u.reshape(1,1,2,8,128,2).permute(0,1,4,3,5,2).contiguous()
 assert torch.equal(qr,wantq) and torch.equal(sr,wants)
 code=torch.arange(16,device='cuda').view(1,16,1);lut=torch.tensor([0.,.5,1.,1.5,2.,3.,4.,6.],device='cuda');orig=torch.where(code&8!=0,-lut[code&7],lut[code&7])*sf.view(torch.float8_e4m3fn).float()
 nc=(code^((neg.long())*8));nc=torch.where(zero,0,nc);v=torch.exp2((u.int()>>3).float()-21)*(1+(u.int()&7).float()/8);got=torch.where(nc&8!=0,-lut[nc&7],lut[nc&7])*v
 assert torch.equal(got,orig)
 print('EXACT_RECODE',254*16*16,'finite scale/code/slot values and packed bits',flush=True)

def main():
 exhaustive();torch.manual_seed(7741);w=make_weights();w['w13_gs_gate']=w['w13_gs'].clone();w['w13_gs_up']=w['w13_gs']*torch.exp(.55*torch.randn(32,device='cuda'));banks=rc_prepare(w);ds=[]
 for mode in ('balanced','random','hot','empty','single','random','empty','balanced'):
  ids=torch.full((256,8),255,device='cuda',dtype=torch.int32)
  if mode=='balanced':ids[:,0]=torch.arange(256,device='cuda')%32
  if mode=='hot':ids[:,0]=0
  if mode=='single':ids[7,2]=31
  if mode=='random':ids=torch.rand((256,256),device='cuda').topk(8,-1).indices.int()
  ds.append(dict(hidden_states=torch.randn((256,4096),device='cuda').bfloat16(),topk_ids=ids,topk_weights=torch.softmax(torch.randn((256,8),device='cuda'),-1),config={'local_start':0}))
 wants=dc_oracles(ds,w);num=0;worst=0.;rowmax=0.
 for which in range(2):
  for grid in (528,1056,2112):
   for bank in banks:
    static={k:v.clone() if torch.is_tensor(v) else v for k,v in ds[0].items()};plan=rc_plan(static,bank,which,grid);steps=plan['steps'];del plan
    # Force allocator churn after the plan container dies, before eager or captured launches.
    churn=[torch.empty((256,4096),device='cuda').fill_(77) for _ in range(4)]
    for f in steps:got=f()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
     for f in steps:got=f()
    for d,want in zip(ds,wants):
     for key in ('hidden_states','topk_ids','topk_weights'):static[key].copy_(d[key])
     graph.replay();torch.cuda.synchronize();diff=got.float()-want.float();e=float(diff.norm()/want.float().norm().clamp_min(1e-12));den=want.float().norm(dim=-1);row=float((diff.norm(dim=-1)/den.clamp_min(den.mean()*.1+1e-12)).max());assert bool(torch.isfinite(got).all()) and e<.01 and row<.05,(which,grid,bank['encoded'],e,row)
     num+=1;worst=max(worst,e);rowmax=max(rowmax,row)
    print('RECODE_GRAPH',which,grid,bank['encoded'],worst,flush=True)
 for key in ('w13','w2'):
  raw=torch.randint(256,w[key+'_sf'].shape,device='cuda',dtype=torch.uint8);raw[(raw&127)==127]=0;w[key+'_sf']=raw.view(torch.float8_e4m3fn)
 banks=rc_prepare(w);wants=dc_oracles(ds[:3],w)
 for bank in banks:
  for which in range(2):
   for d,want in zip(ds[:3],wants):
    dd=dict(d,topk_ids=d['topk_ids']+64,config={'local_start':64})
    for f in rc_plan(dd,bank,which,1056)['steps']:got=f()
    torch.cuda.synchronize();e=float((got.float()-want.float()).norm()/want.float().norm());assert bool(torch.isfinite(got).all()) and e<.01,(e,bank['encoded']);num+=1;worst=max(worst,e)
 # Stock NT1 uses the actual model's shared W13 per-expert global scale.
 w=rg_generate_weights(5353,device='cuda');w['post_norm_w']=w['post_expert_norm'];w['w13_gs_gate']=w['w13_gs'];w['w13_gs_up']=w['w13_gs'];fixtures=dc_fixtures(w,256,True);st=n1_prepare(w);wants=dc_oracles(fixtures,w)
 for d,want in zip(fixtures,wants):
  graph=torch.cuda.CUDAGraph();got=n1_call(d,st)
  with torch.cuda.graph(graph):got=n1_call(d,st)
  graph.replay();torch.cuda.synchronize();e=float((got.float()-want.float()).norm()/want.float().norm());assert bool(torch.isfinite(got).all()) and e<.01,e;num+=1;worst=max(worst,e)
 (H/'agreement.json').write_text(json.dumps(dict(exact_cases=65024,graph_and_scale_checks=num,max_L2=worst,max_row=rowmax,allocator_churn=True,independent_gate_up_GS=True),indent=2)+'\n');print('RECODE_ALL_PASS',num,worst,rowmax,flush=True)

if __name__=='__main__':main()
