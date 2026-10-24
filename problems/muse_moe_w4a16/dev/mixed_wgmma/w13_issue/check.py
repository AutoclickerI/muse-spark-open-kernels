"""Force the pointer launch glue, shared layout and new vector loaders through real math."""
import sys,json
from pathlib import Path
import torch
H=Path(__file__).resolve().parent;P=H.parent;R=P.parents[3]
sys.path.insert(0,str(P));sys.path.insert(0,str(P/'recode'));sys.path.insert(0,str(H));sys.path.insert(0,str(R/'prof/lab'))
from adapter import wi_prepare,wi_plan
from validate import make_weights
from moe_w2fix_m256 import dc_oracles

def main():
 torch.manual_seed(5544);w=make_weights();w['w13_gs_gate']=w['w13_gs'].clone();w['w13_gs_up']=w['w13_gs']*torch.exp(.55*torch.randn(32,device='cuda'));bank=wi_prepare(w);ds=[]
 for mode in ('balanced','random','hot','empty','single','random','empty','balanced'):
  ids=torch.full((256,8),255,device='cuda',dtype=torch.int32)
  if mode=='balanced':ids[:,0]=torch.arange(256,device='cuda')%32
  if mode=='hot':ids[:,0]=0
  if mode=='single':ids[7,2]=31
  if mode=='random':ids=torch.rand((256,256),device='cuda').topk(8,-1).indices.int()
  ds.append(dict(hidden_states=torch.randn((256,4096),device='cuda').bfloat16(),topk_ids=ids,topk_weights=torch.softmax(torch.randn((256,8),device='cuda'),-1),config={'local_start':0}))
 wants=dc_oracles(ds,w);num=0;worst=0.;rowmax=0.
 for policy in (-1,0,1,2,3,4,5,6,7,8,9):
  static={k:v.clone() if torch.is_tensor(v) else v for k,v in ds[0].items()};plan=wi_plan(static,bank,policy);steps=plan['steps'];del plan
  churn=[torch.empty((256,4096),device='cuda').fill_(99) for _ in range(4)]
  for f in steps:got=f()
  graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph):
   for f in steps:got=f()
  for d,want in zip(ds,wants):
   for k in ('hidden_states','topk_ids','topk_weights'):static[k].copy_(d[k])
   graph.replay();torch.cuda.synchronize();diff=got.float()-want.float();err=float(diff.norm()/want.float().norm().clamp_min(1e-12));den=want.float().norm(dim=-1);row=float((diff.norm(dim=-1)/den.clamp_min(den.mean()*.1+1e-12)).max());assert bool(torch.isfinite(got).all()) and err<.01 and row<.05,(policy,err,row);num+=1;worst=max(worst,err);rowmax=max(rowmax,row)
  print('ISSUE_POLICY_PASS',policy,worst,rowmax,flush=True)
 for k in ('w13','w2'):
  raw=torch.randint(256,w[k+'_sf'].shape,device='cuda',dtype=torch.uint8);raw[(raw&127)==127]=0;w[k+'_sf']=raw.view(torch.float8_e4m3fn)
 bank=wi_prepare(w);wants=dc_oracles(ds[:3],w)
 for policy in (-1,0,1,2,3,4,5,6,7,8,9):
  for d,want in zip(ds[:3],wants):
   dd=dict(d,topk_ids=d['topk_ids']+64,config={'local_start':64})
   for f in wi_plan(dd,bank,policy)['steps']:got=f()
   torch.cuda.synchronize();err=float((got.float()-want.float()).norm()/want.float().norm());assert bool(torch.isfinite(got).all()) and err<.01,(policy,err);num+=1;worst=max(worst,err)
 # Nearby shapes use the same grid, vector predicates and descriptor layout.
 for m in (128,192,257):
  ids=torch.rand((m,256),device='cuda').topk(8,-1).indices.int();d=dict(hidden_states=torch.randn((m,4096),device='cuda').bfloat16(),topk_ids=ids,topk_weights=torch.softmax(torch.randn((m,8),device='cuda'),-1),config={'local_start':0});want=dc_oracles([d],w)[0]
  for policy in (-1,0,1,2,3,4,5,6,7,8,9):
   for f in wi_plan(d,bank,policy)['steps']:got=f()
   torch.cuda.synchronize();err=float((got.float()-want.float()).norm()/want.float().norm());assert bool(torch.isfinite(got).all()) and err<.01,(m,policy,err);num+=1;worst=max(worst,err)
 (H/'agreement.json').write_text(json.dumps(dict(checks=num,max_L2=worst,max_row=rowmax,allocator_churn=True,nearby_shapes=[128,192,257]),indent=2)+'\n');print('ISSUE_ALL_PASS',num,worst,rowmax,flush=True)

if __name__=='__main__':main()
