"""Independent refreshed graph gates and W2 phase clocks, with Hopper address emulation."""
import ast,json,sys,statistics
from pathlib import Path
import torch
from validate import make_weights
from w2fix import fx_prepare,fx_work,fx_plan,FX_CONFIGS,fx_profile
H=Path(__file__).resolve().parent;R=H.parents[3]

def oracle():
 tree=ast.parse((R/'prof/lab/moe_fused_m256.py').read_text());n=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='dc_oracles');ns={'torch':torch};exec(compile(ast.Module(body=[n],type_ignores=[]),'oracle','exec'),ns);return ns['dc_oracles']

def main():
 torch.manual_seed(5544);w=make_weights();w['w13_gs_gate']=w['w13_gs'].clone();w['w13_gs_up']=w['w13_gs']*torch.exp(.55*torch.randn(32,device='cuda'));bank=fx_prepare(w);ref=oracle();ds=[]
 for mode in ('balanced','random','hot','empty','single','random','empty','balanced'):
  ids=torch.full((256,8),255,device='cuda',dtype=torch.int32)
  if mode=='balanced':ids[:,0]=torch.arange(256,device='cuda')%32
  if mode=='hot':ids[:,0]=0
  if mode=='single':ids[7,2]=31
  if mode=='random':ids=torch.rand((256,256),device='cuda').topk(8,-1).indices.int()
  ds.append(dict(hidden_states=torch.randn((256,4096),device='cuda').bfloat16(),topk_ids=ids,topk_weights=torch.softmax(torch.randn((256,8),device='cuda'),-1),config={'local_start':0}))
 wants=ref(ds,w);worst=0.;rowmax=0.;num=0;clocks=[]
 for which in range(len(FX_CONFIGS)):
  for mode in ((0,1) if which==0 else (1,)):
   for grid in (528,1056,2112):
    static={k:v.clone() if torch.is_tensor(v) else v for k,v in ds[0].items()};plan=fx_plan(static,bank,which,grid,mode);fs=plan['steps']
    for f in fs:got=f()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
     for f in fs:got=f()
    for d,want in zip(ds,wants):
     for key in ('hidden_states','topk_ids','topk_weights'):static[key].copy_(d[key])
     graph.replay();torch.cuda.synchronize();delta=got.float()-want.float();err=float(delta.norm()/want.float().norm().clamp_min(1e-12));den=want.float().norm(dim=-1);row=float((delta.norm(dim=-1)/den.clamp_min(den.mean()*.1+1e-12)).max())
     assert bool(torch.isfinite(got).all()) and err<=.01 and row<=.05,(which,mode,grid,err,row);worst=max(worst,err);rowmax=max(rowmax,row);num+=1
    fs[0]();fs[1]();plan['profile'].zero_();fx_profile(plan,bank)
    if mode==1:fs[-1]()
    torch.cuda.synchronize();n=int(plan['v2'][7][-1])*64;p=plan['profile'].view(torch.int64).reshape(-1,8)[:n].cpu();assert bool((p[:,0]>0).all())
    stats=[]
    for j in range(6):
     vals=p[:,j][p[:,j]>0].tolist();stats.append([statistics.median(vals) if vals else 0,max(vals) if vals else 0])
    clocks.append(dict(which=which,mode=mode,grid=grid,cycles_median_max=stats,last_groups=int(p[:,6].sum())))
    print('W2FIX_PASS',which,mode,grid,worst,rowmax,'clocks',stats,flush=True)
 for key in ('w13','w2'):
  raw=torch.randint(256,w[key+'_sf'].shape,device='cuda',dtype=torch.uint8);raw[(raw&127)==127]=126;w[key+'_sf']=raw.view(torch.float8_e4m3fn)
 bank=fx_prepare(w);wide=ds[:3];wants=ref(wide,w)
 for which in range(2):
  for mode in (1,):
   for d,want in zip(wide,wants):
    dd=dict(d,topk_ids=d['topk_ids']+64,config={'local_start':64})
    for f in fx_work(dd,bank,which,528,mode):got=f()
    torch.cuda.synchronize();err=float((got.float()-want.float()).norm()/want.float().norm());assert bool(torch.isfinite(got).all()) and err<=.01,(which,mode,err);worst=max(worst,err);num+=1
 (H/'diagnostics/w2fix_agreement.json').write_text(json.dumps(dict(checks=num,max_relative_L2=worst,max_row=rowmax,phase_order=['main','Y_SS','group_ticket','norm','token_ticket','combine'],clocks=clocks),indent=2)+'\n');print('W2FIX_TOTAL',num,worst,rowmax,flush=True)

if __name__=='__main__':main()
