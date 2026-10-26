"""Native TASK and changed-input graph checks for exact S19 address arms on GPU0."""
import hashlib,json,linecache
from pathlib import Path
import torch
D=Path(__file__).resolve().parent;R=D.parents[5]
n={'__name__':'p3a_ipc_check'}
def run(s,f):linecache.cache[f]=(len(s),None,s.splitlines(True),f);exec(compile(s,f,'exec'),n)
run((R/'prof/lab/moe_nt1_reuse.py').read_text(),'<fixtures>');cells=json.loads((D/'cells.json').read_text())
for v in cells.values():run((R/v['cell']).read_text(),'<'+v['prefix']+'>')
w=n['rg_generate_weights'](875353,device=torch.device('cuda'));w['post_norm_w']=w['post_expert_norm'];b=n['nt1_banks'](w);state={k:b[k] for k in ('pnw','gs13','gs2','sms')};state['banks']={'lop3':b['bank']}
for v in cells.values():n[v['prefix']+'_bind'](state,{1:95} if v['M']==[1] else {8:69});print('BUILT',v['prefix'],flush=True)
checks=[]
for mm in (1,8,16,32):
 for start in (0,96,224):
  data=n['fx_make_fixture'](w,mm,879000+mm+start);data['config']['local_start']=start;data['topk_ids'].add_(start)
  for label in ('real','empty','zero','all8'):
   d=dict(data)
   if label=='empty':d['topk_ids']=torch.full_like(d['topk_ids'],-1)
   if label=='zero':d['hidden_states']=torch.zeros_like(d['hidden_states'])
   if label=='all8':d['topk_ids']=(torch.arange(8,device='cuda',dtype=torch.int32)+start).expand(mm,8).contiguous()
   truth=n['nt1_fp32'](d,w)
   for arm in ('BASE','AFFINE','PRIVATE'):
    tag='ipc_'+arm.lower()+str(0 if mm==1 else 1 if mm==8 else 2);fn=lambda:n[tag+'_experts'](d,state,95 if mm==1 else 69);fn();g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):got=fn()
    for replay in range(2):
     got.fill_(float('nan'));g.replay();torch.cuda.synchronize();ok,total,row=n['task_check'](got,truth);assert ok,(arm,mm,start,label,total,row);checks.append([arm,mm,start,label,total,row])
  print('PASS',mm,start,flush=True)
 # Change captured tensors,including routes,without recapture.
 d=n['fx_make_fixture'](w,mm,883123+mm);graphs={}
 for arm in ('BASE','AFFINE','PRIVATE'):
  tag='ipc_'+arm.lower()+str(0 if mm==1 else 1 if mm==8 else 2);g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):got=n[tag+'_experts'](d,state,95 if mm==1 else 69)
  graphs[arm]=(g,got)
 for seed in range(3):
  fresh=n['fx_make_fixture'](w,mm,883353+31*seed+mm)
  for k in ('hidden_states','topk_ids','topk_weights'):d[k].copy_(fresh[k])
  truth=n['nt1_fp32'](d,w)
  for arm,(g,got) in graphs.items():
   got.fill_(float('nan'));g.replay();torch.cuda.synchronize();ok,total,row=n['task_check'](got,truth);assert ok,(arm,mm,seed,total,row);checks.append([arm,mm,'changed',seed,total,row])
r=dict(cells={k:v['cell_sha256'] for k,v in cells.items()},checks=checks,TASK_checks=len(checks),all_pass=True,standins=False,max_total=max(v[-2] for v in checks),max_row=max(v[-1] for v in checks));(D/'component_validation.json').write_text(json.dumps(r,indent=2)+'\n');print('ALL_PASS',r['TASK_checks'],r['max_total'],r['max_row'])
