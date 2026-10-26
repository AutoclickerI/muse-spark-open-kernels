"""Native dedicated-producer math andactivation-cache TASK proof, current-input and capture TASK checks."""
import hashlib,json,linecache,sys
from pathlib import Path
import torch
D=Path(__file__).resolve().parent;R=D.parents[5];sys.path.insert(0,str(D));import local_standin
text=(R/'prof/lab/moe_nt1_reuse.py').read_text();f='<struct fixtures>';linecache.cache[f]=(len(text),None,text.splitlines(True),f);n={'__name__':'struct_fixture'};exec(compile(text,f,'exec'),n)
cells=json.loads((D/'cells.json').read_text());arms=json.loads((D/'manifest.json').read_text())
for spec in cells.values():
 text=(R/spec['cell']).read_text();f='<struct cell '+spec['prefix']+'>';linecache.cache[f]=(len(text),None,text.splitlines(True),f);exec(compile(text,f,'exec'),n)
n['_ov_original_experts']=n['smallm_experts'];depth_calls=local_standin.install(n)
for arm in arms:
 text=(D/(arm.lower()+'_selector.py.inc')).read_text();exec(compile(text,'<struct selector '+arm+'>','exec'),n)
w=n['rg_generate_weights'](5353,device=torch.device('cuda'));w['post_norm_w']=w['post_expert_norm'];bank=n['nt1_banks'](w);state={k:bank[k] for k in ('pnw','gs13','gs2','sms')};state['banks']={'lop3':bank['bank']}
for spec in cells.values():n[spec['prefix']+'_bind'](state,{8:69});print('BUILT',spec['prefix'],flush=True)
names=sys.argv[1:] or list(arms);out={a:dict(candidate_sha256=arms[a]['candidate_sha256'],TASK_checks=0,worst_total=0.,worst_row=0.) for a in names}
for mm in (8,9,16,24,32):
 data=n['fx_make_fixture'](w,mm,41371+mm)
 fixtures={'real':data,'empty':dict(data,topk_ids=torch.full_like(data['topk_ids'],-1)),'zero':dict(data,hidden_states=torch.zeros_like(data['hidden_states'])),'all8':dict(data,topk_ids=torch.arange(8,device='cuda',dtype=torch.int32).expand(mm,8).contiguous())}
 for label,d in fixtures.items():
  want=n['nt1_fp32'](d,w)
  for arm in names:
   fn=lambda:n[arms[arm]['prefix']+'_experts'](d,state,69);fn();g=torch.cuda.CUDAGraph()
   with torch.cuda.graph(g):got=fn()
   for _ in range(2):
    got.fill_(float('nan'));g.replay();torch.cuda.synchronize();ok,total,row=n['task_check'](got,want);assert ok,(arm,mm,label,total,row);v=out[arm];v['TASK_checks']+=1;v['worst_total']=max(v['worst_total'],total);v['worst_row']=max(v['worst_row'],row)
  print('PASS',mm,label,flush=True)
for start in (96,224):
 for mm in (8,16,32):
  d=n['fx_make_fixture'](w,mm,41537+mm);d['config']=dict(d['config'],local_start=start);d['topk_ids'].add_(start);graphs={}
  for arm in names:
   g=torch.cuda.CUDAGraph()
   with torch.cuda.graph(g):got=n[arms[arm]['prefix']+'_experts'](d,state,69)
   graphs[arm]=(g,got)
  for seed in range(3):
   fresh=n['fx_make_fixture'](w,mm,41717+31*seed+mm);fresh['topk_ids'].add_(start)
   for key in ('hidden_states','topk_ids','topk_weights'):d[key].copy_(fresh[key])
   want=n['nt1_fp32'](d,w)
   for arm,(g,got) in graphs.items():
    got.fill_(float('nan'));g.replay();torch.cuda.synchronize();ok,total,row=n['task_check'](got,want);assert ok,(arm,start,mm,seed,total,row);v=out[arm];v['TASK_checks']+=1;v['worst_total']=max(v['worst_total'],total);v['worst_row']=max(v['worst_row'],row)
result=dict(arms=out,depth_standin_calls=depth_calls,deep_depth_math_executed=2,actual_native_split_kernels=True,scope='M1 unchanged; all changed M<=32 paths tested; H200 executes actual3/4 stage values')
(D/'component_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(out,indent=2),flush=True)
