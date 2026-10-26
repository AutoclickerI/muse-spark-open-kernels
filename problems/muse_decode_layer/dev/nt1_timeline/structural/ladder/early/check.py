import ast,hashlib,json,linecache,sys
from pathlib import Path
import torch
D=Path(__file__).resolve().parent;R=D.parents[6];source=(R/'prof/lab/moe_nt1_reuse.py').read_text();f='<early-fixture>';linecache.cache[f]=(len(source),None,source.splitlines(True),f);n={'__name__':'early_check'};exec(compile(source,f,'exec'),n)
for v in json.load(open(D/'cells.json')).values():
 if not v['prefix'].startswith('ear'):continue
 p=R/v['cell'];s=p.read_text();fn=str(p);linecache.cache[fn]=(len(s),None,s.splitlines(True),fn);exec(compile(s,fn,'exec'),n)
w=n['rg_generate_weights'](1575353,device=torch.device('cuda'));w['post_norm_w']=w['post_expert_norm'];b=n['nt1_banks'](w);state={k:b[k] for k in ('pnw','gs13','gs2','sms')};state['banks']={'lop3':b['bank']}
for i in (1,2):n['ear'+str(i)+'_bind'](state,{8:69})
rows=[]
for M in (8,9,16,24,32):
 pr,rs=n['routed_fixture'](w,M,1579000+M);x,ids,tw,lids=n['route_call'](pr,rs)
 for mode in ('real','all','empty'):
  for start in (0,96,224):
   ll=lids.clone() if mode=='real' else torch.arange(M*8,device='cuda',dtype=torch.int32).view(M,8)%32 if mode=='all' else torch.full_like(lids,-1);ll=torch.where(ll>=0,ll+start,ll);data=dict(hidden_states=x.clone(),topk_ids=ll,topk_weights=tw,config={'local_start':start});fn=n['ear'+str(1 if M==8 else 2)+'_experts'];g=n['TimelineGraph'](lambda:fn(data,state,69))
   for scale in (1.,.75,1.25):
    data['hidden_states'].mul_(scale);got=g.replay();want=n['nt1_fp32'](data,w);ok,total,row=n['task_check'](got,want);assert ok,(M,mode,start,total,row);rows.append([M,mode,start,scale,total,row])
r=dict(checks=len(rows),all_pass=True,native_early_no_standin=True,changed_captured_inputs=True,max_total=max(v[-2] for v in rows),max_row=max(v[-1] for v in rows),rows=rows,manifest_sha256=hashlib.sha256((D/'manifest.json').read_bytes()).hexdigest());(D/'validation.json').write_text(json.dumps(r,indent=2)+'\n');print('EARLY_NATIVE_PASS',len(rows),r['max_total'],r['max_row'])
