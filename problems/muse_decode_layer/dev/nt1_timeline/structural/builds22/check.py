import argparse,hashlib,json,linecache,sys,time
from pathlib import Path
import torch
from local_support import install_cache
D=Path(__file__).resolve().parent;R=D.parents[5];s=(R/'prof/lab/moe_nt1_reuse.py').read_text();f='<p3a-builds22-fixture>';linecache.cache[f]=(len(s),None,s.splitlines(True),f);n={'__name__':'builds22_check'};exec(compile(s,f,'exec'),n)
cells=json.loads((D/'cells.json').read_text());manifest=json.loads((D/'manifest.json').read_text())
for c in cells.values():
 p=R/c['cell'];text=p.read_text();linecache.cache[str(p)]=(len(text),None,text.splitlines(True),str(p));exec(compile(text,str(p),'exec'),n)
ap=argparse.ArgumentParser();ap.add_argument('--arms',nargs='+');args=ap.parse_args()
emulation=install_cache(n);w=n['rg_generate_weights'](1675353,device=torch.device('cuda'));w['post_norm_w']=w['post_expert_norm'];b=n['nt1_banks'](w);state={k:b[k] for k in ('pnw','gs13','gs2','sms')};state['banks']={'lop3':b['bank']};out=json.loads((D/'leaf_validation.json').read_text()) if (D/'leaf_validation.json').exists() else {}
out={k:v for k,v in out.items() if v['source_sha256']==manifest[k]['candidate_sha256']}
for arm in manifest:
 if args.arms and arm not in args.arms:continue
 for i in range(3):n[f'b22_{arm.lower()}{i}_bind'](state,{1:95,8:69})
 rows=[]
 for M in (1,8,9,16,24,32):
  pr,rs=n['routed_fixture'](w,M,1679000+M);x,ids,tw,lids=n['route_call'](pr,rs);i=0 if M==1 else 1 if M==8 else 2;fn=n[f'b22_{arm.lower()}{i}_experts'];F=95 if M==1 else 69
  for mode in ('real','all','empty','single','dense'):
   for start in (0,96,224):
    ll=lids.clone() if mode=='real' else torch.arange(M*8,device='cuda',dtype=torch.int32).view(M,8)%32 if mode=='all' else torch.arange(8,device='cuda',dtype=torch.int32)[None,:].expand(M,8).contiguous() if mode=='dense' else torch.full_like(lids,-1)
    if mode=='single':ll[0,0]=0
    ll=torch.where(ll>=0,ll+start,ll);data=dict(hidden_states=x.clone(),topk_ids=ll,topk_weights=tw,config={'local_start':start});g=n['TimelineGraph'](lambda:fn(data,state,F))
    for scale in (1.,.75,1.25):
     data['hidden_states'].mul_(scale);got=g.replay();want=n['nt1_fp32'](data,w);ok,total,row=n['task_check'](got,want);assert ok,(arm,M,mode,start,scale,total,row);rows.append([M,mode,start,scale,total,row])
  print('NATIVE',arm,M,'pass',flush=True)
 out[arm]=dict(source_sha256=manifest[arm]['candidate_sha256'],checks=len(rows),all_pass=True,changed_captured_inputs=True,max_total=max(v[-2] for v in rows),max_row=max(v[-1] for v in rows),rows=rows,standin=emulation if arm=='CACHE1' else [],native_arithmetic=True)
 (D/'leaf_validation.json').write_text(json.dumps(out,indent=2)+'\n')
print('ALL_LEAF_PASS',sum(x['checks'] for x in out.values()),flush=True)
