"""Native LAZY leaf, nearby shapes/local_start/skew and changed captured inputs."""
import hashlib,json,linecache
from pathlib import Path
import torch
D=Path(__file__).resolve().parent;R=D.parents[5];p=R/'prof/lab/moe_p3a_operand.py';s=p.read_text();f='<operand-leaf-check>';linecache.cache[f]=(len(s),None,s.splitlines(True),f);n={'__name__':'operand_check'};exec(compile(s,f,'exec'),n)
w=n['rg_generate_weights'](1575353,device=torch.device('cuda'));w['post_norm_w']=w['post_expert_norm'];b=n['nt1_banks'](w);state={k:b[k] for k in ('pnw','gs13','gs2','sms')};state['banks']={'lop3':b['bank']}
for i in range(3):n['opa'+str(i)+'_bind'](state,{1:95,8:69})
rows=[]
for M in (1,8,9,16,24,32):
 pr,rs=n['routed_fixture'](w,M,1579000+M);x,ids,tw,lids=n['route_call'](pr,rs);i=0 if M==1 else 1 if M==8 else 2;mod=n['opa'+str(i)+'_module']();F=95 if M==1 else 69
 for mode in ('real','all','empty'):
  for start in (0,96,224):
   ll=lids.clone() if mode=='real' else torch.arange(M*8,device='cuda',dtype=torch.int32).view(M,8)%32 if mode=='all' else torch.full_like(lids,-1);ll=torch.where(ll>=0,ll+start,ll);data=dict(hidden_states=x.clone(),topk_ids=ll,topk_weights=tw,config={'local_start':start});aa=n['op_args'](data,state,'LAZY');ptrs=[t.data_ptr() for t in aa['tensors']];grid=aa['info'][-1]
   def run():
    mod.op_run(ptrs,M,start,2,grid,torch.cuda.current_stream().cuda_stream,F,9)
    return aa['tensors'][12]
   g=n['TimelineGraph'](run)
   for scale in (1.,.75,1.25):
    data['hidden_states'].mul_(scale);got=g.replay();want=n['nt1_fp32'](data,w);ok,total,row=n['task_check'](got,want);assert ok,(M,mode,start,total,row);rows.append([M,mode,start,scale,total,row])
out=dict(checks=len(rows),all_pass=True,native=True,changed_captured_inputs=True,payload_sha256=hashlib.sha256(s.encode()).hexdigest(),max_total=max(v[-2] for v in rows),max_row=max(v[-1] for v in rows),rows=rows,scope='Native small-M leaf only;whole S21 graph not yet measured')
(D/'leaf_validation.json').write_text(json.dumps(out,indent=2)+'\n');print('OPERAND_LAZY_NATIVE_PASS',len(rows),out['max_total'],out['max_row'])
