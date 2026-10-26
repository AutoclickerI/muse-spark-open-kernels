"""Force raw-fallback packet transfers through the actual complete MoE kernel."""
import json,linecache
from pathlib import Path
import torch
D=Path(__file__).resolve().parent;R=D.parents[4];s=(R/'prof/lab/moe_nt1_reuse.py').read_text();f='<raw packet fixtures>';linecache.cache[f]=(len(s),None,s.splitlines(True),f);n={'__name__':'raw_packet'};exec(compile(s,f,'exec'),n)
s=(D/'pack5_cell.py').read_text();f='<raw packet cell>';linecache.cache[f]=(len(s),None,s.splitlines(True),f);exec(compile(s,f,'exec'),n)
w=n['rg_generate_weights'](5353,device=torch.device('cuda'));w['post_norm_w']=w['post_expert_norm']
for name in ('w13_sf','w2_sf'):
 shape=w[name].shape;pattern=torch.arange(1,127,device='cuda',dtype=torch.uint8);w[name]=pattern.repeat((w[name].numel()+125)//126)[:w[name].numel()].reshape(shape).view(torch.float8_e4m3fn)
b=n['nt1_banks'](w);st={k:b[k] for k in ('pnw','gs13','gs2','sms')};st['banks']={'lop3':b['bank']};n['rvq_bind'](st,{1:95,8:69});records=[]
for m in (1,16,32):
 d=n['fx_make_fixture'](w,m,19901+m);d['topk_ids']=torch.arange(8,device='cuda',dtype=torch.int32).expand(m,8).contiguous();want=n['nt1_fp32'](d,w);g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g):out=n['rvq_experts'](d,st,95 if m==1 else 69)
 for _ in range(3):
  out.fill_(float('nan'));g.replay();torch.cuda.synchronize();ok,total,row=n['task_check'](out,want);assert ok,(m,total,row);records.append([m,total,row])
proof={k:dict(raw_bytes=b['bank'][k].numel(),coded_bytes=v.numel()) for k,v in st['banks']['packet5'].items()};assert all(v['coded_bytes']>v['raw_bytes'] for v in proof.values())
result=dict(TASK_checks=len(records),records=records,raw_packet_transfer_forced=True,banks=proof)
(D/'raw_moe_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True)
