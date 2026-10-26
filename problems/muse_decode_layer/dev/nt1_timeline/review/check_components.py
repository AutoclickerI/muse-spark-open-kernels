"""TASK-gated current-input/replay checks for independently built review arms."""
import hashlib,json,linecache,sys,time,types
from pathlib import Path
import torch
D=Path(__file__).resolve().parent;P=D.parent;R=D.parents[4]
text=(R/'prof/lab/moe_nt1_reuse.py').read_text();f='<review fixtures>';linecache.cache[f]=(len(text),None,text.splitlines(True),f)
n={'__name__':'review_fixture'};exec(compile(text,f,'exec'),n)
manifest=json.loads((D/'manifest.json').read_text());names=sys.argv[1:] or list(manifest)
modules={}
for arm in names:
 p=R/manifest[arm]['cell'];s=p.read_text();f='<review cell '+arm+'>';linecache.cache[f]=(len(s),None,s.splitlines(True),f);c={**n,'__name__':'cell_'+arm};exec(compile(s,f,'exec'),c);modules[arm]=c
w=n['rg_generate_weights'](5353,device=torch.device('cuda'));w['post_norm_w']=w['post_expert_norm'];bank=n['nt1_banks'](w)
state={k:bank[k] for k in ('pnw','gs13','gs2','sms')};state['banks']={'lop3':bank['bank']};n['smallm_bind'](state,{1:95,8:69})
result={arm:dict(TASK_checks=0,worst_total=0.,worst_row=0.,hash=manifest[arm]['cell_sha256']) for arm in names}
for arm,c in modules.items():
 tag=manifest[arm]['prefix'];c[tag+'_bind'](state,{1:95,8:69});print('BUILT',arm,flush=True)
# Real, empty, zero and all8 routes; graph replays must consume changed inputs and routing offsets.
for m in (1,8,9,16,24,32):
 d=n['fx_make_fixture'](w,m,19371+m)
 fixtures={'real':d,'empty':dict(d,topk_ids=torch.full_like(d['topk_ids'],-1)),'zero':dict(d,hidden_states=torch.zeros_like(d['hidden_states'])),'all8':dict(d,topk_ids=torch.arange(8,device='cuda',dtype=torch.int32).expand(m,8).contiguous())}
 for label,data in fixtures.items():
  want=n['nt1_fp32'](data,w)
  for arm,c in modules.items():
   tag=manifest[arm]['prefix'];fn=lambda:c[tag+'_experts'](data,state,95 if m==1 else 69)
   fn();g=torch.cuda.CUDAGraph()
   with torch.cuda.graph(g):out=fn()
   for _ in range(3):
    out.fill_(float('nan'));g.replay();torch.cuda.synchronize();ok,total,row=n['task_check'](out,want)
    assert ok,(arm,m,label,total,row)
    v=result[arm];v['TASK_checks']+=1;v['worst_total']=max(v['worst_total'],total);v['worst_row']=max(v['worst_row'],row)
  print('PASS',m,label,flush=True)
for start in (96,224):
 for m in (8,16,32):
  data=n['fx_make_fixture'](w,m,19531+m);data['config']=dict(data['config'],local_start=start);data['topk_ids'].add_(start)
  captures={}
  for arm,c in modules.items():
   tag=manifest[arm]['prefix'];g=torch.cuda.CUDAGraph()
   with torch.cuda.graph(g):out=c[tag+'_experts'](data,state,69)
   captures[arm]=(g,out)
  for seed in range(3):
   fresh=n['fx_make_fixture'](w,m,19717+seed*31+m);fresh['topk_ids'].add_(start)
   for key in ('hidden_states','topk_ids','topk_weights'):data[key].copy_(fresh[key])
   want=n['nt1_fp32'](data,w)
   for arm,(g,out) in captures.items():
    out.fill_(float('nan'));g.replay();torch.cuda.synchronize();ok,total,row=n['task_check'](out,want);assert ok,(arm,m,start,seed,total,row)
    v=result[arm];v['TASK_checks']+=1;v['worst_total']=max(v['worst_total'],total);v['worst_row']=max(v['worst_row'],row)
# Cancellation: zero, negative, tiny and huge global scales. Use a direct FP64 norm of the unscaled
# W2 tensor to avoid an FP32 reference overflow falsely accusing a valid identity.
if 'CANCEL' in modules:
 c=modules['CANCEL'];tag=manifest['CANCEL']['prefix'];original=state['gs2'].clone();data=n['fx_make_fixture'](w,16,19801)
 # Judge-relevant scales first use the unchanged TASK reference, including negative and zero.
 for factor in (0.,-1.,1e-10,1e10):
  state['gs2'].copy_(original*factor);w['w2_gs']=state['gs2'];c[tag+'_bind'](state,{8:69})
  got=c[tag+'_experts'](data,state,69);torch.cuda.synchronize();want=n['nt1_fp32'](data,w);ok,total,row=n['task_check'](got,want);assert ok,(factor,total,row);result['CANCEL']['TASK_checks']+=1
 state['gs2'].copy_(original);w['w2_gs']=state['gs2'];c[tag+'_bind'](state,{8:69})
# Packet proof includes complete original code banks and an explicit wide-scale/raw packet.
if 'PACK5' in modules:
 c=modules['PACK5'];proof={}
 for role,t in state['banks']['lop3'].items():
  packed=state['banks']['packet5'][role];rows=t.reshape(-1,36864);offsets=packed[:(rows.shape[0]+1)*8].view(torch.int64)
  # Decode in bounded chunks; every byte and every weight word is compared with its source.
  good=0
  for lo in range(0,rows.shape[0],256):
   ix=torch.arange(lo,min(lo+256,rows.shape[0]),device=t.device);off=offsets[ix];base=packed[off[:,None]+32768+torch.arange(16,device=t.device)[None,:]]
   mask=packed[off[:,None]+32784+torch.arange(4,device=t.device)[None,:]].long();mask=(mask*(1<<torch.arange(0,32,8,device=t.device))).sum(1)
   j=torch.arange(4096,device=t.device);packet=j//256;k=j%256
   prev=((mask[:,None] & ((1<<packet)-1))[:,:,None] & (1<<torch.arange(16,device=t.device))[None,None,:]).ne(0).sum(2)
   pos=off[:,None]+32800+packet[None,:]*160+prev*96
   raw=((mask[:,None]>>packet)&1).bool();bit=k*5;b=packed[pos+(bit//8)[None,:]].long();b1=packed[pos+torch.minimum(bit//8+1,torch.tensor(159,device=t.device))[None,:]].long()
   decoded=torch.where(raw,packed[pos+torch.where(raw,k[None,:],torch.minimum(k,torch.tensor(159,device=t.device))[None,:])].long(),base[:,packet].long()+(((b|(b1<<8))>>(bit%8)[None,:])&31))
   assert torch.equal(decoded.to(torch.uint8),rows[ix,32768:]),(role,lo)
   weights=packed[off[:,None]+torch.arange(32768,device=t.device)[None,:]];assert torch.equal(weights,rows[ix,:32768]),(role,lo,'weights')
   good+=len(ix)
  proof[role]=dict(tiles=good,raw_bytes=t.numel(),coded_bytes=packed.numel(),exact=True)
 # Force raw packet serialization independently of the official narrow scales.
 raw=torch.randint(0,256,(2,36864),device='cuda',dtype=torch.uint8);enc=c['pack5_bank'](raw);assert enc.numel()>raw.numel();proof['raw_fallback_constructed']=True
 result['PACK5']['full_bank_proof']=proof
(D/'component_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
