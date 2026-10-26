"""Native slice math,lossless bank permutation and changed-input graph TASK checks."""
import hashlib,json,linecache
from pathlib import Path
D=Path(__file__).resolve().parent;R=D.parents[5];O=R/'runs/screen/p3a_dependency'
for arm in ('a64','a128','p128','a128d','p128d'):
 r=json.loads((O/(arm+'_static.json')).read_text());assert all(v['ok'] for v in r['static'])
import torch
n={'__name__':'dependency_check'}
def execute(s,f):linecache.cache[f]=(len(s),None,s.splitlines(True),f);exec(compile(s,f,'exec'),n)
execute((R/'prof/lab/moe_nt1_reuse.py').read_text(),'<fixtures>');cells=json.loads((D/'cells.json').read_text())
for v in cells.values():execute((R/v['cell']).read_text(),'<'+v['prefix']+'>')
w=n['rg_generate_weights'](1075353,device=torch.device('cuda'));w['post_norm_w']=w['post_expert_norm'];b=n['nt1_banks'](w);state={k:b[k] for k in ('pnw','gs13','gs2','sms')};state['banks']={'lop3':b['bank']}
proof=[]
for S in (64,128):
 n['dep_bind'](state,S);t=state['banks']['dep'+str(S)];E=32;GP=S//64;NS=4096//S;W2W=8*GP*512;raw=t[...,:W2W].contiguous().view(E,NS,256,GP,32,16).permute(0,2,1,3,4,5).contiguous().view(E,256,32768)
 sf=t[...,W2W:].contiguous().view(E,NS,256,32,GP,2).permute(0,2,1,4,3,5).contiguous().view(E,256,8,8,32,2).permute(0,1,2,4,3,5).contiguous().view(E,256,4096)
 original=b['bank']['w2p'].view(E,256,36864);assert torch.equal(raw,original[:,:,:32768]) and torch.equal(sf,original[:,:,32768:]);assert t.numel()==original.numel();proof.append(dict(S=S,bytes=t.numel(),weight_bits_exact=True,scale_bytes_exact=True))
 print('PACK_EXACT',S,t.numel(),flush=True)
assert torch.equal(n['dpx_pack'](b['bank']['w2p'],128),state['banks']['dep128']);n['dpx_bind'](state,128)
for name in ('dp_base1','dp_base2','dp_nowait1','dp_nowait2'):n[name+'_bind'](state,{8:69});print('BUILT',name,flush=True)
print('SLICE_RESOURCES',[n['dep_module']().dep_info(S,mode) for S,mode in ((64,0),(128,0),(128,1),(128,2))],flush=True)
arms={'A64':(64,0,'dep'),'A128':(128,0,'dep'),'P128':(128,1,'dep'),'A128D':(128,0,'dpx'),'P128D':(128,1,'dpx')};checks=[]
for M in (8,9,16,24,32):
 for start in (0,96,224):
  d=n['fx_make_fixture'](w,M,1080000+M+start);d['config']['local_start']=start;d['topk_ids'].add_(start)
  for label in ('real','empty','zero','all8'):
   data=dict(d)
   if label=='empty':data['topk_ids']=torch.full_like(d['topk_ids'],-1)
   if label=='zero':data['hidden_states']=torch.zeros_like(d['hidden_states'])
   if label=='all8':data['topk_ids']=(torch.arange(8,device='cuda',dtype=torch.int32)+start).expand(M,8).contiguous()
   truth=n['nt1_fp32'](data,w)
   for arm,(S,mode,prefix) in arms.items():
    fn=lambda:n[prefix+'_experts'](data,state,S,mode);fn();g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):out=fn()
    for rep in range(2):
     out.fill_(float('nan'));g.replay();torch.cuda.synchronize();ok,total,row=n['task_check'](out,truth);assert ok,(arm,M,start,label,total,row);checks.append([arm,M,start,label,total,row])
  print('TASK',M,start,flush=True)
 # Captured data changes with no recapture.
 data=n['fx_make_fixture'](w,M,1084123+M);graphs={}
 for arm,(S,mode,prefix) in arms.items():
  g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):out=n[prefix+'_experts'](data,state,S,mode)
  graphs[arm]=(g,out)
 for seed in range(3):
  fresh=n['fx_make_fixture'](w,M,1085353+31*seed+M)
  for k in ('hidden_states','topk_ids','topk_weights'):data[k].copy_(fresh[k])
  truth=n['nt1_fp32'](data,w)
  for arm,(g,out) in graphs.items():
   out.fill_(float('nan'));g.replay();torch.cuda.synchronize();ok,total,row=n['task_check'](out,truth);assert ok,(arm,M,seed,total,row);checks.append([arm,M,'changed',seed,total,row])
 # Timing-only wait removal executes the exact BASE69 API;no correctness verdict.
 fn=n['dp_nowait1_experts'] if M<=8 else n['dp_nowait2_experts'];fn(data,state,69);torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g):ignored=fn(data,state,69)
 for _ in range(3):g.replay()
 torch.cuda.synchronize()
r=dict(all_pass=True,standins=False,TASK_checks=len(checks),checks=checks,pack_bitproof=proof,cells={k:v['cell_sha256'] for k,v in cells.items()},max_total=max(v[-2] for v in checks),max_row=max(v[-1] for v in checks),oracle='NOWAIT runs capture/replay;deliberately incorrect,outputs ignored,not gated or nominated')
(D/'component_validation.json').write_text(json.dumps(r,indent=2)+'\n');print('ALL_PASS',len(checks),r['max_total'],r['max_row'],flush=True)
