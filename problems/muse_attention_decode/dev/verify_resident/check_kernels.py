"""Fresh-input correctness of the full resident history and WGMMA step math, using explicit local reductions."""
import json,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];sys.path[:0]=[str(R/'tools'),str(R/'problems/muse_attention_decode')]
import torch,local_eval,runtime,local_standin
runtime.KERNELS={};standin=local_standin.install(runtime.__dict__)
ref=local_eval.load_module('reference',R/'problems/muse_attention_decode/reference.py');ref._fa3_scheduler_metadata=lambda *a,**kw:None
cases=[dict(batch=16,ctx=80000,page=1,spread='agentic',dist='soft'),dict(batch=32,ctx=80000,page=1,spread='agentic',dist='soft'),
 dict(batch=3,ctx=4000,page=1,kv_scale=.015625,dist='soft'),dict(batch=1,ctx=10,page=64,dist='edges'),dict(batch=4,ctx=6000,page=1,dist='edges'),dict(batch=2,ctx=300000,page=1,dist='soft')]
rows=[]
for case in cases:
 for name in runtime.TOPOLOGY:
  for splits in (0,3):
   data=ref.generate_input(**case,q_len=8,heads='tp8',layer='sliding',seed=9151091)
   cfg=data['config'];W=data['qkv'].shape[-1];T=cfg['batch']*8;part=None
   if splits:
    part=(torch.randn((splits,T,W+64),device='cuda')*.03)[:,:,:W];part[-1]+=data['qkv'].reshape(T,W).float()-part.sum(0)
    data['qkv'].copy_(part.sum(0).to(torch.bfloat16).view_as(data['qkv']))
   fn=lambda:runtime.attention(data,None if part is None else (part,splits),name)
   fn();torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
   with torch.cuda.graph(g):out=fn()
   fresh=ref.generate_input(**case,q_len=8,heads='tp8',layer='sliding',seed=9311097)
   if splits:
    part.copy_(torch.randn_like(part)*.03);part[-1]+=fresh['qkv'].reshape(T,W).float()-part.sum(0)
    fresh['qkv']=part.sum(0).to(torch.bfloat16).view_as(fresh['qkv'])
   local_eval.copy_into(data,fresh)
   for _ in range(8):g.replay()
   torch.cuda.synchronize();ok,msg=ref.check_implementation(data,ref.collect(data,out));assert ok,(case,name,splits,msg)
   assert all(int(torch.count_nonzero(c))==0 for c in runtime._COUNTERS.values())
   rows.append(dict(case=case,name=name,splits=splits,ok=ok,msg=msg));print('PASS',case['batch'],case['ctx'],name,splits,msg[:100],flush=True)
(H/'out/component_admission.json').write_text(json.dumps(dict(standin=standin,rows=rows),indent=2)+'\n');print('PASS',len(rows),'fresh/partial/counter cells')
