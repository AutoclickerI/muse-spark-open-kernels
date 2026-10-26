"""Check both independent WGs, partial inputs, fresh captures and last-owner counter reset."""
import hashlib,json,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3]
sys.path[:0]=[str(R/'tools'),str(R/'problems/muse_attention_decode')]
import torch,local_eval,kernel,runtime
pipeline='--pipeline' in sys.argv
if pipeline:runtime.TOPOLOGY={'PIPE2':2}
runtime.KERNELS={n:kernel.source(c,local=True,stream=n.startswith(('STREAM','PIPE')),pipeline=n.startswith('PIPE')) for n,c in runtime.TOPOLOGY.items()}
ref=local_eval.load_module('reference',R/'problems/muse_attention_decode/reference.py');ref._fa3_scheduler_metadata=lambda *a,**kw:None
cases=[dict(batch=16,ctx=80000,page=1,spread='agentic',dist='soft'),dict(batch=32,ctx=80000,page=1,spread='agentic',dist='soft'),
       dict(batch=3,ctx=4000,page=1,kv_scale=.015625,dist='soft'),dict(batch=1,ctx=10,page=64,dist='edges'),dict(batch=4,ctx=6000,page=1,dist='edges')]
rows=[]
for case in cases:
    for name in runtime.TOPOLOGY:
        for splits in (0,3):
            d=ref.generate_input(**case,q_len=8,heads='tp8',layer='sliding',seed=20100107)
            width=d['qkv'].shape[-1];T=case['batch']*8;part=None
            if splits:
                part=(torch.randn((splits,T,width+64),device='cuda')*.03)[:,:,:width]
                part[-1]+=d['qkv'].reshape(T,width).float()-part.sum(0)
                d['qkv'].copy_(part.sum(0).to(torch.bfloat16).view_as(d['qkv']))
            fn=lambda:runtime.attention(d,None if part is None else (part,splits),name)
            fn();torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):out=fn()
            fresh=ref.generate_input(**case,q_len=8,heads='tp8',layer='sliding',seed=20204836)
            if splits:
                part.copy_(torch.randn_like(part)*.03);part[-1]+=fresh['qkv'].reshape(T,width).float()-part.sum(0)
                fresh['qkv']=part.sum(0).to(torch.bfloat16).view_as(fresh['qkv'])
            local_eval.copy_into(d,fresh)
            for _ in range(8):graph.replay()
            torch.cuda.synchronize();ok,msg=ref.check_implementation(d,ref.collect(d,out));assert ok,(case,name,splits,msg)
            assert all(int(torch.count_nonzero(v))==0 for v in runtime._COUNTERS.values())
            rows.append(dict(case=case,name=name,splits=splits,ok=ok,msg=msg));print('PASS',case['batch'],name,splits,msg[:95],flush=True)
(H/('pipeline_admission.json' if pipeline else 'kernel_admission.json')).write_text(json.dumps(dict(rows=rows,standin='Two real WGs per CTA; C4 slices for shared capacity, nonspinning global last-WG reduction instead of native DSM; H200 source compiles separately',source_sha256={n:hashlib.sha256(s.encode()).hexdigest() for n,s in runtime.KERNELS.items()}),indent=2)+'\n')
