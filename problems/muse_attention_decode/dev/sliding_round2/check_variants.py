"""Run all actual per-shape arms via local_eval's input/reference/capture helpers."""
import importlib.util,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'problems/muse_attention_decode')]
import torch,local_eval,kernels,runtime
runtime.KERNELS={k:kernels.source(k) for k in ('last','early','dsm')};runtime.KERNELS['dsm_old']=kernels.build.source()
ref=local_eval.load_module('reference',ROOT/'problems/muse_attention_decode/reference.py');ref._fa3_scheduler_metadata=lambda *a,**kw:None
real=runtime.attention
standins=[]
def local_attention(data,qkv=None,kind='last',qt=1,ns=8):
    if kind in ('dsm','dsm_old') and torch.cuda.get_device_capability()[0]!=9:
        standins.append([data['config']['batch'],data['config']['q_len'],kind,qt,ns]);ns=1
    return real(data,qkv,kind,qt,ns)
runtime.attention=local_attention
cases=[((16,1),dict(batch=16,ctx=80000,page=1,spread='agentic',dist='soft')),
       ((32,1),dict(batch=32,ctx=80000,page=1,spread='agentic',dist='soft')),
       ((1,8),dict(batch=1,q_len=8,ctx=80000,page=1,dist='soft')),
       ((1,8),dict(batch=1,q_len=8,ctx=10,page=64,dist='edges')),
       ((1,8),dict(batch=1,q_len=8,ctx=2305,page=64,kv_scale=.015625,dist='soft'))]
records=[]
for key,case in cases:
    for name,arm in runtime.ARMS[key].items():
        for splits in (0,3):
            data=ref.generate_input(**case,heads='tp8',layer='sliding',seed=671003)
            cfg=data['config'];W=data['qkv'].shape[-1];T=cfg['batch']*cfg['q_len'];partial=None
            if splits:
                partial=(torch.randn((splits,T,W+64),device='cuda')*.03)[:,:,:W]
                partial[-1]+=data['qkv'].reshape(T,W).float()-partial.sum(0)
                data['qkv'].copy_(partial.sum(0).to(torch.bfloat16).view_as(data['qkv']))
            fn=lambda:runtime.apply_arm(data,None if partial is None else (partial,splits),arm)
            fn();torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):out=fn()
            fresh=ref.generate_input(**case,heads='tp8',layer='sliding',seed=713009)
            if splits:
                partial.copy_(torch.randn_like(partial)*.03);partial[-1]+=fresh['qkv'].reshape(T,W).float()-partial.sum(0)
                fresh['qkv']=partial.sum(0).to(torch.bfloat16).view_as(fresh['qkv'])
            local_eval.copy_into(data,fresh)
            for _ in range(8):graph.replay()
            torch.cuda.synchronize();ok,msg=ref.check_implementation(data,ref.collect(data,out))
            assert ok,(key,name,splits,msg)
            assert all(int(torch.count_nonzero(c))==0 for c in runtime._COUNTERS.values())
            records.append(dict(case=case,key=key,name=name,arm=arm,splits=splits,ok=ok,msg=msg))
            print('PASS',key,name,splits,msg[:100],flush=True)
(HERE/'out/variant_admission.json').write_text(json.dumps(dict(records=records,standins=standins),indent=2)+'\n')
print('PASS all',len(records),'variant cells; DSM gangs use genuine C1 math onsm120; native SASS audited separately')
