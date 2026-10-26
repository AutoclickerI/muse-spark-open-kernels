"""Public attention TASK on fresh/strided partial inputs, plus the approximation's numerical envelope."""
import hashlib,importlib.util,json,math,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];sys.path[:0]=[str(R/'tools'),str(R/'problems/muse_attention_decode')]
import torch,local_eval
spec=importlib.util.spec_from_file_location('p5_so_runtime',H/'runtime.py');rt=importlib.util.module_from_spec(spec);spec.loader.exec_module(rt)
import structural
ref=local_eval.load_module('reference',R/'problems/muse_attention_decode/reference.py');ref._fa3_scheduler_metadata=lambda *a,**kw:None
rt.LOCAL=True;rt.SOURCES={f+'EA':structural.source(f,True) for f in ('d','v')}
cases=[dict(batch=B,q_len=L,ctx=80000,page=1,spread='agentic',dist='soft') for B,L in ((16,1),(32,1),(16,8),(32,8))]
cases += [dict(batch=3,q_len=L,ctx=4000,page=64,kv_scale=.015625,dist='edges') for L in (1,8)]
cases += [dict(batch=1,q_len=L,ctx=10,page=64,dist='edges') for L in (1,8)]
cases += [dict(batch=4,q_len=L,ctx=6000,page=1,dist=dist) for L in (1,8) for dist in ('soft','coherent')]
rows=[]
for case in cases:
 for splits in (0,3):
    d=ref.generate_input(**case,heads='tp8',layer='sliding',seed=34000037);T=case['batch']*case['q_len'];width=d['qkv'].shape[-1];part=None
    if splits:
        part=(torch.randn((splits,T,width+64),device='cuda')*.03)[:,:,:width];part[-1]+=d['qkv'].reshape(T,width).float()-part.sum(0);d['qkv'].copy_(part.sum(0).bfloat16().view_as(d['qkv']))
    family='d' if case['q_len']==1 else 'v';fn=lambda:rt.attention(d,None,family+'EA',qkv=None if part is None else (part,splits))
    fn();torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):out=fn()
    fresh=ref.generate_input(**case,heads='tp8',layer='sliding',seed=34104766)
    if splits:
        part.normal_(0,.03);part[-1]+=fresh['qkv'].reshape(T,width).float()-part.sum(0);fresh['qkv']=part.sum(0).bfloat16().view_as(fresh['qkv'])
    local_eval.copy_into(d,fresh)
    for _ in range(5):g.replay()
    torch.cuda.synchronize();ok,msg=ref.check_implementation(d,ref.collect(d,out));assert ok,(case,splits,msg)
    assert all(int(torch.count_nonzero(v))==0 for v in rt.COUNTERS.values())
    rows.append(dict(case=case,splits=splits,TASK=ok,message=msg));print('PASS',case['batch'],case['q_len'],case['dist'],splits,msg[:80],flush=True)
# A numerical characterization, not a stricter correctness gate.
x=torch.linspace(-120,8,500001,device='cuda');approx=torch.round(torch.addcmul(torch.full_like(x,1064992256.),x,torch.full_like(x,8388608.))).to(torch.int32).view(torch.float32)
relative=(approx/torch.exp2(x)-1);envelope=[float(relative.min()),float(relative.max())]
result=dict(TASK_passed=len(rows),rows=rows,normal_exp_relative_envelope=envelope,source_sha256={n:hashlib.sha256(s.encode()).hexdigest() for n,s in rt.SOURCES.items()},standins=['decode C1 DSM every key','verify two real WGs/global last-WG combine'])
(H/'structural_admission.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS',len(rows),'TASK; exp envelope',envelope)
