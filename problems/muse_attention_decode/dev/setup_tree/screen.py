"""GPU0 TASK checks on fresh captured inputs and strided partials; local DSM C1."""
import hashlib,importlib.util,json,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];sys.path[:0]=[str(R/'tools'),str(R/'problems/muse_attention_decode')]
import torch,local_eval,kernels_new as k
sp=importlib.util.spec_from_file_location('p5_setup_runtime',H.parent/'sliding_oracle/runtime.py');rt=importlib.util.module_from_spec(sp);sp.loader.exec_module(rt)
ref=local_eval.load_module('reference',R/'problems/muse_attention_decode/reference.py');ref._fa3_scheduler_metadata=lambda *a,**kw:None
hoist='--hoist' in sys.argv
arms=lambda f:('H',) if hoist else ('A','Q') if f=='d' else ('A','Q','T')
rt.LOCAL=True;rt.SOURCES={f+a+'A':k.source(f,a,True) for f in ('d','v') for a in arms(f)}
rt.SOURCES.update({f+'M':k.oracle(f,True) for f in ('d','v')})
cases=[dict(batch=B,q_len=L,ctx=80000,page=1,spread='agentic',dist='soft') for B,L in ((16,1),(32,1),(16,8),(32,8))]
cases += [dict(batch=B,q_len=L,ctx=ctx,page=page,kv_scale=scale,dist='edges') for B,ctx,page,scale in ((1,10,64,1.),(3,4000,64,.015625)) for L in (1,8)]
rows=[]
for case in cases:
 for splits in (0,3):
    d=ref.generate_input(**case,heads='tp8',layer='sliding',seed=41000017);T=case['batch']*case['q_len'];width=d['qkv'].shape[-1];part=None
    if splits:
        part=(torch.randn((splits,T,width+64),device='cuda')*.03)[:,:,:width];part[-1]+=d['qkv'].reshape(T,width).float()-part.sum(0);d['qkv'].copy_(part.sum(0).bfloat16().view_as(d['qkv']))
    f='d' if case['q_len']==1 else 'v'
    for a in arms(f):
        fn=lambda:rt.attention(d,None,f+a+'A',qkv=None if part is None else (part,splits))
        fn();torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):out=fn()
        fresh=ref.generate_input(**case,heads='tp8',layer='sliding',seed=41104746)
        if splits:
            part.normal_(0,.03);part[-1]+=fresh['qkv'].reshape(T,width).float()-part.sum(0);fresh['qkv']=part.sum(0).bfloat16().view_as(fresh['qkv'])
        local_eval.copy_into(d,fresh)
        for _ in range(3):g.replay()
        torch.cuda.synchronize();ok,msg=ref.check_implementation(d,ref.collect(d,out));assert ok,(case,splits,a,msg)
        rows.append(dict(case=case,splits=splits,arm=a,TASK=ok,message=msg));print('PASS',case['batch'],case['q_len'],splits,a,flush=True)
    if not splits and not hoist:
        coords=rt.coordinates(d)
        for n,mode in k.MODES.items():
            if n=='G':continue
            fn=lambda:rt.attention(d,coords,f+'M',mode);fn();torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):out=fn()
            g.replay();torch.cuda.synchronize()
            if n in ('M','M2'):
                ok,msg=ref.check_implementation(d,ref.collect(d,out));assert ok,(case,n,msg)
            else:ok=None
            rows.append(dict(case=case,arm=n,TASK=ok))
(H/('hoist_admission.json' if hoist else 'admission.json')).write_text(json.dumps(dict(rows=rows,TASK=sum(x['TASK'] is True for x in rows),altered_work=sum(x['TASK'] is None for x in rows),sources={n:hashlib.sha256(s.encode()).hexdigest() for n,s in rt.SOURCES.items()},standin='Decode C1; verify one CTA/two real WGs; H page registers expand to sixteen tiles locally, native four tiles; native DSM runs only H200'),indent=2)+'\n')
