"""Fresh captured inputs, coordinate equivalence and all oracle branches; timings are not native evidence."""
import hashlib,importlib.util,json,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];sys.path[:0]=[str(R/'tools'),str(R/'problems/muse_attention_decode')]
import torch,local_eval
spec=importlib.util.spec_from_file_location('p5_oracle_runtime',H/'runtime.py');runtime=importlib.util.module_from_spec(spec);spec.loader.exec_module(runtime)
import source
ref=local_eval.load_module('reference',R/'problems/muse_attention_decode/reference.py');ref._fa3_scheduler_metadata=lambda *a,**kw:None
runtime.LOCAL=True
runtime.SOURCES={f+k:(source.original(f,True) if k=='A' else source.source(f,True,clocks=k=='CLOCK')) for f in ('d','v') for k in ('A','M','CLOCK')}
cases=[dict(batch=B,q_len=L,ctx=80000,page=1,spread='agentic',dist='soft') for B,L in ((16,1),(32,1),(16,8),(32,8))]
cases+= [dict(batch=3,q_len=L,ctx=4000,page=64,kv_scale=.015625,dist='edges') for L in (1,8)]
rows=[]
for case in cases:
    d=ref.generate_input(**case,heads='tp8',layer='sliding',seed=29000011);coords=runtime.coordinates(d);family='d' if case['q_len']==1 else 'v'
    modes={'A':0,**source.MODES,'CLOCK':0}
    for name,mode in modes.items():
        module=family+('A' if name=='A' else 'CLOCK' if name=='CLOCK' else 'M')
        clk=torch.zeros((case['batch']*2*16,24),device='cuda',dtype=torch.int64)
        fn=lambda:runtime.attention(d,coords,module,mode,clk if name=='CLOCK' else None)
        fn();torch.cuda.synchronize();g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):out=fn()
        fresh=ref.generate_input(**case,heads='tp8',layer='sliding',seed=29104740);local_eval.copy_into(d,fresh);coords.copy_(runtime.coordinates(d))
        for _ in range(3):g.replay()
        torch.cuda.synchronize()
        if name in ('A','M','M2','C','CLOCK','W'):
            ok,msg=ref.check_implementation(d,ref.collect(d,out));assert ok,(case,name,msg)
        else:ok=None;msg='altered-work oracle: no TASK correctness claim'
        assert all(int(torch.count_nonzero(v))==0 for v in runtime.COUNTERS.values())
        rows.append(dict(case=case,name=name,TASK=ok,note=msg));print('PASS',case['batch'],case['q_len'],name,flush=True)
result=dict(rows=rows,TASK_passed=sum(r['TASK'] is True for r in rows),oracle_branches=sum(r['TASK'] is None for r in rows),sources={n:hashlib.sha256(s.encode()).hexdigest() for n,s in runtime.SOURCES.items()},standins=['decode DSM C1 keeps every key, not native D8/D4 geometry','verify two genuine WGs and nonspinning global last-WG combine instead of DSM','W is standalone ready-input only, never a producer-dependent candidate'])
(H/'admission.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS',result['TASK_passed'],'TASK',result['oracle_branches'],'oracle branches')
