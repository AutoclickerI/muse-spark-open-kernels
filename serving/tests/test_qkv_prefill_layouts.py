"""Fresh graph admission and cold paired timing of the QKV layout hypotheses."""
import hashlib
import json
from pathlib import Path
import time

import torch
from sglang.srt.layers import muse_prefill_gemm as p

torch.manual_seed(312614)
w=torch.randn((2304,8192),device='cuda',dtype=torch.bfloat16)*.01
op=p.PrefillGemmPlan({'qkv_gate':w},tokens=(16384,),budget_s=0,
                     incumbent=lambda x,role:torch.mm(x,w.T))
rows=[];replays=0
for m in (753,3012,16384):
    x=torch.randn((m,8192),device='cuda',dtype=torch.bfloat16)
    banks=[torch.randn_like(x) for _ in range(2)]
    probe=torch.zeros_like(x);probe[:,127]=.5;banks.append(probe)
    flush=torch.zeros(128<<20,device='cuda',dtype=torch.uint8)
    configs=[('qkv_pad',width,nn,0) for width in p.PG_QKV_PAD for nn in (0,1)]
    configs += [('qkv_swap',0,nn,0) for nn in (0,1)]
    configs += [('qkv_swap_persistent',i,s) for i in (0,1,2) for s in (0,1)]
    configs += [('qkv_chunk',chunk,nn,0) for chunk in p.PG_QKV_CHUNKS if chunk<m and not m%chunk for nn in (-1,0,1)]
    with p._PgGraph(lambda:op._stock(x,'qkv_gate')) as base:
        for config in configs:
            with p._PgGraph(lambda c=config:op.run(x,'qkv_gate',c)) as trial:
                for fresh in banks:
                    x.copy_(fresh);base.graph.replay();trial.graph.replay()
                    assert p._pg_equal(trial.out,base.out),(m,config)
                    replays+=1
                ratio,old,new=p._pg_paired(base,trial,x,banks,flush,20)
                assert int(op.counts.count_nonzero())==0
                row=dict(m=m,config=config,ratio=ratio,stock_us=old,new_us=new)
                rows.append(row);print(json.dumps(row),flush=True)
assert p._PgGraph.active==0
report=dict(ok=True,cc=torch.cuda.get_device_capability(),fresh_admission_replays=replays,
            timed_pairs=len(rows)*20,rows=rows,
            module_sha256=hashlib.sha256(Path(p.__file__).read_bytes()).hexdigest())
Path('runs/screen/prefill_gemm/qkv_layouts.json').write_text(json.dumps(report,indent=2)+'\n')
for m in (753,3012,16384):print('BEST',m,min((r for r in rows if r['m']==m),key=lambda r:r['ratio']),flush=True)
