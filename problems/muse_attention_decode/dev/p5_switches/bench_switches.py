"""Paired cold-cache local switch comparison; H200 promotion still requires its receipt."""
import hashlib
import json
import statistics
import sys
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT),str(ROOT/'problems/muse_attention_decode')]
from tools.local_eval import load_module,capture,Timer
p=ROOT/'problems/muse_attention_decode/cand_p5_rescale_deferred.py'
m=load_module('p5_candidate',p)
ref=load_module('reference',ROOT/'problems/muse_attention_decode/reference.py')
ref._fa3_scheduler_metadata=lambda *a,**kw:None
clock=Timer();rows=[]
for family,B,L,layer in [('q1',1,1,'sliding'),('q1',16,1,'full'),('q1',32,1,'full'),
                         ('group',16,8,'sliding'),('group',32,8,'sliding'),
                         ('flashdecode',16,8,'full'),('flashdecode',32,8,'full')]:
    data=ref.generate_input(B,80000,'tp8',layer,3173,q_len=L,page=1,spread='agentic',
                            dist='soft' if layer=='sliding' else 'coherent')
    graphs=[];outs=[]
    for enabled in (False,True):
        m.Q1_CONDITIONAL=enabled;m.Q8_DEFERRED_DENOM=enabled
        def fn(data):
            if family=='q1':return m.q1_attention(data)
            if family=='group':return m.xqa_group_attention(data)
            return m.verify8_attention(data['qkv'],data)
        graph,out=capture(fn,data);graph.replay();torch.cuda.synchronize()
        ok,msg=ref.check_implementation(data,ref.collect(data,out));assert ok,msg
        graphs.append(graph);outs.append(out)
    samples=[[],[]]
    for rep in range(6):
        for side in ((0,1) if rep%2==0 else (1,0)):
            samples[side].append(clock.replay_us(graphs[side])-clock.fixed_us)
    control,new=map(statistics.median,samples)
    relative=float((outs[0].float()-outs[1].float()).norm()/outs[0].float().norm().clamp_min(1e-12))
    assert relative<2.e-5,(family,B,layer,relative)
    row=dict(family=family,batch=B,q_len=L,layer=layer,control_us=control,switched_us=new,
             ratio=new/control,gate=True,control_relative_error=relative,bitwise=bool(torch.equal(outs[0],outs[1])))
    rows.append(row);print(row,flush=True)
    del graphs,outs,data
    torch.cuda.empty_cache()
(ROOT/'problems/muse_attention_decode/dev/p5_switches/timings_local.json').write_text(json.dumps(dict(
    candidate_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),rows=rows),indent=2)+'\n')
