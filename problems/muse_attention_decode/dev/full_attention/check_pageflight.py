"""Require page-flight scheduling to preserve exact attention/cache results on full fixtures."""
import hashlib
import importlib.util
import json
import linecache
from pathlib import Path
import torch

H = Path(__file__).resolve().parent
def load(name):
    p=H/(name+'.py');spec=importlib.util.spec_from_file_location(name,p)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
old,new=load('q1_runtime'),load('q1_pageflight')
s=(H/'phases_lab.py').read_text();common=s.split('\nSOURCES=',1)[0];ns={}
linecache.cache['<full-inputs>']=(len(common),None,common.splitlines(True),'<full-inputs>')
exec(compile(common,'<full-inputs>','exec'),ns)
rows=[]
for b,ctx,page,scale in ((1,257,64,1/64),(16,4096,1,1),(32,80000,1,1),(3,33,64,1)):
    data=ns['generate_input'](batch=b,ctx=ctx,heads='tp8',layer='full',q_len=1,page=page,dist='coherent',spread='agentic' if b>1 else 'even',kv_scale=scale,seed=61791)
    a,plan=old.q1_attention(data)
    hk=data['config']['num_kv_heads'];slots=data['cache_loc'].reshape(-1)
    ca=tuple(data[k].view(-1,hk,64)[slots].clone() for k in ('k_cache','v_cache'))
    z,_=new.q1_attention(data,plan=plan)
    torch.cuda.synchronize()
    assert torch.equal(a,z),(b,ctx,float((a.float()-z.float()).norm()))
    assert all(torch.equal(v,data[k].view(-1,hk,64)[slots]) for k,v in zip(('k_cache','v_cache'),ca))
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):z,_=new.q1_attention(data,plan=plan)
    g.replay();torch.cuda.synchronize();assert torch.equal(a,z)
    want=ns['gate_prepare'](data);ok,msg,err=ns['gate_cached'](data,z,want);assert ok,msg
    rows.append(dict(batch=b,ctx=ctx,page=page,scale=scale,plan=plan,bitwise=True,error=err))
    print('PASS',rows[-1],flush=True)
    del data,a,z,g,want
    torch.cuda.empty_cache()
(H/'pageflight_validation.json').write_text(json.dumps(dict(passed=True,source_sha256=hashlib.sha256((H/'q1_pageflight.py').read_bytes()).hexdigest(),rows=rows),indent=2)+'\n')
