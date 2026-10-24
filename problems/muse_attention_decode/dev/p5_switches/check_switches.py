"""Check each switch position against the task gate and its unchanged control."""
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT),str(ROOT/'problems/muse_attention_decode')]
from tools.local_eval import load_module,capture,copy_into

path=ROOT/'problems/muse_attention_decode/cand_p5_rescale_deferred.py'
m=load_module('p5_candidate',path)
ref=load_module('reference',ROOT/'problems/muse_attention_decode/reference.py')
ref._fa3_scheduler_metadata=lambda *a,**kw:None
rows=[]
for layer in ('sliding','full'):
    for dist in ('normal','coherent','edges'):
        data=ref.generate_input(3,4097,'tp8',layer,819,dist=dist,page=1)
        outputs=[];raws=[]
        for enabled in (False,True):
            m.Q1_CONDITIONAL=enabled
            g,out=capture(m.q1_attention,data)
            fresh=ref.generate_input(3,4097,'tp8',layer,820,dist=dist,page=1)
            copy_into(data,fresh);g.replay();torch.cuda.synchronize()
            ok,msg=ref.check_implementation(data,ref.collect(data,out));assert ok,msg
            outputs.append(out.clone());raws.append(ref.collect(data,out).clone())
            rows.append(dict(kind='q1',enabled=enabled,layer=layer,dist=dist,gate=ok,msg=msg))
        assert torch.equal(outputs[0],outputs[1]),('q1 output not bitwise',layer,dist)
        assert torch.equal(raws[0],raws[1]),('q1 output/cache not bitwise',layer,dist)
        print('Q1_BITWISE',layer,dist,True,flush=True)

for family in ('group','flashdecode'):
    for layer in (('sliding','full') if family=='group' else ('full',)):
        for dist in ('normal','coherent','edges'):
            data=ref.generate_input(2,4099,'tp8',layer,719,q_len=8,dist=dist,page=1,kv_scale=1./64.)
            actual=[]
            for enabled in (False,True):
                m.Q8_DEFERRED_DENOM=enabled
                def fn(data):
                    if family=='group':return m.xqa_group_attention(data,(3,5))
                    return m.verify8_attention(data['qkv'],data,(7,3,128))
                g,out=capture(fn,data)
                fresh=ref.generate_input(2,4099,'tp8',layer,720,q_len=8,dist=dist,page=1,kv_scale=1./64.)
                copy_into(data,fresh);g.replay();torch.cuda.synchronize()
                ok,msg=ref.check_implementation(data,ref.collect(data,out));assert ok,msg
                actual.append(out.clone())
                rows.append(dict(kind=family,enabled=enabled,layer=layer,dist=dist,gate=ok,msg=msg))
            relative=float((actual[0].float()-actual[1].float()).norm()/actual[0].float().norm().clamp_min(1e-12))
            assert relative<2.e-5,(family,layer,dist,relative)
            rows[-1]['control_relative_error']=relative
            rows[-1]['bitwise']=bool(torch.equal(actual[0],actual[1]))
            print('DEFER_CONTROL',family,layer,dist,relative,rows[-1]['bitwise'],flush=True)
        torch.cuda.empty_cache()

m.Q1_CONDITIONAL=True;m.Q8_DEFERRED_DENOM=True
adapter_path=ROOT/'problems/muse_decode_layer/dev/verify1_sliding_pick.py'
a=load_module('verify1_pick',adapter_path)
for count in (0,1,4):
    data=ref.generate_input(1,8192,'tp8','sliding',913,q_len=8,page=1,dist='soft')
    if count:
        torch.manual_seed(231+count)
        partial=torch.randn((count,8,2304),device='cuda',dtype=torch.float32)
        data['qkv']=partial.sum(0).to(torch.bfloat16).view(1,8,2304)
        qkv=(partial,count)
    else:qkv=data['qkv']
    g,out=capture(lambda data:a.attention_verify1_sliding(qkv,data).view(1,8,1024),data)
    if count:
        partial.add_(.03125);data['qkv']=partial.sum(0).to(torch.bfloat16).view(1,8,2304)
    else:data['qkv'].add_(.03125)
    g.replay();torch.cuda.synchronize()
    ok,msg=ref.check_implementation(data,ref.collect(data,out));assert ok,msg
    rows.append(dict(kind='verify1_adapter',projection_splits=count,gate=ok,msg=msg))
    print('VERIFY1_ADAPTER',count,ok,msg,flush=True)

result=dict(candidate_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            verify1_adapter_sha256=hashlib.sha256(adapter_path.read_bytes()).hexdigest(),checks=rows)
(ROOT/'problems/muse_attention_decode/dev/p5_switches/switches_local.json').write_text(json.dumps(result,indent=2)+'\n')
