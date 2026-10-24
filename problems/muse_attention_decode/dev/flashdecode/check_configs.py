"""Gate every producer configuration and the layer projection ABI locally."""
import hashlib
import json
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity,profile

ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT),str(ROOT/'problems/muse_attention_decode')]
from tools.local_eval import load_module,capture,copy_into

ref=load_module('reference',ROOT/'problems/muse_attention_decode/reference.py')
ref._fa3_scheduler_metadata=lambda *args,**kwargs:None
candidate=ROOT/'problems/muse_attention_decode/cand_verify8_flashdecode.py'
adapter_path=ROOT/'problems/muse_decode_layer/dev/verify8_flashdecode.py'
mod=load_module('verify8_candidate',candidate)
adapter=load_module('verify8_adapter',adapter_path)
checks=[]
for stages in (3,4):
    for kt in (64,128):
        for seed,ctx,heads,dist,page,scale,ns in (
            (1021,10,'tp8','normal',1,1.,16),
            (1022,4099,'tp8','soft',64,1./64.,7),
            (1023,6077,'tp8','coherent',1,1.,9),
            (1024,3097,'dp','edges',64,1.,3),
        ):
            data=ref.generate_input(2,ctx,heads,'full',seed,q_len=8,page=page,dist=dist,kv_scale=scale,spread='mixed')
            setting=(ns,stages,kt)
            def fn(data,setting=setting):
                return mod.verify8_attention(data['qkv'],data,setting)
            graph,out=capture(fn,data)
            fresh=ref.generate_input(2,ctx,heads,'full',seed+100,q_len=8,page=page,dist=dist,kv_scale=scale,spread='mixed')
            copy_into(data,fresh);graph.replay();torch.cuda.synchronize()
            ok,msg=ref.check_implementation(data,ref.collect(data,out))
            row=dict(stages=stages,kt=kt,ns=ns,ctx=ctx,heads=heads,dist=dist,scale=scale,graph=True,ok=ok,msg=msg)
            checks.append(row);print(row,flush=True)
            assert ok,row
            del graph,out,data,fresh
            torch.cuda.empty_cache()

for B in (1,16,32):
    data=ref.generate_input(B,8192,'tp8','full',317,q_len=8,page=1,dist='coherent',spread='agentic')
    for count in (0,1,4):
        torch.manual_seed(903+B+count)
        if count:
            partial=torch.randn((count,B*8,2304),device='cuda',dtype=torch.float32)
            data['qkv']=partial.sum(0).to(torch.bfloat16).view(B,8,2304)
            qkv=(partial,count)
        else:
            qkv=data['qkv']
        def fn(data,qkv=qkv):
            return adapter.attention_verify8(qkv,data).view(B,8,1024)
        graph,out=capture(fn,data)
        if not count:
            data['qkv'].add_(.03125)
        if count:
            partial.add_(.03125)
            data['qkv']=partial.sum(0).to(torch.bfloat16).view(B,8,2304)
        graph.replay();torch.cuda.synchronize()
        ok,msg=ref.check_implementation(data,ref.collect(data,out))
        row=dict(layer_B=B,projection_splits=count,graph=True,ok=ok,msg=msg)
        if B==1 and count in (0,4):
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                fn(data);torch.cuda.synchronize()
            names=[e.name for e in prof.events() if e.device_type==torch.autograd.DeviceType.CUDA]
            assert len(names)==2,names
            assert 'verify8_main' in names[0] and 'verify8_combine' in names[1],names
            row['kernels']=names
        checks.append(row);print(row,flush=True)
        assert ok,row
        del graph,out
    del data
    torch.cuda.empty_cache()
result=dict(candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
            adapter_sha256=hashlib.sha256(adapter_path.read_bytes()).hexdigest(),checks=checks)
(ROOT/'problems/muse_attention_decode/dev/flashdecode/configs_local.json').write_text(json.dumps(result,indent=2)+'\n')
