"""Local split/tile timings, with a task gate on every measured setting."""
import json
import sys
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT),str(ROOT/'problems/muse_attention_decode')]
from tools.local_eval import load_module,Timer,capture

ref=load_module('reference',ROOT/'problems/muse_attention_decode/reference.py')
ref._fa3_scheduler_metadata=lambda *args,**kwargs:None
mod=load_module('verify8_candidate',ROOT/'problems/muse_attention_decode/cand_verify8_flashdecode.py')
timer=Timer();results=[]
for B in (1,16,32):
    data=ref.generate_input(B,80000,'tp8','full',731,q_len=8,page=1,dist='coherent',spread='agentic')
    reference=None
    for waves in (2,4,8):
        ns=min(128,max(1,(132*waves+B*2-1)//(B*2)))
        for kt in (64,128):
            for stages in (3,4):
                setting=(ns,stages,kt)
                def fn(data,setting=setting):
                    return mod.verify8_attention(data['qkv'],data,setting)
                graph,out=capture(fn,data)
                graph.replay();torch.cuda.synchronize()
                ok,msg=ref.check_implementation(data,ref.collect(data,out))
                assert ok,(B,setting,msg)
                us=sorted(timer.replay_us(graph)-timer.fixed_us for _ in range(5))[2]
                row=dict(B=B,waves=waves,splits=ns,kt=kt,stages=stages,ok=ok,us=us)
                results.append(row);print(row,flush=True)
                del graph,out
    torch.cuda.empty_cache()
(ROOT/'problems/muse_attention_decode/dev/flashdecode/tuning_local.json').write_text(json.dumps(results,indent=2)+'\n')
