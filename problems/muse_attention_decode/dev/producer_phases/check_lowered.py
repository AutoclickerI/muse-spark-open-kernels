"""Compare the emulated-Hopper producer structure with explicit stage widening."""
import hashlib,json,sys
from pathlib import Path
import torch
from force_lowering import forced_lowering
ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'problems/muse_attention_decode')]
from local_eval import load_module
m=load_module('forced_lowered_producer',ROOT/'prof/lab/verify8_producer_phases.py')
m.PHASE_CUDA=forced_lowering(m.PHASE_CUDA)
rows=[]
flush=torch.empty(128<<20,dtype=torch.uint8,device='cuda')
for B in (16,32):
    d=m.generate_input(B,80000,'tp8','full',967+B,q_len=8,page=1,spread='agentic',dist='coherent')
    for wide,setting in ((False,(17,3,128)),(True,(17,3,128)),(True,(34,3,64))):
        us,out,g=m.timed(lambda x:m.phase_run(x,setting,False,wide)[0],d,flush,rounds=7)
        expected=m.gate_prepare(d);ok,msg,err=m.gate_cached(d,out,expected);assert ok,msg
        rows.append(dict(B=B,wide=wide,setting=setting,local_us=us,gate=ok,error=err))
        print(rows[-1],flush=True)
result=dict(note='sm120 with explicit sm90a FP8-mma lowering; not FA3 or H200 timing',
    cuda_sha256=hashlib.sha256(m.PHASE_CUDA.encode()).hexdigest(),checks=rows)
(ROOT/'problems/muse_attention_decode/dev/producer_phases/lowered_local.json').write_text(json.dumps(result,indent=2)+'\n')
