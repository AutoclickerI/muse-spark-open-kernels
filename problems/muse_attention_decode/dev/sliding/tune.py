"""Time and gate split choices locally; local timing is not H200 evidence."""
import sys
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[4]
sys.path[:0]=[str(ROOT),str(ROOT/'problems/muse_attention_decode')]
from tools.local_eval import load_module, Timer, capture
mod=load_module('sliding_candidate',ROOT/'problems/muse_attention_decode/cand_sliding_singlelaunch.py')
ref=load_module('reference',ROOT/'problems/muse_attention_decode/reference.py')
ref._fa3_scheduler_metadata=lambda *args,**kwargs:None

timer=Timer()
for B in (1,16,32):
    for L in (1,8):
        data=ref.generate_input(B,80000,'tp8','sliding',731,q_len=L,dist='soft',spread='agentic',page=1)
        cfg=data['config'];raw=data['qkv'];freq=mod._inv_freq(cfg['rope_theta'],raw.device)
        out=torch.empty((B,L,1024),device=raw.device,dtype=torch.bfloat16)
        for ns in (1,2,4,8,16):
            parts=torch.empty((B*16*ns*L*68,),dtype=torch.float32,device=raw.device)
            counters=mod._counters(B*16,raw.device)
            def call(data):
                mod._window_ext().window_forward(raw.data_ptr(),data['positions'].data_ptr(),data['cache_loc'].data_ptr(),
                    data['page_table'].data_ptr(),data['seq_lens'].data_ptr(),data['k_cache'].data_ptr(),data['v_cache'].data_ptr(),
                    data['k_scale'].data_ptr(),data['v_scale'].data_ptr(),freq.data_ptr(),out.data_ptr(),parts.data_ptr(),
                    counters.data_ptr(),B,L,16,2,raw.shape[-1],data['page_table'].shape[1],1,2048,ns,0,
                    cfg['qk_scale_factor'],cfg['sm_scale'],torch.cuda.current_stream().cuda_stream)
                return out
            graph,_=capture(call,data)
            ok,msg=ref.check_implementation(data,ref.collect(data,out))
            assert ok,(B,L,ns,msg)
            us=sorted(timer.replay_us(graph)-timer.fixed_us for _ in range(9))[4]
            print(f'B={B} L={L} splits={ns} us={us:.2f} gate={ok}',flush=True)
        torch.cuda.empty_cache()
