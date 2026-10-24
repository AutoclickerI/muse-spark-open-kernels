"""Independent FP32 and fresh route-graph check of the final native hot-first portfolio."""
import importlib.util
import json
from pathlib import Path
import sys
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'problems/muse_decode_layer'))
def load(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
ref=load('public_ref',ROOT/'problems/muse_decode_layer/reference.py')
d=load('final_dropin',ROOT/'problems/muse_moe_w4a16/dev/v9_reuse/nt3_dropin.py')
cache=torch.load(ROOT/'runs/screen/v9_seed/cache.pt',weights_only=False)
checked=[]
for i in range(12):
    seed=5353+7919*i;w=ref.generate_weights(seed);w['post_norm_w']=w['post_expert_norm']
    bank=d.rv_prepare(w);state=dict(native=bank,hot=dict(bank,mod=d.hv__module()))
    for m in (128,256):
        data={k:v.cuda() if torch.is_tensor(v) else v for k,v in cache[f'{seed}:{m}']['data'].items()}
        want=d.rr_reuse_fp32(data,w)
        for choice in ((0,1,2),(0,2,2),(2,0,1),(2,0,2),(2,32,1),(3,0,2)):
            g=d.seed_graph(lambda x,c=choice:d.seed_call(x,state,c),data)
            worst=0.
            for j in range(4):
                # Change routing values on the same captured buffers; reference and kernel recompute.
                fresh=dict(data,topk_ids=torch.roll(data['topk_ids'],j,0),hidden_states=torch.roll(data['hidden_states'],-j,0),topk_weights=torch.roll(data['topk_weights'],j,0))
                if j==1:
                    fresh['topk_ids']=torch.where(torch.arange(m,device='cuda')[:,None]%3==0,-1,fresh['topk_ids'])
                elif j==2:
                    fresh['topk_ids']=torch.full_like(data['topk_ids'],-1);fresh['topk_ids'][0,0]=3
                elif j==3:fresh['topk_ids']=torch.full_like(data['topk_ids'],-1)
                target=want if j==0 else d.rr_reuse_fp32(fresh,w)
                got=d.seed_refresh(g,fresh);torch.cuda.synchronize();err=d.rr_reuse_relative(got,target)
                assert torch.isfinite(got).all() and err<.01,(seed,m,choice,j,err)
                worst=max(worst,err)
            checked.append(dict(seed=seed,M=m,choice=choice,fp32=worst))
    print('PASS seed',seed,flush=True)
(ROOT/'runs/screen/v9_seed/nt3_dropin_fp32.json').write_text(json.dumps(dict(rows=checked,graphs=len(checked),fresh_replays=len(checked)*4,max_fp32=max(x['fp32'] for x in checked)),indent=2)+'\n')
print('PASS',len(checked),'captured configs;',len(checked)*4,'fresh routing/input replays; independent FP32; no decoded weight bank retained',flush=True)
