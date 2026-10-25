"""Check the new S11 producer wrapper against a CPU reconstruction of every ABI field."""
import importlib.util
import json
from pathlib import Path

import torch
from patch_route_meta import patch

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
src=REPO/'problems/muse_decode_layer/cand_r3_best7_s11_final.py'
candidate=HERE/'route_meta_candidate.py'
candidate.write_text(patch(src.read_text()))
spec=importlib.util.spec_from_file_location('route_meta_candidate',candidate)
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)


def expected(lids,start):
    t=lids.shape[0];flat=lids.flatten().cpu().tolist()
    lists=[[i for i,e in enumerate(flat) if e==expert] for expert in range(32)]
    counts=[len(row) for row in lists];offset=[0]
    for n in counts:offset.append(offset[-1]+n)
    blocks=[(n+7)//8 for n in counts]
    units13=[0];units2=[0]
    for n in blocks:units13.append(units13[-1]+n*512);units2.append(units2[-1]+n*256)
    pair=[i for row in lists for i in row];pos=[-1]*256
    for i,p in enumerate(pair):pos[p]=i
    token=[sum(e>=0 for e in flat[i*8:i*8+8]) if i<t else 0 for i in range(32)]
    return [1,t,start,len(pair)]+counts+offset+blocks+units13+units2+token+pair+[-1]*(256-len(pair))+pos


state={'router_bias':torch.randn(256,device='cuda')*.05,
       'pre_expert_w':torch.randn(4096,device='cuda')*.1+1,'cluster_route_meta_on':True}
mod.cluster_route_meta_prepare(state,'cuda')
rows=[]
for t in (1,8,16,32):
    for start in (0,96,224):
        p=torch.randn(6,t,4608,device='cuda')
        data={'local_start':torch.tensor([start],device='cuda',dtype=torch.int32)}
        def fn():return mod._best7_route((p,6),state,data,{},None)
        fn();torch.cuda.synchronize()
        gr=torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):out=fn()
        for replay in range(6):
            p.normal_();gr.replay();torch.cuda.synchronize()
            meta,arrive=state['cluster_route_meta'][t]
            want=expected(out[3],start)
            assert len(want)==711
            assert meta[:711].cpu().tolist()==want,(t,start,replay)
            assert int(arrive.item())==0
        rows.append(dict(tokens=t,local_start=start,meaningful_words=711,fresh_replays=6))
        print(rows[-1],flush=True)
(HERE/'route_meta_validation.json').write_text(json.dumps(rows,indent=2)+'\n')
print('PASS',len(rows)*6,'fresh graphs')
