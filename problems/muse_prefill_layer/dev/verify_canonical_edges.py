"""Canonical prefill readers: zero/all-local routes at verify-size boundaries."""
import sys,json
from pathlib import Path
sys.path[:0]=['tools','problems/muse_prefill_layer/dev']
import torch,triton
import local_eval as le
import compare_canonical as c
root=Path('problems/muse_prefill_layer')
g=le.load_module('canonical_edge_grouped',root/'cand_canonical_grouped.py')
state,raw=c.make_weights()
state['tiled_mod']=c.custom_host().load_module(str(root/'dev/moe_w4a16_canonical.cu'))
state['grouped_mod']=g._grouped_module()
records=[]
for T in (8,128,256):
    for mode in ('zero','all8'):
        fixture=c.make_fixture(T,seed=911)
        work=c.custom_workspace(fixture)
        def fill_routes(seed):
            gen=torch.Generator(device='cuda').manual_seed(seed)
            if mode=='zero':
                ids=128+torch.arange(8,device='cuda')[None,:].expand(T,-1)
            else:
                ids=torch.rand((T,32),device='cuda',generator=gen).topk(8,-1).indices
            fixture['ids'].copy_(ids)
            fixture['x'].normal_(generator=gen)
            fixture['probs'].uniform_(generator=gen)
            fixture['probs'].div_(fixture['probs'].sum(-1,keepdim=True))
        fill_routes(912)
        def call(fn):
            c.baseline_module('rs').route(fixture['ids'],work['counts'],work['offsets'],work['cursors'],
                work['pairs'],work['positions'],work['tasks'],0,128)
            c.slot_metadata[(triton.cdiv(T*8,256),)](fixture['ids'],work['positions'],work['offsets'],
                work['slot_e'],work['slot_pos'],SIZE=T*8,BLOCK=256)
            return fn(fixture['x'],fixture['probs'],work['slot_e'],work['slot_pos'],work['counts'],state)
        for name,fn in (('pipeline',c.custom_host().run_raw),('grouped',g.run_grouped_raw)):
            for _ in range(2):out=call(fn)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):out=call(fn)
            for seed in (913,914):
                fill_routes(seed);graph.replay();torch.cuda.synchronize()
                expected=c.reference(fixture,state,raw)
                error=c.error_metrics(out,expected)
                assert error['relative_l2']<.015 and error['worst_row']<.075,error
                records.append(dict(T=T,routes=mode,reader=name,seed=seed,**error))
            print(T,mode,name,'fresh graph FP32 reference PASS',flush=True)
            del out,graph
        del fixture,work
(root/'dev/canonical_edges.json').write_text(json.dumps(records,indent=2))
# Check unnormalized intermediate rows too: post-expert norm would cancel a missing
# uniform up/down scale, so the final normalized output alone cannot prove this.
fixture=c.make_fixture(8,seed=944,distribution='hot')
x=fixture['x'].float()
w13=c.nvfp4.dequantize(raw['w13'][0],raw['sf13'][0],1.)
h=x@w13.T
gate=(h[:,:4096]*state['gs13_gate'][0]).to(torch.bfloat16).float()
up=(h[:,4096:]*state['gs13_up'][0]).to(torch.bfloat16).float()
refinter=(torch.nn.functional.silu(gate)*up).to(torch.bfloat16)
w2=c.nvfp4.dequantize(raw['w2'][0],raw['sf2'][0],1.)
refdown=((refinter.float()@w2.T)*state['gs2'][0]).to(torch.bfloat16)
for name in ('grouped','pipeline'):
    work=c.workspace(fixture) if name=='grouped' else c.custom_workspace(fixture)
    if name=='grouped':
        c.run_baseline('rs128',fixture,state,work)
        positions=work['positions'].view(8,8)[:,0].long()
        inter=work['inter'][positions]
        down=work['y'][positions]
    else:
        c.run_custom(fixture,state,work)
        rows=work['rows'][:,0].contiguous()
        inter=torch.empty_like(refinter);down=torch.empty_like(refdown)
        c.custom_host().untile_rows_kernel[(8,)](work['inter'],rows,inter,D=4096,BNC=128)
        c.custom_host().untile_rows_kernel[(8,)](work['y'],rows,down,D=4096,BNC=128)
    errors=dict(inter=c.error_metrics(inter,refinter),down=c.error_metrics(down,refdown))
    assert all(v['relative_l2']<.015 and v['worst_row']<.075 for v in errors.values()),errors
    print(name,'unnormalized intermediate/down, unequal GSgate/up PASS',errors,flush=True)
    records.append(dict(T=8,routes='hot_raw_stages',reader=name,**errors))
(root/'dev/canonical_edges.json').write_text(json.dumps(records,indent=2))
