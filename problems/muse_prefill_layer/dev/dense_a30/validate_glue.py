"""Execute the added prefill dense glue on SM120 with the real Lt ABI.

The SM90-only prepare decision is forced; tensor operations still execute on
this device. Independent fresh torch products check each selected Lt/persistent
path, followed by graph replay and route-map checks. No judge calls.
"""
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'problems/muse_prefill_layer'))
import reference as ref

candidate = ROOT / 'problems/muse_prefill_layer/cand_dense_router_a30.py'
spec = importlib.util.spec_from_file_location('dense_a30', candidate)
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


def relative(got, want):
    return float((got.float()-want.float()).norm()/want.float().norm().clamp_min(1e-12))


def main():
    torch.manual_seed(308)
    weights = ref.generate_weights(ref.WEIGHTS_SEED)
    # Force the complete Hopper prepare Python branch. The existing base
    # stage tuners are already screened; only the new call/ABI is substituted.
    calls = []
    original = p.prefill_gemm_prepare
    def plan_standin(w, **kw):
        assert kw['tokens'] == (16384,) and kw['budget_s'] == 75
        assert all(k in w for k in ('pooler','router','unpooler','qkv_gate','o'))
        calls.append('prepare')
        return original(w, tokens=kw['tokens'], budget_s=0, incumbent=kw['incumbent'])
    with patch.object(p, 'IS_HOPPER', True), \
         patch.object(p, '_tune_score_dense', lambda *a: None), \
         patch.object(p, '_tune_score_experts', lambda *a: None), \
         patch.object(p, '_tune_score_attention', lambda *a: None), \
         patch.object(p, 'prefill_gemm_prepare', plan_standin):
        state = p.prepare(weights)
    assert calls == ['prepare'] and state['pg'] is not None
    plan = state['pg']
    # budget=0 factory released losing mixed layouts; restore them only for
    # explicitly forcing all trial branches in this test.
    plan.bank = torch.cat((plan.weights['pooler'], plan.hi), 0)
    plan.lt_bank = torch.cat((plan.bank, plan.lo), 0)
    # This wrapper has host-only Lt calls; compile it for SM90a and actually
    # exercise its ABI on SM120 rather than mocking its argument conventions.
    p._pg_lib = p._pg_native(arch='9.0a')
    records = []
    for m in (753, 3012, 16384):
        for role in ('pooler','unpooler','qkv_gate','o','router','pool_router'):
            k = weights['pooler' if role == 'pool_router' else role].shape[1]
            x = torch.randn((m,k), device='cuda', dtype=torch.bfloat16)
            if role == 'pool_router':
                def want():
                    pair = torch.mm(x, state['router_cat'].T, out_dtype=torch.float32)
                    return torch.mm(x, weights['pooler'].T), pair[:,:256]+pair[:,256:]
            elif role == 'router':
                def want():
                    pair = torch.mm(x, state['router_cat'].T, out_dtype=torch.float32)
                    return pair[:,:256]+pair[:,256:]
            else:
                def want():
                    return torch.mm(x, weights[role].T)
            for config in (('lt',0,0), ('lt',1,0), ('persistent',0,0), ('persistent',0,1)):
                worst = 0.
                with p._PgGraph(lambda: plan.run(x, role, config)) as graph:
                    for _ in range(2):
                        x.normal_(); graph.graph.replay()
                        reference = want()
                        pairs = zip(graph.out,reference) if isinstance(reference,tuple) else [(graph.out,reference)]
                        for got, expected in pairs:
                            assert torch.isfinite(got).all()
                            err = relative(got,expected)
                            assert err <= (.01 if got.dtype == torch.bfloat16 else 1e-4), (m,role,config,err)
                            worst = max(worst,err)
                records.append(dict(m=m,role=role,config=config,relative_l2=worst))
        # Exercise both route consumers and every new dense dispatch helper.
        h = torch.randn((m,8192),device='cuda',dtype=torch.bfloat16)
        local_start = torch.zeros((),device='cuda',dtype=torch.int32)
        for config in (('stock',),('lt',0,0),('lt',1,0),('persistent',0,0)):
            plan.plans[(m,'pool_router')] = config
            with p._PgGraph(lambda:p._pg_layer_group(h,state,local_start,'mixed')) as graph:
                for _ in range(2):
                    h.normal_(); graph.graph.replay()
                    pooled,ids,probs,slot_e,slot_pos,counts = graph.out
                    pair=torch.mm(h,state['router_cat'].T,out_dtype=torch.float32)
                    biased=torch.sigmoid((pair[:,:256]+pair[:,256:])*.1767766952966369)+state['router_bias']
                    values,indices=torch.topk(biased,9,dim=1)
                    chosen=ids.long()
                    assert (biased.gather(1,chosen)>=values[:,7:8]-.05).all()
                    required=values[:,:8]>values[:,8:9]+.04
                    included=(indices[:,:8,None]==chosen[:,None,:]).any(2)
                    assert (included|~required).all()
                    assert torch.isfinite(probs).all() and torch.allclose(probs.sum(1),torch.ones(m,device='cuda'),atol=1e-5)
                    assert int(counts.sum()) == int((slot_e>=0).sum())
                    for e in range(32):
                        positions=slot_pos[slot_e==e].sort().values
                        assert torch.equal(positions,torch.arange(len(positions),device='cuda',dtype=torch.int32))
                    assert relative(pooled,torch.mm(h,weights['pooler'].T))<=.01
            records.append(dict(m=m,role='routing',config=config,ok=True))
    for name in ('qkv_gate','o','unpooler'):
        k=weights[name].shape[1]
        x=torch.randn((16384,k),device='cuda',dtype=torch.bfloat16)
        key='qkv' if name=='qkv_gate' else name
        plan.plans[(16384,name)]=('stock',)
        assert p._pg_equal(p._pg_dense_run(x,state,name,key),torch.mm(x,weights[name].T))
    with patch.object(p, '_PgGraph', side_effect=RuntimeError('injected graph failure')):
        p._pg_tune_dense_groups(state, p.time.monotonic()+5)
    assert state['pg_groups'][16384]=='a30'
    assert p._PgGraph.active == 0
    report=dict(ok=True,device=torch.cuda.get_device_name(),sm90a_lt_wrapper=True,
                forced_prepare=True,consumer_failure_rejected=True,fresh_graph_replays=2*len(records),records=records)
    out=ROOT/'runs/screen/dense_a30_glue.json'
    out.write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='records'}),flush=True)


if __name__=='__main__':
    main()
