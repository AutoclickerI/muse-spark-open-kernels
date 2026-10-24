"""Build a standalone H200 measurement from the exact serving bridge sources."""

import hashlib
from pathlib import Path

from build_moe_real_routing import namespace
from build_moe_v9_reuse import reference_generator

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent

DRIVER=r'''
import statistics as bridge_statistics
import time as bridge_time


def bridge_graph(fn, data):
    static={k:v.clone() if torch.is_tensor(v) else v for k,v in data.items()}
    fn(static)
    torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): out=fn(static)
    return graph,out,static


def bridge_timer(graph, flush, repeats=20):
    samples=[]
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    for _ in range(repeats):
        flush.sum()
        a.record();graph.replay();b.record();b.synchronize()
        samples.append(a.elapsed_time(b)*1000)
    return bridge_statistics.median(samples)


def lab(deadline, weights, host_state=None):
    lines=[]
    try:
        start=bridge_time.monotonic()
        device=weights['w13'].device
        reference=ma_prepare(weights)
        rb=reference['banks'][(0,32)]
        t,j=torch.arange(512,device=device),torch.arange(8,device=device)
        row13=torch.cat((t[:,None]*8+j,4096+t[:,None]*8+j),1)
        row2=torch.arange(4096,device=device).view(256,16)
        packed=[]
        for name,rows in [('w13',row13),('w2',row2)]:
            bank=torch.empty((32,rows.shape[0],36864),device=device,dtype=torch.uint8)
            for e in range(32): bank[e]=v9__repack(weights[name][e],weights[name+'_sf'][e],rows)
            packed.append(bank)
        state=dict(w13p=packed[0],w2p=packed[1],mod=v9__module(),
                   gs13_gate=weights['w13_gs'][:32].bfloat16().float(),
                   gs13_up=weights['w13_gs'][:32].bfloat16().float(),
                   gs2=weights['w2_gs'][:32].bfloat16().float(),
                   pnw=weights['post_norm_w'],local_start=0,
                   grid=torch.cuda.get_device_properties(device).multi_processor_count)
        copy=dict(rb)
        for key in ('w13','w2','w13_s','w2_s'): copy[key]=torch.empty_like(rb[key])
        target=dict(reference,banks={(0,32):copy})

        def convert(data=None):
            br_v9_to_marlin(packed[0],copy['w13'],copy['w13_s'],gated=True)
            br_v9_to_marlin(packed[1],copy['w2'],copy['w2_s'],gated=False)
            return copy['w2']

        convert()
        for key in ('w13','w2','w13_s','w2_s'):
            assert torch.equal(rb[key].view(torch.uint8),copy[key].view(torch.uint8)),key
        flush=torch.empty(128<<20,device=device,dtype=torch.uint8)
        cg,co,cs=bridge_graph(convert,{})
        repack_us=bridge_timer(cg,flush)
        cg.reset()
        lines.append(f'BRIDGE bytes_exact=1 workspace_bytes=905969664 cold_graph20_us={repack_us:.3g}')
        # Generator mirrors decode-layer router/bias/pooler distributions;
        # expert bytes above are the host problem's loaded NVFP4 weights.
        routing=rg_generate_weights(1234,device=str(device),router_seed=5353)

        def expert_v9(data):
            dense,mapping=v9__v9_call(data,state,return_dense=True)
            weights=torch.where(mapping>=0,data['topk_weights'],0)
            return ma_fused_post_expert_norm_weighted_sum(
                dense,weights,norm_weight=reference['norm_w'],eps=1e-8,weight_first=True)

        def bridged(data):
            convert()
            return ma_kernel_fn(data,target)

        for m in (8,32,128,256,257,512,753,1024,2048,16384):
            if bridge_time.monotonic()>deadline-25:
                lines.append(f'REJECT incomplete_before_M={m} budget_exhausted')
                return lines
            ds=[]
            for seed in (9001,16920,24839):
                q=8 if m<=256 else 1
                d=fx_make_fixture(routing,m,seed,mode='real',q_len=q)
                d['topk_ids']=torch.where(d['topk_ids']>=0,d['topk_ids'],255)
                ds.append({k:d[k] for k in ('hidden_states','topk_ids','topk_weights','config')})
            old,old_out,old_data=bridge_graph(lambda d:ma_kernel_fn(d,reference),ds[0])
            fn=expert_v9 if m<=256 else bridged
            new,new_out,new_data=bridge_graph(fn,ds[0])
            worst=0.0;rel=0.0
            for d in ds:
                for key in ('hidden_states','topk_ids','topk_weights'):
                    old_data[key].copy_(d[key]);new_data[key].copy_(d[key])
                old.replay();new.replay();torch.cuda.synchronize()
                diff=(new_out.float()-old_out.float()).abs()
                worst=max(worst,float(diff.max()))
                rel=max(rel,float(diff.norm()/old_out.float().norm().clamp_min(1e-20)))
                assert rel<=.015,(m,rel,worst)
            a=bridge_timer(old,flush);b=bridge_timer(new,flush)
            summary=fx_route_summary(dict(ds[0],topk_ids=torch.where(ds[0]['topk_ids']<32,ds[0]['topk_ids'],-1)))
            lines.append(f'M={m} producer={"V9" if m<=256 else "Marlin_bridge"} stock_us={a:.3g} new_us={b:.3g} '
                         f'speedup={a/b:.3g} maxdiff={worst:.3g} rel={rel:.3g} '
                         f'local_pairs={int(summary["pairs"])} max_expert={int(summary["max_expert"])}')
            old.reset();new.reset()
        lines.append(f'BRIDGE_SUMMARY PASS shapes=10 fresh=3 cold_graph_median=20 prepare_s={bridge_time.monotonic()-start:.3g}')
    except Exception as exc:
        lines.append('REJECT '+type(exc).__name__+': '+str(exc)[:400])
    return lines
'''


def main():
    parts=['"""Serving single-bank V9 + timed Marlin workspace; H200 admission diagnostic."""\n']
    for prefix,path in (
        ('ma_','problems/muse_moe_w4a16/baseline.py'),
        ('v9_','serving/sglang/python/sglang/srt/layers/muse_spark_moe.py'),
        ('br_','serving/sglang/python/sglang/kernels/ops/quantization/muse_v9_marlin.py'),
        ('fx_','problems/muse_decode_layer/moe_tune.py'),
    ):
        source=(ROOT/path).read_text()
        if prefix=='fx_': source=source[:source.index('\ndef _graph_time')]
        parts.append(f'# Source {path} sha256={hashlib.sha256((ROOT/path).read_bytes()).hexdigest()}\n')
        parts.append(namespace(source,prefix))
    parts.extend([reference_generator(),'\nimport torch\n',DRIVER])
    output=HERE/'shared_bank_bridge.py';output.write_text('\n'.join(parts))
    print(output.relative_to(ROOT),hashlib.sha256(output.read_bytes()).hexdigest())


if __name__=='__main__': main()
