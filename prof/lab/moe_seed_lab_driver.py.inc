import base64
import statistics
import time
import zlib
import torch

LAB_SEED_PATHS=((0,1,2),(0,2,2),(0,4,2),(0,2,1),(1,0,2),(2,0,1),(2,0,2))


def lab_seed_fn(data,state,choice):
    return seed_call(data,state,choice)


def lab_seed_measure(state,datasets,w,flush,paths):
    wants=[rr_reuse_fp32(d,w) for d in datasets]
    graphs={c:seed_graph(lambda d,c=c:lab_seed_fn(d,state,c),datasets[0]) for c in paths}
    worst={c:0. for c in paths};times={c:[] for c in paths};ratios={c:[] for c in paths}
    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    for d,want in zip(datasets,wants):
        for c,g in graphs.items():
            got=seed_refresh(g,d);torch.cuda.synchronize()
            err=rr_reuse_relative(got,want) if bool(torch.isfinite(got).all()) else float('inf')
            worst[c]=max(worst[c],err)
    for i in range(8):
        d=datasets[i%len(datasets)]
        for g in graphs.values():
            for key in ('hidden_states','topk_ids','topk_weights'):g[0][key].copy_(d[key])
        order=list(paths);order=order[i%len(order):]+order[:i%len(order)]
        if i%2:order=order[::-1]
        pair={}
        for c in order:
            flush.zero_();a.record();graphs[c][1].replay();b.record();b.synchronize();pair[c]=a.elapsed_time(b)*1000
        for c in order:times[c].append(pair[c]);ratios[c].append(pair[c]/pair[paths[0]])
    return {c:(statistics.median(times[c]),statistics.median(ratios[c]),worst[c]) for c in paths}


def lab(deadline,weights,state=None):
    device=next(v.device for v in weights.values() if torch.is_tensor(v))
    out=['META native seed portfolio: raw cold graph GPU us, 8 rotated orders, FP32 gate<=.01; R0 exact captured IDs with refreshed x/tw; S public generate_weights(seed,fields={}), real-router moe_tune fixture (not full attention front); fields={} because published benchmarks omit router_load. Physical grid fixed for graph; hot path active grid uses device row-block counts.']
    names=('NT1','NT2','NT4','NT2s1','hotNT1','hotNT2s1','hotNT2s2')
    out.append('ORDER '+','.join(names))
    raw=zlib.decompress(base64.b64decode(CAPTURED_R0));ms=(8,16,32,128,256)
    flush=torch.empty(128<<20,device=device,dtype=torch.uint8);rows=[]
    for i in range(12):
        if time.monotonic()>deadline-12:out.append(f'BUDGET before_seed={i}');break
        seed=5353+7919*i;w=rg_generate_weights(seed,device=device);w['post_norm_w']=w['post_expert_norm']
        bank=rv_prepare(w);state=dict(native=bank,hot=dict(bank,mod=hv__module()))
        for m in (128,256):
            fresh=[fx_make_fixture(w,m,9000+m+7919*j) for j in range(3)]
            if i==0:
                offset=8*sum(ms[:ms.index(m)]);ids=torch.tensor(list(raw[offset:offset+m*8]),device=device,dtype=torch.int32).view(m,8)
                ids=torch.where(ids<32,ids,-1);ds=[dict(d,topk_ids=ids.clone()) for d in fresh]
                vals=lab_seed_measure(state,ds,w,flush,LAB_SEED_PATHS)
                for name,c in zip(names,LAB_SEED_PATHS):
                    us,ratio,err=vals[c];out.append(f'T {name} M={m} R0 us={us:.1f} ratio={ratio:.4f} fp32={err:.5f} '+('PASS' if err<=.01 else 'REJECT'))
            vals=lab_seed_measure(state,fresh,w,flush,LAB_SEED_PATHS)
            counts=torch.bincount(fresh[0]['topk_ids'][fresh[0]['topk_ids']>=0],minlength=32)
            good=all(v[2]<=.01 for v in vals.values());v=[vals[c][0] for c in LAB_SEED_PATHS]
            out.append(f'S seed={seed} M={m} P={int(counts.sum())} E={int((counts>0).sum())} hot={int(counts.max())} us='+','.join(f'{us:.1f}' for us in v)+f' fp32={max(a[2] for a in vals.values()):.5f} '+('PASS' if good else 'REJECT'))
            if good:rows.append((m,seed,v))
    for m in (128,256):
        local=[r[2] for r in rows if r[0]==m]
        if local:
            for j,name in enumerate(names):
                vals=[r[j] for r in local];mean=statistics.mean(vals)
                out.append(f'AGG {name} M={m} seeds={len(vals)} mean={mean:.1f} min={min(vals):.1f} max={max(vals):.1f} max/min={max(vals)/min(vals):.4f} max/mean={max(vals)/mean:.4f}')
    return out
