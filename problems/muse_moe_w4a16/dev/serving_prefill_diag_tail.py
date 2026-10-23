# Diagnostic only. The scored dispatch below is cand_v9_phase's fixed portfolio.
import math
import statistics as _stats
import time as _clock


def _prefill_router(dev):
    gen = torch.Generator(device=dev).manual_seed(20261024)
    cpu = torch.Generator().manual_seed(5354)
    mean = torch.randn(8192, generator=cpu).to(dev)
    shared = torch.randn(4, 8192, generator=cpu).to(dev)
    router = .045 * torch.randn(256, 8192, generator=gen, device=dev)
    router += .07 * torch.randn(256, 1, generator=gen, device=dev) * mean / math.sqrt(8192)
    return dict(mean=mean, shared=shared, router=router,
        bias=.055 * torch.randn(256, generator=gen, device=dev),
        pooler=(torch.randn(4096, 8192, generator=gen, device=dev) / math.sqrt(8192)).bfloat16(),
        norm1=(.1 * torch.randn(8192, generator=gen, device=dev)).bfloat16(),
        norm2=(.1 * torch.randn(4096, generator=gen, device=dev)).bfloat16())


def _prefill_input(m, seed, dev, w):
    gen = torch.Generator(device=dev).manual_seed(seed)
    coef = torch.randn(m, 4, generator=gen, device=dev)
    stream = .7 * w['mean'] + math.sqrt(.9 / 4) * (coef @ w['shared'])
    stream += math.sqrt(.1) * torch.randn(m, 8192, generator=gen, device=dev)
    h = stream * torch.rsqrt(stream.square().mean(-1, keepdim=True) + 1e-5)
    h = (h * (1 + w['norm1'].float())).bfloat16()
    scores = torch.sigmoid((h.float() @ w['router'].T) * .1767766952966369)
    ids = (scores + w['bias']).topk(8, -1).indices
    picked = scores.gather(1, ids)
    tw = (picked / picked.sum(-1, keepdim=True)).contiguous()
    # Same busiest-EP-rank rotation as the decode-layer realistic routing fixture.
    busiest = torch.bincount((ids // 32).flatten(), minlength=8).argmax()
    ids = ((ids - busiest * 32) % 256).int().contiguous()
    x = h @ w['pooler'].T
    xf = x.float()
    x = (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-5)
         * (1 + w['norm2'].float())).bfloat16()
    return dict(hidden_states=x, topk_ids=ids, topk_weights=tw,
        config=dict(bs=m, d_hidden=4096, d_expert=4096, n_experts=256,
                    local_start=0, local_count=32, top_k=8, sharding='ep8'))


def _serving_forward(data, state, norm):
    dense, mapping = _serving_dense(state, data['hidden_states'],
                                   data['topk_weights'], data['topk_ids'])
    return fused_post_expert_norm_weighted_sum(
        dense, data['topk_weights'], norm_weight=norm, eps=1e-8,
        weight_first=True, slot_expert_ids=mapping)


def _prefill_compare(a, b):
    af, bf = a.float(), b.float()
    diff = af - bf
    rel = float(torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(bf).clamp_min(1e-9))
    # Zero-local rows must be zero; nonzero rows have a separate error guard.
    row = torch.linalg.vector_norm(diff, dim=1) / torch.linalg.vector_norm(bf, dim=1).clamp_min(1e-9)
    worst = float(row.max())
    maximum = float(diff.abs().max())
    if not bool(torch.isfinite(a).all()) or rel > .015 or worst > .03:
        raise ValueError(f'output rel={rel:.6g} row={worst:.6g} max={maximum:.6g}')
    return rel, worst, maximum


def _prefill_graph(fn, data):
    for _ in range(2):
        fn(data)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn(data)
    graph.replay()
    torch.cuda.synchronize()
    return graph, out


def _prefill_diagnostic(weights, native, started):
    dev = weights['w13'].device
    results = []
    stock = _marlin_prepare(weights)
    serving = dict(native, mod=_serving_module(), local_start=0,
        gs13_gate=weights['w13_gs'][:32].float().contiguous(),
        gs13_up=weights['w13_gs'][:32].float().contiguous())
    router = _prefill_router(dev)
    flush = torch.empty(1 << 30, dtype=torch.uint8, device=dev)
    for m in (257, 320, 384, 448, 512, 1024, 2048):
        if _clock.monotonic() - started > 210:
            results.append(f'M={m} REJECT budget; marlin=NA v9=NA winner=NONE')
            continue
        try:
            data = _prefill_input(m, 17001, dev, router)
            fns = dict(marlin=lambda d: _marlin_forward(d, stock),
                       v9=lambda d: _serving_forward(d, serving, stock['norm_w']))
            eager = {name: fn(data) for name, fn in fns.items()}
            rel, row, maximum = _prefill_compare(eager['v9'], eager['marlin'])
            captures = {name: _prefill_graph(fn, data) for name, fn in fns.items()}
            graph_rel, _, _ = _prefill_compare(captures['v9'][1], captures['marlin'][1])
            # The same fresh router stream feeds both sides; L2 is cold for every
            # replay. Alternating order guards against drifting GPU conditions.
            timings = {name: [] for name in fns}
            err = max(rel, graph_rel)
            for rep in range(20):
                if _clock.monotonic() - started > 230:
                    raise TimeoutError('prepare diagnostic exceeded 230s guard')
                fresh = _prefill_input(m, 18000 + rep, dev, router)
                for key in ('hidden_states', 'topk_ids', 'topk_weights'):
                    data[key].copy_(fresh[key])
                order = ('marlin', 'v9') if rep % 2 == 0 else ('v9', 'marlin')
                for name in order:
                    flush.zero_()
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    begin.record()
                    captures[name][0].replay()
                    end.record()
                    end.synchronize()
                    timings[name].append(begin.elapsed_time(end) * 1000)
                r, wr, mx = _prefill_compare(captures['v9'][1], captures['marlin'][1])
                err, row, maximum = max(err, r), max(row, wr), max(maximum, mx)
            us = {name: _stats.median(samples) for name, samples in timings.items()}
            counts = torch.bincount(data['topk_ids'][data['topk_ids'] < 32].long(), minlength=32)
            winner = min(us, key=us.get)
            results.append(f'M={m} PASS marlin={us["marlin"]:.3g}us v9={us["v9"]:.3g}us '
                f'winner={winner} ratio={us["marlin"]/us["v9"]:.3g} n=20 '
                f'local={int(counts.sum())} distinct={int((counts>0).sum())} hot={int(counts.max())} '
                f'rel={err:.3g} row={row:.3g} max={maximum:.3g}')
            del captures, eager, data, fresh
        except Exception as exc:
            results.append(f'M={m} REJECT marlin=NA v9=NA winner=NONE {type(exc).__name__}: {str(exc)[:180]}')
    return results


def prepare(weights):
    started = _clock.monotonic()
    native = _prepare_v9(weights, 'native')
    state = dict(native=native, humming=_prepare_humming(weights),
        dispatch={1:'v9', 8:'v9', 16:'v9', 32:'v9', 128:'v9', 256:'wide',
                  753:'wide', 3012:'wide', 16384:'wide'})
    try:
        lines = _prefill_diagnostic(weights, native, started)
    except Exception as exc:
        lines = [f'ALL REJECT marlin=NA v9=NA winner=NONE {type(exc).__name__}: {str(exc)[:240]}']
    # LAST output in prepare: small enough to survive the receipt's 16k tail.
    verdict = 'PASS' if len(lines) == 7 and all(' PASS ' in line for line in lines) else 'REJECT'
    print(f'SERVING_PREFILL_GATE status={verdict} elapsed={_clock.monotonic()-started:.3g}s '
          'EP8local32 coldL2=1GiB graphs=1 paired=20 router=decode_layer5354', flush=True)
    for line in lines:
        print('SERVING_PREFILL ' + line, flush=True)
    return state
