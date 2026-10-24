import base64
import hashlib
import statistics
import time
import zlib

DC_NAMES = ('NT1', 'NT2', 'raw17', 'raw64', 'humming_masked', 'dg_w8')
DC_SEEDS = tuple(5353 + 7919 * i for i in range(12))


class DCLocalDGStandin:
    """Explicit local ABI check only; zero arithmetic is never gated or timed."""
    def __init__(self):
        self.calls = 0

    def m_grouped_bf16_gemm_nt_contiguous(self, a, b, c, ends, **kw):
        assert a.dtype == b.dtype == c.dtype == torch.bfloat16
        assert a.shape[1] == b.shape[2] and c.shape[1] == b.shape[1]
        assert b.shape[0] == 32 and ends.shape == (32,) and kw['use_psum_layout']
        self.calls += 1
        c.zero_()


class DCRawModule:
    def __init__(self, mod):
        self.moe_gemm = mod.raw17_gemm
        self.moe_gemm64 = mod.raw64_gemm


def dc_weights(seed, device):
    # Preserve all public weight/router draws and fixture seeds, adding independent
    # gate/up GS with a separate RNG so the exact public route tracks are unchanged.
    w = rg_generate_weights(seed, device=device)
    gen = torch.Generator(device=device).manual_seed(seed + 1000003)
    w['w13_gs_gate'] = w['w13_gs'].clone()
    w['w13_gs_up'] = w['w13_gs'] * torch.exp(.55 * torch.randn(32, generator=gen, device=device))
    w['post_norm_w'] = w['post_expert_norm']
    assert bool((w['w13_gs_gate'] != w['w13_gs_up']).all())
    return w


def dc_prepare(w, *, local_glue=False):
    dev = w['w13'].device
    native = rv_prepare(w)
    gs13 = torch.cat((w['w13_gs_gate'], w['w13_gs_up'])).float().contiguous()
    native['gs13'] = gs13
    st = dict(pre_expert=torch.ones(4096, device=dev), ones=torch.ones(32, device=dev),
              post_expert=(1 + w['post_norm_w'].float()).bfloat16(),
              post_expert_bf16=(1 + w['post_norm_w'].float()).bfloat16(),
              sms=torch.cuda.get_device_properties(dev).multi_processor_count,
              gs13_gate=w['w13_gs_gate'], gs13_up=w['w13_gs_up'], gs2=w['w2_gs'],
              w13_gs=gs13, w2_gs=w['w2_gs'], w13p=native['w13p'], w2p=native['w2p'])
    raw = dict(st, tiled_mod=DCRawModule(pf__grouped_module()))
    for key, gated in (('w13', True), ('w2', False)):
        b, sf = pf__raw_repack(w[key], w[key + '_sf'], gated)
        out = torch.empty_like(b)
        pf__encode_blob_fast[(b.numel() // 2048,)](b.view(torch.int32), sf, out.view(torch.int32), num_warps=4)
        raw['blob13' if gated else 'blob2'] = out
        raw['sfb13' if gated else 'sfb2'] = sf  # unchanged E4M3 bytes, layout only
    st['raw17'] = raw
    st['raw64'] = raw
    # With distinct stacked GS Humming's importer expands block scales to BF16.
    # Unit W13 GS retains E4M3; real FP32 GS are used in portfolio_swiglu instead.
    hum_weights = dict(w, w13_gs=torch.ones(32, device=dev))
    hum = pf__Experts(hum_weights, 0, 32)
    hum.gemm_type = pf_GemmType.GROUPED_MASKED
    hum.compile([128, 256])
    st['humming'] = hum
    # Check precisely the scale-bearing tensors that the timed producer reads.
    for key in ('w13', 'w2'):
        assert hum.weight(key)['weight_scale'].dtype == torch.float8_e4m3fn
        assert hum.weight(key)['weight'].element_size() <= 4
    if local_glue:
        st['_dg'] = DCLocalDGStandin()
    else:
        assert torch.cuda.get_device_capability(dev) == (9, 0), 'H200 lab only; use local_glue for local ABI screen'
        import deep_gemm
        st['_dg'] = deep_gemm
    return native, st


def dc_call(data, native, st, name):
    if name in ('NT1', 'NT2'):
        return rv_kernel_fn(data, dict(native, nt=1 if name == 'NT1' else 2))
    x = data['hidden_states']
    m = x.shape[0]
    counts = torch.empty(32, device=x.device, dtype=torch.int32)
    slots = torch.empty((32, m), device=x.device, dtype=torch.int32)
    pos = torch.empty((m, 8), device=x.device, dtype=torch.int32)
    mw_route[(32,)](data['topk_ids'], counts, slots, pos, m, 0, triton.next_power_of_2(m), num_warps=4)
    local = torch.where((data['topk_ids'] >= 0) & (data['topk_ids'] < 32), data['topk_ids'], -1).int()
    args = (x, data['topk_weights'], local, pos, counts)
    if name == 'raw17':
        return pf__raw_experts_tiled(*args, st['raw17'])
    if name == 'raw64':
        return pf__raw_experts_tiled64(*args, st['raw64'])
    if name == 'humming_masked':
        return pf__experts_humming_masked(*args, st)
    assert name == 'dg_w8'
    return pf__run_canonical_dg(*args, st, 'dg_w8')


def dc_oracles(ds, w):
    """Independent FP32 scratch oracle; no expanded bank survives this function."""
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        wants = [torch.zeros_like(d['hidden_states'], dtype=torch.float32) for d in ds]
        lut = torch.tensor((0., .5, 1., 1.5, 2., 3., 4., 6.), device=w['w13'].device)
        for e in range(32):
            active = [torch.where(d['topk_ids'] == e) for d in ds]
            if not any(t.numel() for t, _ in active):
                continue
            def dq(key):
                q = w[key][e]
                c = torch.stack((q & 15, q >> 4), -1).flatten(-2).long()
                return lut[c & 7] * torch.where(c & 8 != 0, -1., 1.) * w[key + '_sf'][e].float().repeat_interleave(16, -1)
            w13 = dq('w13')
            w2 = dq('w2') * w['w2_gs'][e]
            for d, out, (t, j) in zip(ds, wants, active):
                if t.numel() == 0:
                    continue
                h = d['hidden_states'][t].float() @ w13.t()
                g = h[:, :4096] * w['w13_gs_gate'][e]
                u = h[:, 4096:] * w['w13_gs_up'][e]
                a = (torch.nn.functional.silu(g) * u).bfloat16()
                y = a.float() @ w2.t()
                y *= 1 + w['post_norm_w'].float()
                y *= torch.rsqrt(y.square().mean(-1, keepdim=True) + 1.e-8)
                out.index_add_(0, t, y * d['topk_weights'][t, j, None])
        return [out.bfloat16() for out in wants]
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old


def dc_gate(graph, ds, wants):
    worst, row = 0., 0.
    for d, want in zip(ds, wants):
        graph.refresh(d)
        got = graph.replay()
        torch.cuda.synchronize()
        if not bool(torch.isfinite(got).all()):
            return float('inf'), float('inf')
        diff = got.float() - want.float()
        den = want.float().norm(dim=-1)
        worst = max(worst, float(diff.norm() / want.float().norm().clamp_min(1.e-12)))
        row = max(row, float((diff.norm(dim=-1) / den.clamp_min(den.mean() * .1).clamp_min(1.e-12)).max()))
    return worst, row


def dc_fixtures(w, m, r0=False):
    ds = [fx_make_fixture(w, m, 9000 + m + 7919 * j) for j in range(3)]
    if r0:
        raw = zlib.decompress(base64.b64decode(CAPTURED_R0))
        ms = (8, 16, 32, 128, 256)
        offset = 8 * sum(ms[:ms.index(m)])
        ids = torch.tensor(list(raw[offset:offset + m * 8]), device=w['w13'].device, dtype=torch.int32).view(m, 8)
        ids = torch.where(ids < 32, ids, -1)
        ds = [dict(d, topk_ids=ids.clone()) for d in ds]
    return ds


def dc_measure(native, st, ds, w, flush):
    wants = dc_oracles(ds, w)
    graphs, errors = {}, {}
    for name in DC_NAMES:
        try:
            graph = MWGraph(lambda d, n=name: dc_call(d, native, st, n), ds[0])
            err, row = dc_gate(graph, ds, wants)
            errors[name] = (err, row)
            # Every measured row must pass the independent FP32 gate first.
            if err <= .01 and row <= .05:
                graphs[name] = graph
        except Exception as exc:
            errors[name] = (float('inf'), float('inf'))
            print(f'DC_DIAG {name} {type(exc).__name__}: {str(exc).splitlines()[0][:160]}', flush=True)
    if not all(name in graphs for name in ('NT1', 'NT2')):
        raise ValueError(f'V9 controls failed FP32 gate: {errors}')
    times = {name: [] for name in graphs}
    ratios = {name: [] for name in graphs}
    a = torch.cuda.Event(enable_timing=True)
    b = torch.cuda.Event(enable_timing=True)
    for j in range(8):
        for graph in graphs.values():
            graph.refresh(ds[j % 3])
        order = list(graphs)
        order = order[j % len(order):] + order[:j % len(order)]
        if j % 2:
            order.reverse()
        pair = {}
        for name in order:
            flush.zero_()
            a.record()
            graphs[name].replay()
            b.record()
            b.synchronize()
            pair[name] = a.elapsed_time(b) * 1000.
        for name in pair:
            times[name].append(pair[name])
            ratios[name].append(pair[name] / pair['NT1'])
    return {name: (statistics.median(times[name]), statistics.median(ratios[name]), *errors[name])
            if name in graphs else (float('nan'), float('nan'), *errors[name]) for name in DC_NAMES}


def lab(deadline, weights, state=None):
    dev = next(v.device for v in weights.values() if torch.is_tensor(v))
    lines = ['META decode adapters; ORDER=' + ','.join(DC_NAMES),
             'META R0 exact captured IDs, refreshed x/tw, seed5353 bank; S public seeds5353+7919*i i=0..11 fields={} and identical fx_make_fixture seeds9000+M+7919*j j=0..2; independent upGS RNG seed+1000003 exp(.55*N); gateGS public draw',
             'META all measured rows FP32 relativeL2<=.01, max token-rowL2<=.05, finite; 3 fresh graph gates; 8 cold128MiB paired rounds, rotating/reversed order, GPU us include replay overhead; NVFP4 and E4M3 stored bytes stay packed/raw, DG expands each timed call; Humming W13 global scales applied in SwiGLU to avoid importer BF16 block scales']
    flush = torch.empty(128 << 20, device=dev, dtype=torch.uint8)
    rows = {m: {n: [] for n in DC_NAMES} for m in (128, 256)}
    nrows = 0
    for i, seed in enumerate(DC_SEEDS):
        if time.monotonic() > deadline - 15:
            lines.append(f'INCOMPLETE budget before_seed={seed}')
            break
        w = dc_weights(seed, dev)
        native, st = dc_prepare(w)
        for m in (128, 256):
            for is_r0 in ((True, False) if i == 0 else (False,)):
                ds = dc_fixtures(w, m, r0=is_r0)
                vals = dc_measure(native, st, ds, w, flush)
                counts = torch.bincount(ds[0]['topk_ids'][ds[0]['topk_ids'] >= 0], minlength=32)
                prefix = f'R0 M={m}' if is_r0 else f'S {seed} M={m}'
                us = ','.join(f'{vals[n][0]:.1f}' for n in DC_NAMES)
                ratios = ','.join(f'{vals[n][1]:.4f}' for n in DC_NAMES)
                errs = ','.join(f'{vals[n][2]:.4f}' for n in DC_NAMES)
                erows = ','.join(f'{vals[n][3]:.4f}' for n in DC_NAMES)
                status = ','.join('PASS' if vals[n][2] <= .01 and vals[n][3] <= .05 else 'REJECT' for n in DC_NAMES)
                lines.append(f'{prefix} P={int(counts.sum())} E={int((counts>0).sum())} hot={int(counts.max())} us={us} ratioNT1={ratios} L2={errs} row={erows} {status}')
                if not is_r0:
                    nrows += 1
                    for name in DC_NAMES:
                        if vals[name][2] <= .01 and vals[name][3] <= .05:
                            rows[m][name].append(vals[name][0])
        # No closures retain old banks once dc_measure returns. Reclaim per seed.
        del native, st, w, ds
        torch.cuda.empty_cache()
    for m in (128, 256):
        for name in DC_NAMES:
            vals = rows[m][name]
            mean = statistics.mean(vals) if vals else float('nan')
            lines.append(f'AGG {name} M={m} seeds={len(vals)}/12 mean_us={mean:.2f}' + (' COMPLETE' if len(vals) == 12 else ' INCOMPLETE_OR_REJECTED'))
    lines.append(f'END seed_shape_rows={nrows}/24 remaining_s={deadline-time.monotonic():.1f}')
    assert sum(len(line) + 1 for line in lines) <= 9000, 'lab output would be truncated'
    return lines
