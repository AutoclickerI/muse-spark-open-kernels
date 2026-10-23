_X_FAMILIES = {}
_NEXT_PLAN = {}
_NEXT_READY = False
X_FAMILY = 'warp'
X_STAGES = 3
X_WAVES = 2


def _xext():
    family = X_FAMILY
    if family not in _X_FAMILIES:
        major, minor = torch.cuda.get_device_capability()
        arch = '90a' if major == 9 else f'{major}{minor}'
        src = G_CUDA if family == 'group' else X_CUDA
        _X_FAMILIES[family] = load_inline(
            name=f'muse_xqa_next_{family}_sm{arch}', cpp_sources=X_CPP, cuda_sources=src,
            functions=['xforward'], extra_cuda_cflags=['-O3', '-std=c++20',
                '--expt-relaxed-constexpr', f'-gencode=arch=compute_{arch},code=sm_{arch}',
                '-lineinfo'], verbose=False)
    return _X_FAMILIES[family]


def _next_set(v):
    global X_FAMILY, X_STAGES, X_WAVES
    X_FAMILY, X_STAGES, X_WAVES = v


def _xqa_verify(data):
    cfg = data['config']; qkv = data['qkv']; device = qkv.device
    batch, hq, hkv, ps = cfg['batch'], cfg['num_q_heads'], cfg['num_kv_heads'], cfg['page_size']
    window = cfg['window']; maxpages = data['page_table'].shape[1]
    maxkeys = min(maxpages * ps, window) if window else maxpages * ps
    tile = 64 if X_FAMILY == 'group' else 128
    ns = max(1, min(64, (maxkeys + tile - 1) // tile,
                   (_num_sms(device) * X_WAVES + batch * hkv - 1) // (batch * hkv)))
    out = torch.empty((batch, 8, hq * 64), dtype=torch.bfloat16, device=device)
    part = torch.empty((batch * hkv * ns * 64 * 68,), dtype=torch.float32, device=device)
    counts = _counters(batch * hkv, device)
    freq = _inv_freq(cfg['rope_theta'], device) if cfg['rope_theta'] else None
    _xext().xforward(qkv.data_ptr(), data['positions'].data_ptr(), data['cache_loc'].data_ptr(),
        data['page_table'].data_ptr(), data['seq_lens'].data_ptr(), data['k_cache'].data_ptr(),
        data['v_cache'].data_ptr(), data['k_scale'].data_ptr(), data['v_scale'].data_ptr(),
        freq.data_ptr() if freq is not None else 0, part.data_ptr(), counts.data_ptr(), out.data_ptr(),
        batch, hq, hkv, qkv.shape[-1], maxpages, ps, window, ns,
        cfg['sm_scale'] * cfg['qk_scale_factor'] * 1.4426950408889634,
        X_STAGES, 0, torch.cuda.current_stream(device).cuda_stream)
    return out


def _next_input(batch, window):
    device = torch.device('cuda')
    gen = torch.Generator(device=device).manual_seed(18751 + batch + window)
    maxpages = 86008 if batch > 1 else 80008
    seq = torch.full((batch,), 80000, dtype=torch.int32, device=device)
    if batch > 1:
        seq = (seq.float() * (1. + .075 * (2 * torch.rand(batch, device=device, generator=gen) - 1))).to(torch.int32)
    pt = torch.randperm(batch * maxpages, device=device, generator=gen, dtype=torch.int64).to(torch.int32).view(batch, maxpages)
    positions = (seq.long()[:, None] + torch.arange(8, device=device)).reshape(-1)
    loc = torch.gather(pt.long(), 1, positions.view(batch, 8)).reshape(-1)
    shape = (batch * maxpages, 1, 2, 64)
    k = torch.randn(shape, generator=gen, device=device, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    v = torch.randn(shape, generator=gen, device=device, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    qkv = torch.randn((batch, 8, 36 * 64), generator=gen, device=device, dtype=torch.bfloat16)
    return dict(qkv=qkv, positions=positions, cache_loc=loc, cache_seqlens=seq + 8,
        cu_seqlens_q=torch.arange(0, batch * 8 + 1, 8, device=device, dtype=torch.int32),
        scheduler_metadata=None, k_cache=k, v_cache=v, k_scale=torch.ones((), device=device),
        v_scale=torch.ones((), device=device), page_table=pt, seq_lens=seq,
        config=dict(batch=batch, q_len=8, num_q_heads=16, num_kv_heads=2, head_dim=64,
            page_size=1, window=window, rope_theta=10000. if window else None,
            qk_scale_factor=5.473, sm_scale=.125, kv_scale=1.))


def _next_diagnose():
    import statistics
    flush = torch.empty(1 << 28, dtype=torch.uint8, device='cuda')
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    variants = [('group', st, w) for st in (5, 7) for w in (1, 2, 4, 8)]
    variants += [('warp', st, w) for st in (3, 4) for w in (1, 2, 4, 8)]
    for batch, window in ((16, 0), (32, 0), (16, 2048), (32, 2048), (1, 0), (1, 2048)):
        data = _next_input(batch, window); timings = []
        expected = _fa3_verify(data)
        errors = []
        for variant in variants + ['fa3']:
            fn = _fa3_verify if variant == 'fa3' else (_next_set(variant) or _xqa_verify)
            actual = fn(data)
            # Synthetic prepare-only validation makes a descriptor/warpgroup
            # fault visible even if FA3 would otherwise win the timing sweep.
            err = ((actual.float() - expected.float()).square().sum() /
                   expected.float().square().sum().clamp_min(1e-20)).sqrt().item()
            errors.append((round(err, 6), variant))
            if err > .03 or not (err >= 0):
                raise RuntimeError(f'XQA_NEXT agreement failed: B{batch} window{window} {variant} L2={err}')
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph): out = fn(data)
            values = []
            for rep in range(3):
                flush.zero_(); start.record(); graph.replay(); end.record(); end.synchronize()
                values.append(start.elapsed_time(end) * 1000)
            timings.append((statistics.median(values), variant))
            del graph, out
        timings.sort(key=lambda r: r[0])
        best = next(v for us, v in timings if v != 'fa3')
        _next_set(best); graphs = []
        for fn in (_xqa_verify, _fa3_verify):
            fn(data); torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g): o = fn(data)
            graphs.append((g, o))
        sides = [[], []]
        for rep in range(6):
            for k in ((0, 1) if rep & 1 else (1, 0)):
                flush.zero_(); start.record(); graphs[k][0].replay(); end.record(); end.synchronize()
                sides[k].append(start.elapsed_time(end) * 1000)
        ratio = statistics.median([(sides[1][i] + sides[1][i+1]) /
            (sides[0][i] + sides[0][i+1]) for i in (0, 2, 4)])
        selected = best if ratio > 1.015 else 'fa3'
        _NEXT_PLAN[batch, bool(window)] = selected
        print('XQA_NEXT_DIAG', batch, window, 'us', [(round(us, 2), v) for us, v in timings], flush=True)
        print('XQA_NEXT_PAIR', batch, window, best, 'speedup', round(ratio, 4), 'selected', selected, flush=True)
        print('XQA_NEXT_AGREE', batch, window, 'worst', max(e for e, v in errors), flush=True)
        del graphs, actual, expected, data
    del flush
    torch.cuda.synchronize()


def kernel_fn(data: Inputs) -> torch.Tensor:
    cfg = data['config']
    if cfg['q_len'] == 1: return _DECODE_FN(data)
    selected = _NEXT_PLAN.get((cfg['batch'], bool(cfg['window'])))
    if selected == 'fa3': return _fa3_verify(data)
    if selected is None: selected = ('group', 5, 1 if cfg['window'] else 2)
    _next_set(selected)
    return _xqa_verify(data)


def prepare():
    global _NEXT_READY
    if not _NEXT_READY:
        if torch.cuda.get_device_capability()[0] == 9: _next_diagnose()
        _NEXT_READY = True


if torch.cuda.is_available(): prepare()
