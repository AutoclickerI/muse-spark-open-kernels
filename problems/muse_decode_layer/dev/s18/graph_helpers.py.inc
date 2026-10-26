AB_CASES = ((1, 1, 2048), (1, 1, 0), (1, 8, 2048), (1, 8, 0), (16, 1, 2048), (16, 1, 0), (32, 1, 2048), (32, 1, 0),
            (16, 8, 2048), (16, 8, 0), (32, 8, 2048), (32, 8, 0))

AB_LOADS = {(1, 1): 'decode 1', (16, 1): 'decode 16', (32, 1): 'decode 32', (1, 8): 'verify 1', (16, 8): 'verify 16',
            (32, 8): 'verify 32'}

def ab_label(key):
    b, q, w = key
    return ('d' if q == 1 else 'v') + str(b) + ('S' if w else 'F')

def ab_namespace(overrides):
    """Every plain function of the host module rebound into a copy of its globals, then the overrides."""
    g = kernel_fn.__globals__
    v = dict(g)
    for k, f in g.items():
        if isinstance(f, types.FunctionType) and f.__globals__ is g:
            h = types.FunctionType(f.__code__, v, f.__name__, f.__defaults__, f.__closure__)
            h.__kwdefaults__ = f.__kwdefaults__
            h.__dict__.update(f.__dict__)
            v[k] = h
    v.update(overrides)
    return v

def ab_copy(dst, src):
    if isinstance(dst, torch.Tensor):
        if dst.shape != src.shape:raise RuntimeError(f'fixture shape {tuple(src.shape)} != {tuple(dst.shape)}')
        dst.copy_(src);return
    if isinstance(dst, dict):
        for k in dst:ab_copy(dst[k], src[k])
    elif isinstance(dst, (list, tuple)):
        for a, b in zip(dst, src):ab_copy(a, b)

def ab_graph(fn):
    for _ in range(2):fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):out = fn()
    graph.replay();torch.cuda.synchronize()
    return graph, out

def ab_tensors(out):
    if isinstance(out, torch.Tensor):return [out]
    if isinstance(out, (list, tuple)):return [t for o in out for t in ab_tensors(o)]
    return []

def ab_gate(got, want):
    """'' if close; else a short reason. Near-tie route flips (<= 2% of tokens) leave the float comparison."""
    got, want = ab_tensors(got), ab_tensors(want)
    flipped = None
    for a, b in zip(got, want):
        if a.dtype in (torch.int32, torch.int64):
            rows = (a.reshape(a.shape[0], -1).sort(-1).values != b.reshape(b.shape[0], -1).sort(-1).values).any(-1)
            n = int(rows.sum())
            if n > max(1, a.shape[0] // 50):return f'ids{n}/{a.shape[0]}'
            flipped = rows if flipped is None else flipped | rows
    worst = 0.
    for a, b in zip(got, want):
        if a.dtype in (torch.int32, torch.int64):continue
        if flipped is not None and a.shape[0] == flipped.shape[0] and bool(flipped.any()):a, b = a[~flipped], b[~flipped]
        if a.numel() == 0:continue
        af, bf = a.float(), b.float()
        if not bool(torch.isfinite(af).all()):return 'nonfinite'
        worst = max(worst, float((af - bf).norm() / bf.norm().clamp_min(1e-12)))
    return f'L2={worst:.4f}' if worst > .01 else ''

def ab_time(graphs, rounds):
    """name -> per-round us; every round replays each graph once in a rotating order, cold L2 before each."""
    names = list(graphs);times = {n: [] for n in names}
    for r in range(rounds):
        order = names[r % len(names):] + names[:r % len(names)]
        if r % 2:order = order[::-1]
        evs = []
        for n in order:
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            lab_cold_l2();s.record();graphs[n][0].replay();e.record();evs.append((n, s, e))
        torch.cuda.synchronize()
        for n, s, e in evs:times[n].append(s.elapsed_time(e) * 1000.)
    return times
