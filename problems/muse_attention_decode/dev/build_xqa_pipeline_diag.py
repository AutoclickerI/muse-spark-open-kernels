"""Add a single prepare-time diagnostic/dispatch to the private K128 pipeline copy."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
base = (root / 'dev/xqa_pipeline128_candidate.py').read_text()
base += r'''

_PIPE_PLAN = {}
_PIPE_READY = False

def _pipe_set(v):
    global X_P8, X_STAGES, X_WAVES
    X_P8, X_STAGES, X_WAVES = v

def _pipe_input(batch, window):
    device = torch.device('cuda')
    gen = torch.Generator(device=device).manual_seed(5801 + batch + window)
    maxpages = 86008 if batch > 1 else 80008
    seq = torch.full((batch,), 80000, dtype=torch.int32, device=device)
    if batch > 1:
        seq = (seq.float() * (1.0 + .075 * (2 * torch.rand(batch, device=device, generator=gen) - 1))).to(torch.int32)
    pt = torch.arange(batch * maxpages, device=device, dtype=torch.int32).view(batch, maxpages)
    positions = (seq.long()[:, None] + torch.arange(8, device=device)).reshape(-1)
    loc = torch.gather(pt.long(), 1, positions.view(batch, 8)).reshape(-1)
    shape = (batch * maxpages, 1, 2, 64)
    k = torch.randn(shape, generator=gen, device=device, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    v = torch.randn(shape, generator=gen, device=device, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    qkv = torch.randn((batch, 8, 36 * 64), generator=gen, device=device, dtype=torch.bfloat16)
    cfg = dict(batch=batch, q_len=8, num_q_heads=16, num_kv_heads=2, head_dim=64,
               page_size=1, window=window, rope_theta=10000. if window else None,
               qk_scale_factor=5.473, sm_scale=.125, kv_scale=1.)
    return dict(qkv=qkv, positions=positions, cache_loc=loc, cache_seqlens=seq + 8,
                cu_seqlens_q=torch.arange(0, batch * 8 + 1, 8, device=device, dtype=torch.int32),
                scheduler_metadata=None, k_cache=k, v_cache=v, k_scale=torch.ones((), device=device),
                v_scale=torch.ones((), device=device), page_table=pt, seq_lens=seq, config=cfg)

def _pipe_diagnose():
    import statistics
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    flush = torch.empty(1 << 28, dtype=torch.uint8, device='cuda')
    # Full-specialized K128 with three gather stages is the measured winner;
    # retain three occupancy choices in this shipped diagnostic. FP8-P and
    # four-stage variants remain in the earlier sent diagnostic, while their
    # sm_90a builds are spill-prone on this private pipeline.
    variants = [(False, 3, w) for w in (2, 4, 8)]
    for batch, window in ((16, 0), (32, 0), (16, 2048), (32, 2048), (1, 0), (1, 2048)):
        data = _pipe_input(batch, window)
        timings = []
        for variant in variants + ['fa3']:
            fn = _fa3_verify if variant == 'fa3' else (_pipe_set(variant) or _xqa_verify)
            fn(data)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = fn(data)
            values = []
            for _ in range(5):
                flush.sum()
                start.record(); graph.replay(); end.record(); end.synchronize()
                values.append(start.elapsed_time(end) * 1000)
            timings.append((statistics.median(values), variant))
            del graph, out
        timings.sort(key=lambda r: r[0])
        best = next(v for us, v in timings if v != 'fa3')
        _pipe_set(best)
        pair = []
        graphs = []
        for fn in (_xqa_verify, _fa3_verify):
            fn(data); torch.cuda.synchronize(); graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph): out = fn(data)
            graphs.append((graph, out))
        side = [[], []]
        for rep in range(6):
            for k in ((0, 1) if rep & 1 else (1, 0)):
                flush.sum(); start.record(); graphs[k][0].replay(); end.record(); end.synchronize()
                side[k].append(start.elapsed_time(end) * 1000)
        ratios = [(side[1][i] + side[1][i+1]) / (side[0][i] + side[0][i+1]) for i in range(0, 6, 2)]
        speedup = statistics.median(ratios)
        selected = best if speedup > 1.03 else 'fa3'
        _PIPE_PLAN[(batch, bool(window))] = selected
        print('XQA_PIPE_DIAG', batch, window, 'us', [(round(us, 2), v) for us, v in timings], flush=True)
        print('XQA_PIPE_PAIR', batch, window, best, 'speedup', round(speedup, 4), 'selected', selected, flush=True)
        del graphs, data
    del flush
    torch.cuda.synchronize()

def kernel_fn(data: Inputs) -> torch.Tensor:
    cfg = data['config']
    if cfg['q_len'] == 1:
        return _DECODE_FN(data)
    selected = _PIPE_PLAN.get((cfg['batch'], bool(cfg['window'])))
    if selected == 'fa3':
        return _fa3_verify(data)
    if selected is not None:
        _pipe_set(selected)
    else:
        _pipe_set((False, 3, 8 if cfg['batch'] >= 32 and not cfg['window'] else 4))
    return _xqa_verify(data)

def prepare():
    global _PIPE_READY
    if not _PIPE_READY:
        if torch.cuda.get_device_capability()[0] == 9:
            _pipe_diagnose()
        _PIPE_READY = True

if torch.cuda.is_available():
    prepare()
'''
(root / 'cand_xqa_pipeline.py').write_text(base)
print('Built', root / 'cand_xqa_pipeline.py')
