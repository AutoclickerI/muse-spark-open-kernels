"""Pack the spill-free 128-key XQA family and FA3 into one H200 check."""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
base = (root / 'dev/xqa_w8_candidate.py').read_text()
base += r'''

_W8_PLAN = {}
_W8_READY = False

def _w8_set(v):
    global X_P8, X_STAGES, X_WAVES
    X_P8, X_STAGES, X_WAVES = v

def _w8_input(batch, window):
    device = torch.device('cuda')
    gen = torch.Generator(device=device).manual_seed(1717 + batch + window)
    maxpages = (86000 if batch > 1 else 80000) + 8
    if batch > 1:
        seq = (80000 * (1 + .075 * (2 * torch.rand(batch, device=device, generator=gen) - 1))).to(torch.int32)
    else:
        seq = torch.full((batch,), 80000, device=device, dtype=torch.int32)
    pt = torch.randperm(batch * maxpages, device=device, generator=gen).to(torch.int32).view(batch, maxpages)
    positions = (seq.long()[:, None] + torch.arange(8, device=device)).reshape(-1)
    loc = torch.gather(pt.long(), 1, positions.view(batch, 8)).reshape(-1)
    shape = (batch * maxpages, 1, 2, 64)
    k = torch.randn(shape, dtype=torch.bfloat16, device=device, generator=gen).to(torch.float8_e4m3fn)
    v = torch.randn(shape, dtype=torch.bfloat16, device=device, generator=gen).to(torch.float8_e4m3fn)
    qkv = torch.randn((batch, 8, 36 * 64), device=device, dtype=torch.bfloat16, generator=gen)
    cfg = dict(batch=batch, q_len=8, num_q_heads=16, num_kv_heads=2, head_dim=64,
               page_size=1, window=window, rope_theta=10000. if window else None,
               qk_scale_factor=5.473, sm_scale=.125, kv_scale=1.)
    return dict(qkv=qkv, positions=positions, cache_loc=loc, cache_seqlens=seq + 8,
                cu_seqlens_q=torch.arange(0, batch * 8 + 1, 8, device=device, dtype=torch.int32),
                scheduler_metadata=None, k_cache=k, v_cache=v, k_scale=torch.ones((), device=device),
                v_scale=torch.ones((), device=device), page_table=pt, seq_lens=seq, config=cfg)

def _w8_diagnose(hopper):
    import statistics
    # Both wider rings are real kernels on Hopper. sm_120 can run stage 3.
    stages = (3, 4, 5) if hopper else (3,)
    variants = [(p8, st, waves) for p8 in (False, True)
                for st in stages for waves in (1, 2, 4, 8)]
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    flush = torch.empty(1 << 28, device='cuda', dtype=torch.uint8)
    # Serving priorities first in the diagnostic output as well as the queue.
    for batch, window in ((16, 0), (32, 0), (16, 2048), (32, 2048), (1, 0), (1, 2048)):
        data = _w8_input(batch, window)
        timing = []
        for variant in variants + (['fa3'] if hopper else []):
            if variant == 'fa3':
                fn = _fa3_verify
            else:
                _w8_set(variant)
                fn = _xqa_verify
            fn(data)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = fn(data)
            times = []
            for _ in range(7):
                flush.sum()
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                times.append(start.elapsed_time(end) * 1000)
            timing.append((statistics.median(times), variant))
            del graph, out
        timing.sort(key=lambda r: r[0])
        selected = timing[0][1]
        if hopper:
            # Validate the best custom variant against FA3 with swapped order,
            # rather than selecting from a noisy sequential sweep alone.
            best = next(v for us, v in timing if v != 'fa3')
            _w8_set(best)
            pair_graphs = []
            for fn in (_xqa_verify, _fa3_verify):
                fn(data)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    out = fn(data)
                pair_graphs.append((graph, out))
            pairs = [[], []]
            for rep in range(10):
                for side in ((0, 1) if rep % 2 == 0 else (1, 0)):
                    flush.sum()
                    start.record()
                    pair_graphs[side][0].replay()
                    end.record()
                    end.synchronize()
                    pairs[side].append(start.elapsed_time(end) * 1000)
            ratios = [(pairs[1][r] + pairs[1][r+1]) / (pairs[0][r] + pairs[0][r+1])
                      for r in range(0, 10, 2)]
            speedup = statistics.median(ratios)
            selected = best if speedup > 1.03 else 'fa3'
            print('XQA_W8_PAIR', batch, window, best, 'custom_us', round(statistics.median(pairs[0]), 2),
                  'fa3_us', round(statistics.median(pairs[1]), 2), 'speedup', round(speedup, 4), flush=True)
            del pair_graphs, graph, out
        _W8_PLAN[(batch, bool(window))] = selected
        print('XQA_W8_DIAG', batch, window, 'us', [(round(us, 2), v) for us, v in timing], flush=True)
        print('XQA_W8_PLAN', batch, window, selected, flush=True)
        del data
    del flush
    torch.cuda.synchronize()

def kernel_fn(data: Inputs) -> torch.Tensor:
    cfg = data['config']
    if cfg['q_len'] == 1:
        return _DECODE_FN(data)
    variant = _W8_PLAN.get((cfg['batch'], bool(cfg['window'])))
    if variant == 'fa3':
        return _fa3_verify(data)
    if variant is not None:
        _w8_set(variant)
    else:
        _w8_set((False, 3, 1 if cfg['window'] else 4))
    return _xqa_verify(data)

def prepare():
    global _W8_READY
    if not _W8_READY:
        # The local screen exercises every shape with the real custom kernel;
        # dev/verify_xqa_w8_glue.py forces the Hopper-only selection branches.
        if torch.cuda.get_device_capability()[0] == 9:
            _w8_diagnose(True)
        _W8_READY = True

if torch.cuda.is_available():
    prepare()
'''
import sys
(root / 'dev/xqa_w8_diag_candidate.py').write_text(base)
if '--publish' in sys.argv:
    (root / 'cand_xqa_k128.py').write_text(base)
print('Built', root / 'dev/xqa_w8_diag_candidate.py')
