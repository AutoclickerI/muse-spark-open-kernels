from pathlib import Path
root=Path(__file__).resolve().parents[1]
s=(root/'cand_moe4_attn.py').read_text()
s=s.replace('    BN, KS, BK = GEMV_CFG[(N, K)]\n', '''    choice = _GEMV_CHOICES.get((T, N, K))
    if choice == "cublas":
        y = torch.mm(x, w.t(), out_dtype=torch.float32) if out_dtype == torch.float32 else F.linear(x, w)
        return y.unsqueeze(0), 1
    BN, KS, BK, pf = choice if choice is not None else (*GEMV_CFG[(N, K)], PREFETCH)
''',1)
s=s.replace('            PREFETCH=PREFETCH, num_warps=4, num_stages=4)','            PREFETCH=pf, num_warps=4, num_stages=4)',1)
s+='''

_GEMV_CHOICES = {}
_prepare_before_tuning = prepare

def _cold_us(fn, flush):
    import statistics
    # These synthetic inputs are used only to select launch parameters. No data
    # supplied to kernel_fn or any derived result is retained by prepare.
    for _ in range(2):
        fn()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    a = torch.cuda.Event(enable_timing=True)
    b = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(7):
        flush.zero_()
        a.record(); graph.replay(); b.record(); b.synchronize()
        times.append(a.elapsed_time(b) * 1000)
    return statistics.median(times)


def prepare(weights):
    state = _prepare_before_tuning(weights)
    flush = torch.empty((128 << 20,), dtype=torch.uint8, device=weights["qkv_gate"].device)
    for t in (1, 8, 16, 32):
        for name in ("qkv_gate", "o", "pool_router", "unpooler_slice", "unpooler"):
            w = state[name]
            n, k = w.shape
            if name == "unpooler_slice" and t > 8 or name == "unpooler" and t <= 8:
                continue
            x = torch.randn((t, k), device=w.device, dtype=torch.bfloat16)
            choices = [(*GEMV_CFG[(n,k)], True), (32, 1024 if k >= 4096 else 512, 128, False),
                       (64, 512, 128, False), "cublas"]
            scores = []
            for choice in choices:
                _GEMV_CHOICES[t,n,k] = choice
                dtype = torch.float32 if name == "pool_router" else torch.bfloat16
                score = _cold_us(lambda: gemv(x, w, dtype), flush)
                scores.append((score, choice))
            score, choice = min(scores, key=lambda z:z[0])
            _GEMV_CHOICES[t,n,k] = choice
            print(f"STAGE_TUNE {name} T={t}: {score:.2f} us {choice}; all={scores}", flush=True)
    del flush
    return state
'''
(root/'cand_tuned.py').write_text(s)
