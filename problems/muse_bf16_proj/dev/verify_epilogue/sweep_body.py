import statistics
import time

VE_CONFIGS = __CONFIGS__


def ve_capture(fn):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr, stream=stream):
        out = fn()
    torch.cuda.current_stream().wait_stream(stream)
    gr.replay()
    torch.cuda.synchronize()
    return gr, out, stream


def ve_timing(graphs, rounds):
    vals = {n: [] for n in graphs}
    names = list(graphs)
    for r in range(rounds):
        order = names if r % 2 == 0 else names[::-1]
        for n in order:
            lab_cold_l2()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            graphs[n][0].replay()
            end.record()
            end.synchronize()
            vals[n].append(start.elapsed_time(end) * 1000.)
    return {n: statistics.median(v) for n, v in vals.items()}


def lab(deadline, weights, state=None):
    started = time.monotonic()
    dev = weights['o'].device
    lines = ['META verify epilogue screen; LIB=F.linear+S11 boundary (not the final S11 Lt comparator); '
             'r=variant/LIB; cold READ; BF16 GEMM output then FP32 norm stats; use whole-layer lab for promotion']
    for m in (128, 256):
        for name in ('o', 'unpooler'):
            post = name == 'unpooler'
            w = weights[name]
            gen = torch.Generator(device=dev).manual_seed(18013 + m + int(post))
            x = torch.randn((m, w.shape[1]), device=dev, generator=gen).bfloat16()
            st = torch.randn((m, 8192), device=dev, generator=gen)
            al = torch.rand(8192, device=dev, generator=gen) * .3 + .7
            be = torch.rand(8192, device=dev, generator=gen) * .3 + .3
            pw = torch.randn(8192, device=dev, generator=gen) * .1 + 1 if post else None
            pre = torch.randn(8192, device=dev, generator=gen) * .1 + 1
            def boundary(y):
                out, sout = torch.empty_like(y), torch.empty_like(st)
                fn = _boundary_late if m > 128 else _boundary_fast
                kw = {} if m > 128 else dict(REPLICAS=1)
                fn[(m,) if m > 128 else (m, 1)](
                    y, 1, y.numel(), y.stride(0), st, sout, out, pw if post else al, al, be, pre,
                    NB=8192, HAS_POSTW=post, USE_PDL=True, num_warps=16 if not post or m == 256 else 8,
                    launch_pdl=True, **kw)
                return out, sout
            lib = ve_capture(lambda: boundary(torch.nn.functional.linear(x, w)))
            graphs = {'LIB': lib}
            rejected = []
            for i, cfg in enumerate(VE_CONFIGS):
                if time.monotonic() > deadline - 20:
                    rejected.append('deadline');break
                for stats in (False, True):
                    tag = ('F' if stats else 'G') + str(i)
                    def fn(cfg=cfg, stats=stats):
                        y, p = ve_gemm(x, w, st, al, be, pw, cfg, stats)
                        return ve_apply(y, p, st, al, be, pre, pw, cfg[1]) if stats else boundary(y)
                    try:
                        g = ve_capture(fn)
                        err = max(float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12))
                                  for a, b in zip(g[1], lib[1]))
                        if not all(bool(torch.isfinite(a).all()) for a in g[1]) or err > .005:
                            rejected.append(f'{tag}:L2={err:.4f}');continue
                        graphs[tag] = g
                    except Exception as exc:
                        rejected.append(f'{tag}:{type(exc).__name__}:{str(exc)[:75]}')
            # Short screen, then rotate three fresh inputs on the top four arms.
            screen = ve_timing(graphs, 4)
            keep = ['LIB'] + sorted((n for n in graphs if n != 'LIB'), key=lambda n: screen[n])[:4]
            finalists = {n: graphs[n] for n in keep}
            graphs = None
            rows = {n: [] for n in keep}
            for seed in range(3):
                x.normal_(generator=gen);st.normal_(generator=gen)
                vals = ve_timing(finalists, 12)
                for n in keep:
                    rows[n].append(vals[n] / vals['LIB'])
            lines.append(f'KEY M{m} {name} LIB={screen["LIB"]:.3f}us ' +
                         ' '.join(f'{n}:{screen[n]:.3f}' for n in sorted(screen) if n != 'LIB'))
            for n in keep[1:]:
                lines.append(f'FINAL M{m} {name} {n} cfg={VE_CONFIGS[int(n[1:])]} '
                             f'r={statistics.mean(rows[n]):.5f} seeds={tuple(round(r,5) for r in rows[n])}')
            if rejected:
                lines.append(f'REJECT M{m} {name} ' + ' | '.join(rejected))
            del lib, finalists
    lines.append(f'END elapsed={time.monotonic()-started:.1f}s left={deadline-time.monotonic():.1f}s')
    return lines
