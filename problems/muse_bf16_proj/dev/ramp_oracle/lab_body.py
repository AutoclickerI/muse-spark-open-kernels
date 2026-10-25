def lab(deadline, weights):
    t0 = time.monotonic()
    dev = weights['o'].device
    gen = torch.Generator(device=dev).manual_seed(11)
    lines = ['META paired AB/BA x24, lab_cold_l2 per replay; oracle arms warm their bytes after the flush, outside the '
             'events. Deltas in us = (paired ratio - base ratio) x control median. OATT: qkv bal64/7 -> sumcast -> attn '
             '(S10) -> O (S10 V9) -> b1 -> sink; vs ctrl. POOL: O -> b1 -> pool_router -> route_cl -> sink; bal vs s10, '
             'wJ/bpJ/s10pf vs bal (s10pf vs s10). UNPOOL: unpooler -> b2 -> sink; wJ vs bal (t=1: warm vs s10).']
    for b, w in ((1, 2048), (1, 0), (16, 2048), (32, 2048)):
        if time.monotonic() > deadline - 25:
            lines.append(f'SKIP OATT b={b} w={w}')
            continue
        try:
            v = _ro_vectors(weights, dev, gen, b)
            attn = RoAttention(b, w, dev)
            x = (torch.randn((b, 8192), device=dev, generator=gen) * 0.5).to(torch.bfloat16)
            ow = lambda: ro_warm(v['o'], 64, 1, 1 << 20)  # noqa: E731
            arms = [('ctrl', RoOatt(v, attn, x, False), None), ('AA', RoOatt(v, attn, x, False), None),
                    ('o_pf', RoOatt(v, attn, x, True), None), ('o_warm', RoOatt(v, attn, x, False), ow),
                    ('kv_warm', RoOatt(v, attn, x, False), attn.warm_kv),
                    ('both', RoOatt(v, attn, x, False), lambda: (ow(), attn.warm_kv()))]
            got, err = _ro_case(arms)
            lines.append(f'OATT b={b} w={w} {attn.kind} ns/cs={attn.plan(False)[1]}/{attn.plan(False)[2]} kv={attn.kv_idx.numel() * 256 / 1e6:.1f}MB '
                         f'ctrl={got["ctrl"][0]:.2f}us AA={got["AA"][1]:.4f} | '
                         + _ro_fmt(got, ('o_pf', 'o_warm', 'kv_warm', 'both'), 'ctrl') + f' | e={err:.0e}')
        except Exception as exc:
            lines.append(f'ERROR OATT b={b} w={w} {type(exc).__name__}: {str(exc).splitlines()[0][:150]}')
        torch.cuda.empty_cache()
    for t in (1, 16, 32):
        if time.monotonic() > deadline - 20:
            lines.append(f'SKIP POOL t={t}')
            continue
        try:
            v = _ro_vectors(weights, dev, gen, t)
            wp = v['pool_router']
            a = (torch.randn((t, 1024), device=dev, generator=gen) * 0.5).to(torch.bfloat16)
            s10 = (32, 1024, 128) if t <= 16 else (64, 512, 128)
            arms = [('s10', RoPool(v, a, lambda h: ro_s10(h, wp, s10)), None),
                    ('AA', RoPool(v, a, lambda h: ro_s10(h, wp, s10)), None),
                    ('s10pf', RoPool(v, a, lambda h: ro_s10(h, wp, s10, pf=True)), None),
                    ('bal', RoPool(v, a, lambda h: ro_bal(h, wp, 64, 6)), None)]
            for J in (2, 4, 6):
                arms.append((f'w{J}', RoPool(v, a, lambda h: ro_bal(h, wp, 64, 6)), lambda J=J: ro_warm(wp, 64, 6, J)))
            for J, tag in ((2, 'bp2'), (4, 'bp4'), (64, 'bpall')):
                arms.append((tag, RoPool(v, a, lambda h, J=J: ro_bal(h, wp, 64, 6, BP=J)), None))
            got, err = _ro_case(arms)
            lines.append(f'POOL t={t} s10={got["s10"][0]:.2f}us AA={got["AA"][1]:.4f} | ' + _ro_fmt(got, ('bal',), 's10')
                         + ' s10pf ' + f'{(got["s10pf"][1] - 1) * got["s10"][0]:+.2f} | '
                         + _ro_fmt(got, ('w2', 'w4', 'w6', 'bp2', 'bp4', 'bpall'), 'bal') + f' | e={err:.0e}')
        except Exception as exc:
            lines.append(f'ERROR POOL t={t} {type(exc).__name__}: {str(exc).splitlines()[0][:150]}')
        torch.cuda.empty_cache()
    for t in (1, 16, 32):
        if time.monotonic() > deadline - 15:
            lines.append(f'SKIP UNPOOL t={t}')
            continue
        try:
            v = _ro_vectors(weights, dev, gen, t)
            x = (torch.randn((t, 4096), device=dev, generator=gen) * 0.5).to(torch.bfloat16)
            if t == 1:
                ws = v['unpool_slice']
                arms = [('s10', RoUnpool(v, x, lambda z: ro_s10(z, ws, (64, 512, 128))), None),
                        ('AA', RoUnpool(v, x, lambda z: ro_s10(z, ws, (64, 512, 128))), None),
                        ('warm', RoUnpool(v, x, lambda z: ro_s10(z, ws, (64, 512, 128))),
                         lambda: ro_warm(ws, 64, 1, 1 << 20))]
                names, base = ('warm',), 's10'
            else:
                wu = v['unpooler']
                arms = [('s10', RoUnpool(v, x, lambda z: ro_s10(z, wu, 'cublas')), None),
                        ('AA', RoUnpool(v, x, lambda z: ro_s10(z, wu, 'cublas')), None),
                        ('bal', RoUnpool(v, x, lambda z: ro_bal(z, wu, 64, 2)), None)]
                for J in (3, 6, 10):
                    arms.append((f'w{J}', RoUnpool(v, x, lambda z: ro_bal(z, wu, 64, 2)),
                                 lambda J=J: ro_warm(wu, 64, 2, J)))
                names, base = ('w3', 'w6', 'w10'), 'bal'
            got, err = _ro_case(arms)
            lines.append(f'UNPOOL t={t} s10={got["s10"][0]:.2f}us AA={got["AA"][1]:.4f} | '
                         + (_ro_fmt(got, ('bal',), 's10') + ' | ' if t > 1 else '') + _ro_fmt(got, names, base)
                         + f' | e={err:.0e}')
        except Exception as exc:
            lines.append(f'ERROR UNPOOL t={t} {type(exc).__name__}: {str(exc).splitlines()[0][:150]}')
        torch.cuda.empty_cache()
    lines.append(f'END t={time.monotonic() - t0:.1f}s')
    return lines
