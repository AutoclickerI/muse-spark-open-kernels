import statistics
import time


@functools.lru_cache(None)
def scale_module():
    arch = '90a' if torch.cuda.get_device_capability() == (9, 0) else '120'
    mod = load_inline(name='m128_scales_' + hashlib.sha256(SCALE_CUDA.encode()).hexdigest()[:12],
        cpp_sources=SCALE_CPP, cuda_sources=SCALE_CUDA, functions=SCALE_FUNCTIONS,
        no_implicit_headers=True, with_pytorch_error_handling=False,
        extra_cuda_cflags=['-O3', '-std=c++20', '-lineinfo', '-gencode=arch=compute_' + arch + ',code=sm_' + arch],
        verbose=False)
    mod.msraw_init(); mod.msrec_init()
    return mod


class ScaleABI:
    def __init__(self, name):
        self.mod = scale_module(); self.name = name

    def ft_route(self, ids, c, off, order, task, slots, pos, state, out, nt, start, clear):
        return getattr(self.mod, self.name + '_route')(
            [v.data_ptr() for v in (ids, c, off, order, task, slots, pos, state, out)],
            ids.shape[0], nt, start, clear, torch.cuda.current_stream().cuda_stream)

    def ft_run(self, v, first, which, grid, start):
        return getattr(self.mod, self.name + '_run')([t.data_ptr() for t in v], first, which, grid, start,
            v[11].shape[0], v[9].shape[0], torch.cuda.current_stream().cuda_stream)


def scale_prepare(w, recoded):
    bank = {}
    for key in ('w13', 'w2'):
        q, sf = w[key], w[key + '_sf']
        e, n, k16 = sf.shape
        if recoded:
            # Recode one expert at a time; only packed E2M1 codes and one byte
            # per block survive prepare. Never retain a dequantized bank.
            planes = []; scales = []
            for expert in range(e):
                code, neg, zero = _encode_scales(sf[expert])
                folded = q[expert].reshape(n, k16, 8) ^ (neg.to(torch.uint8)[..., None] * 0x88)
                folded = torch.where(zero[..., None], torch.zeros_like(folded), folded)
                planes.append(mw_repack_planes(folded.reshape(1, n, -1)))
                scales.append(code)
            bank[key] = torch.cat(planes)
            sfbytes = torch.stack(scales)
        else:
            bank[key] = mw_repack_planes(q)
            sfbytes = sf.view(torch.uint8)
        bank[key + '_sf'] = sfbytes.reshape(e, n // 16, 2, 8, k16 // 2, 2).permute(0, 1, 4, 3, 5, 2).contiguous()
    bank['w13_gs'] = torch.cat((w.get('w13_gs_gate', w['w13_gs']), w.get('w13_gs_up', w['w13_gs']))).float().contiguous()
    bank['w2_gs'] = w['w2_gs'].float().contiguous(); bank['pnw'] = w['post_norm_w'].contiguous()
    ng = bank['w13'].shape[1]
    rr = torch.arange(ng // 2, device=w['w13'].device).reshape(-1, 2)
    order = torch.cat((rr, rr + ng // 2), 1).flatten()
    bank['w13'] = bank['w13'].index_select(1, order).contiguous()
    bank['w13_sf'] = bank['w13_sf'].index_select(1, order).contiguous()
    bank['mod'] = ScaleABI('msrec' if recoded else 'msraw')
    return bank


def scale_call(data, bank, which=3, grid=528):
    for f in repair_plan(data, bank, which, grid)['steps']:
        out = f()
    return out


def scale_error(got, want):
    if not bool(torch.isfinite(got).all()):
        return float('inf'), float('inf')
    delta = got.float() - want.float(); den = want.float().norm(dim=-1)
    return float(delta.norm() / want.float().norm().clamp_min(1e-12)), float(
        (delta.norm(dim=-1) / den.clamp_min(den.mean() * .1 + 1e-12)).max())


SCALE_CHOICES = tuple((f'{fmt}{cfg}', fmt, cfg, 528) for cfg in (3, 4, 5) for fmt in ('raw', 'rec'))
SCALE_NAMES = ('NT1rec',) + tuple(c[0] for c in SCALE_CHOICES)
SCALE_RECORDS = []


def scale_measure(w, banks, native, ds, flush):
    wants = dc_oracles(ds, w)
    graphs = [MWGraph(lambda d: v9_kernel_fn(d, native), ds[0])]
    graphs += [MWGraph(lambda d, b=banks[fmt], c=cfg, z=grid: scale_call(d, b, c, z), ds[0])
               for name, fmt, cfg, grid in SCALE_CHOICES]
    errors = [[0., 0.] for g in graphs]; rows = []
    start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
    for fi, (data, want) in enumerate(zip(ds, wants)):
        ok = []
        for j, graph in enumerate(graphs):
            graph.refresh(data); got = graph.replay(); torch.cuda.synchronize()
            e, r = scale_error(got, want)
            errors[j][0] = max(errors[j][0], e); errors[j][1] = max(errors[j][1], r)
            ok.append(e <= .01 and r <= .05)
        if not ok[0]:
            raise RuntimeError('NT1rec independent FP32 gate failed: ' + str(errors[0]))
        samples = [[] for g in graphs]; noise = []
        # Same graph appears twice to measure A:A noise. Rotate and reverse
        # every order; every implementation sees the same refreshed tensors.
        for repeat in range(6):
            order = list(range(len(graphs))) + [0]
            offset = repeat % len(order); order = order[offset:] + order[:offset]
            if repeat % 2: order.reverse()
            aa = []
            for j in order:
                if not ok[j]: continue
                flush.zero_(); start.record(); graphs[j].replay(); end.record(); end.synchronize()
                us = start.elapsed_time(end) * 1000.
                if j == 0: aa.append(us)
                samples[j].append(us)
            noise.append(aa[1] - aa[0])
        for j, graph in enumerate(graphs):
            e, r = scale_error(graph.out, want)
            errors[j][0] = max(errors[j][0], e); errors[j][1] = max(errors[j][1], r)
        counts = torch.bincount(data['topk_ids'][data['topk_ids'] >= 0], minlength=32)
        rows.append(dict(fixture=fi, route=[int(counts.sum()), int((counts > 0).sum()), int(counts.max())],
            us=[statistics.median(s) if s else float('inf') for s in samples],
            aa_us=statistics.mean(noise), aa_samples=noise, samples=samples,
            errors=[list(e) for e in errors]))
    return rows


def lab(deadline, weights, state=None):
    global SCALE_RECORDS
    SCALE_RECORDS = []; started = time.monotonic()
    dev = next(v.device for v in weights.values() if torch.is_tensor(v))
    lines = ['META M128 raw/recoded fused WGMMA versus actual v9 NT1-recoded; same real routes and inputs. '
             'Exact E4M3 exponent/mantissa recode, sign and zero folded into E2M1; no expanded bank. K64 repair retained.',
             'META 12 public-router weight seeds x3 fresh fixtures, six rotated/reversed cold128MiB replays; '
             'A:A=NT1rec duplicate in each order. FP32 finite,L2<=.01,row<=.05 before+after. ORDER ' + ','.join(SCALE_NAMES)]
    if deadline - time.monotonic() < 45:
        return lines + ['INCOMPLETE before_compile need45s']
    mod = scale_module(); _v9_module()
    lines.append(f'BUILD elapsed={time.monotonic()-started:.1f} left={deadline-time.monotonic():.1f}')
    for cfg in (3, 4, 5):
        lines.append(f'CFG {cfg} raw13/2={mod.msraw_info(cfg,1)}/{mod.msraw_info(cfg,0)} '
                     f'rec13/2={mod.msrec_info(cfg,1)}/{mod.msrec_info(cfg,0)}')
    flush = torch.empty(128 << 20, device=dev, dtype=torch.uint8)
    for i in range(12):
        if deadline - time.monotonic() < 8:
            lines.append(f'INCOMPLETE before_seed={i}'); break
        w = dc_weights(5353 + 7919 * i, dev)
        # Board NT1 has one W13 global scale; use the actual common-scale
        # workload. Separate gate/up scales remain covered by local producer gates.
        w['w13_gs_up'] = w['w13_gs']; w['w13_gs_gate'] = w['w13_gs']
        banks = {'raw': scale_prepare(w, False), 'rec': scale_prepare(w, True)}
        native = v9_prepare(w); ds = dc_fixtures(w, 128)
        rows = scale_measure(w, banks, native, ds, flush)
        SCALE_RECORDS.extend(dict(seed=i, **r) for r in rows)
        means = [statistics.mean(r['us'][j] for r in rows) for j in range(len(SCALE_NAMES))]
        lines.append(f'S{i} routes=' + ','.join('/'.join(map(str,r['route'])) for r in rows)
                     + ' us=' + ','.join(f'{v:.1f}' for v in means)
                     + f' AA={statistics.mean(r["aa_us"] for r in rows):.2f}')
        del banks, native, ds, w; gc.collect()
    for j, name in enumerate(SCALE_NAMES):
        if SCALE_RECORDS:
            vals = [r['us'][j] for r in SCALE_RECORDS]
            error = max(r['errors'][j][0] for r in SCALE_RECORDS)
            row = max(r['errors'][j][1] for r in SCALE_RECORDS)
            lines.append(f'MEAN {name} fixtures={len(vals)}/36 us={statistics.mean(vals):.2f} '
                f'vsNT1={statistics.mean(r["us"][j]/r["us"][0] for r in SCALE_RECORDS):.4f} '
                f'L2={error:.5f} row={row:.5f} ' + ('PASS' if error <= .01 and row <= .05 else 'REJECT'))
    for cfg in (3, 4, 5):
        a = SCALE_NAMES.index('raw' + str(cfg)); b = a + 1
        pairs = [r for r in SCALE_RECORDS if all(r['errors'][j][0] <= .01 and r['errors'][j][1] <= .05 for j in (a,b))]
        if pairs:
            lines.append(f'PAIR cfg{cfg} rec:raw={statistics.mean(r["us"][b]/r["us"][a] for r in pairs):.4f}')
    lines.append(f'END fixtures={len(SCALE_RECORDS)}/36 elapsed={time.monotonic()-started:.1f} left={deadline-time.monotonic():.1f}')
    return lines
