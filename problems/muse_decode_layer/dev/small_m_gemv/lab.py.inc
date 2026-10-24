def m1_fp32(data, weights):
    ids = data['topk_ids'][0] - data['config']['local_start']
    out = torch.zeros((1, 4096), device=ids.device)
    x = data['hidden_states'].float()
    for slot, e in enumerate(ids.tolist()):
        if not 0 <= e < 32:
            continue
        h = x @ sm_dequant(weights, 'w13', e).T
        activation = torch.nn.functional.silu(h[:, :4096]) * h[:, 4096:]
        y = activation @ sm_dequant(weights, 'w2', e).T
        z = y * (1. + weights['post_norm_w'].float())
        out += z * torch.rsqrt(z.square().mean(-1, keepdim=True) + 1e-8) * data['topk_weights'][0, slot]
    return out


def m1_first_failure(data, weights, bank, splits, pdl, grid=264):
    scratch = m1_stream_kernel(data, bank, splits, pdl, scratch=True, grid=grid)
    torch.cuda.synchronize()
    inter, parts, y, sums, invnorm, counts, out = scratch
    ids = data['topk_ids'][0] - data['config']['local_start']
    for slot, expert in enumerate(ids.tolist()):
        if not 0 <= expert < 32:
            continue
        h = data['hidden_states'].float() @ sm_dequant(weights, 'w13', expert).T
        a = torch.nn.functional.silu(h[:, :4096]) * h[:, 4096:]
        want_y = a @ sm_dequant(weights, 'w2', expert).T
        want_y *= 1. + weights['post_norm_w'].float()
        for stage, got, want in (('swiglu', inter[slot:slot+1], a), ('w2_postnorm', y[slot:slot+1], want_y)):
            msg = sm_first_difference(stage, got, want, expert, torch.zeros(1, device=ids.device, dtype=torch.int32))
            if msg:
                return f'{msg} slot={slot} split={splits} pdl={int(pdl)}'
    return f'FIRST_FAIL output_only split={splits} pdl={int(pdl)} counts=' + str(counts.cpu().tolist()[-3:])


def lab(deadline, weights):
    import statistics
    import concurrent.futures
    device = weights['w13'].device
    lines = [f'META M1 stream GEMV {torch.cuda.get_device_name(device)}; raw E4M3 canonical bank; persistent W13 rows/W2 splitK, fused norm/sum; cold128MiB graphs',
             'META v9m1=previous lab newNT1; independent FP32 <=1pct/row5pct; S/P=normal/PDL; g=W13 grid,W2=2g; g0=flat tasks']
    if lab_time.monotonic() > deadline - 30:
        return lines + ['BUDGET before_compile']
    w = dict(weights, post_norm_w=weights.get('post_norm_w', weights.get('post_expert_norm')))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        stream_future = pool.submit(m1_stream_module)
        small_future = pool.submit(_small_nt4_module)
        bank = v9_prepare(w)
        stream_mod = stream_future.result()
        small_mod = small_future.result()
    ok, text = sm_bank_check(w, bank)
    lines.append(text)
    if not ok:
        return lines
    bank['stream_mod'] = stream_mod
    small = dict(bank, mod=small_mod, cnt=torch.zeros_like(bank['cnt']))
    rw, source = lab_router(weights, device)
    lines[0] += ' router=' + source
    flush = torch.empty(128 << 20, device=device, dtype=torch.uint8)
    choices = [(s, p, 264) for p in (False, True) for s in (1, 2, 4, 8)]
    choices += [(4, p, g) for g in (0, 132, 528) for p in (False, True)]
    labels = ['NT1', 'v9m1'] + [('P' if pdl else 'S') + str(split) + 'g' + str(grid) for split, pdl, grid in choices]
    lines.append('ORDER ' + ','.join(labels))
    rows = {name: [] for name in labels}
    errors = {name: 0. for name in labels}
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    passed = 0
    try:
        for seed_index in range(12):
            if lab_time.monotonic() > deadline - 5:
                lines.append(f'BUDGET before_route_seed={seed_index}')
                break
            seed = 9001 + 7919 * seed_index
            d = fx_make_fixture(rw, 1, seed)
            want = m1_fp32(d, w)
            functions = [lambda d: v9_kernel_fn(d, bank), lambda d: v9_small_nt1_kernel_fn(d, small)]
            functions += [lambda d, split=split, pdl=pdl, grid=grid: m1_stream_kernel(d, bank, split, pdl, grid=grid) for split, pdl, grid in choices]
            graphs = {}
            for name, fn in zip(labels, functions):
                graph = LabGraph(fn, d)
                got = graph.replay()
                torch.cuda.synchronize()
                rel, row = sm_error(got, want)
                errors[name] = max(errors[name], rel)
                if not bool(torch.isfinite(got).all()) or rel > .01 or row > .05:
                    lines.append(f'REJECT {name} seed={seed} fp32={rel:.6f}/{row:.6f}')
                    if name not in ('NT1', 'v9m1'):
                        split, pdl, grid = choices[labels.index(name) - 2]
                        lines.append(m1_first_failure(d, w, bank, split, pdl, grid))
                    return lines
                graphs[name] = graph
            samples = {name: [] for name in labels}
            ratios = {name: [] for name in labels}
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            for repeat in range(8):
                order = labels[repeat % len(labels):] + labels[:repeat % len(labels)]
                if repeat % 2:
                    order = order[::-1]
                timing = {}
                for name in order:
                    flush.zero_(); a.record(); graphs[name].replay(); b.record(); b.synchronize()
                    timing[name] = a.elapsed_time(b) * 1000
                for name in labels:
                    samples[name].append(timing[name]); ratios[name].append(timing[name] / timing['NT1'])
            anchor = statistics.median(samples['NT1'])
            means = [anchor * statistics.median(ratios[name]) for name in labels]
            for name, value in zip(labels, means):
                rows[name].append(value)
            touched = int(((d['topk_ids'] >= 0) & (d['topk_ids'] < 32)).sum())
            floor = touched * 768 * 36864 / 4.8e6
            lines.append(f'R seed={seed} E={touched} floor_us={floor:.2f} us=' + ','.join(f'{v:.2f}' for v in means))
            passed += 1
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
    for name in labels:
        vals = rows[name]
        if vals:
            lines.append(f'AGG {name} seeds={len(vals)} mean_us={statistics.mean(vals):.2f} min={min(vals):.2f} max={max(vals):.2f} fp32={errors[name]:.6f}')
    return lines + [f'SUMMARY complete_route_seeds={passed}/12 left_s={deadline-lab_time.monotonic():.1f}; raw graph times, unique packed-byte floor, no HBM counter']
