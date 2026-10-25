VE_SEEDS = tuple(172903 + 7919 * i for i in range(12))
VE_CASES = ((16, 8, 2048), (16, 8, 0), (32, 8, 2048), (32, 8, 0))
VE_ROUNDS = 6
VE_CFG = (64, 128, 64, 4, 4, 1)
VE_CFG6 = (64, 128, 64, 6, 4, 1)


def ve_host(weights):
    fname = '<p7_s11_final_verify_epilogue>'
    linecache.cache[fname] = (len(VE_HOST_SRC), None, VE_HOST_SRC.splitlines(True), fname)
    ns = dict(__name__='p7_verify_s11')
    exec(compile(VE_HOST_SRC, fname, 'exec'), ns)
    if torch.cuda.get_device_capability() != (9, 0):
        ns['_ap_fa3_scheduler_metadata'] = lambda *a, **kw: None
    st = ns['prepare'](weights)
    return ns, st


def lab(deadline, weights, state=None):
    global kernel_fn
    started = time.monotonic()
    lines = ['META paired whole S11 final acda6c62; A2=A:A G=own GEMM+old boundary F=own GEMM+norm stats '
             'Fo/Fu=one boundary only F6=6 stages; r=variant/A (<1 wins); 12 seeds x 6 rotating rounds; READ flush']
    host, st = ve_host(weights)
    kernel_fn = host['kernel_fn']
    device = weights['o'].device
    lines.append(f'SETUP {time.monotonic()-started:.1f}s pins=' + repr({k: v for k, v in st['proj_plan'].items() if k[0] >= 128}))
    results = {}
    for key in VE_CASES:
        if time.monotonic() > deadline - 10:
            lines.append('INCOMPLETE ' + ab_label(key));continue
        b, q, window = key
        m = b * q
        mk = lambda seed: host['_ap_generate_input'](
            batch=b, ctx=80000, layer='sliding' if window else 'full', q_len=q,
            spread='agentic', dist='soft', page=1, seed=seed, device=device)
        data = mk(VE_SEEDS[0])
        allrows = {(m, name): (VE_CFG, 1024, True) for name in ('o', 'unpooler')}
        rows = {'A': {}, 'A2': {}, 'F': allrows,
                'G': {k: (cfg, chunk, False) for k, (cfg, chunk, _) in allrows.items()},
                'Fo': {k: v for k, v in allrows.items() if k[1] == 'o'},
                'Fu': {k: v for k, v in allrows.items() if k[1] == 'unpooler'},
                'F6': {k: (VE_CFG6, 1024, True) for k in allrows}}
        # Oversized staging cells are admitted by check_kernels' explicitly
        # recorded local stand-in; this screen executes the supported config.
        if torch.cuda.get_device_capability() != (9, 0):
            rows.pop('F6')
        graphs = {}
        gates = {}
        for arm, r in rows.items():
            armstate = dict(st, ve_rows=r)
            try:
                graphs[arm] = ab_graph(lambda armstate=armstate: kernel_fn(data, armstate))
            except Exception as exc:
                gates[arm] = f'capture:{type(exc).__name__}:{str(exc)[:100]}'
        if 'A' not in graphs:
            lines.append(f'FAIL {ab_label(key)} control {gates}');continue
        per_seed = []
        for seed in VE_SEEDS:
            if time.monotonic() > deadline - 5:
                lines.append(f'INCOMPLETE {ab_label(key)}@{seed}');break
            if seed != VE_SEEDS[0]:
                ab_copy(data, mk(seed))
            for g, _ in graphs.values():
                g.replay()
            torch.cuda.synchronize()
            for arm, (_, out) in graphs.items():
                if arm != 'A' and (reason := ab_gate(out, graphs['A'][1])):
                    gates.setdefault(arm, reason)
            times = ab_time(graphs, VE_ROUNDS)
            per_seed.append({n: statistics.mean(t) for n, t in times.items()})
        if per_seed:
            summary = ab_summary(per_seed)
            results[key] = summary
            for n, (r, se, d, a, seeds) in summary.items():
                lines.append(f'L {ab_label(key)} {n} r={r:.5f} se={se:.5f} d={d:+.3f} A={a:.3f} n={seeds}' +
                             (f' FAIL:{gates[n]}' if n in gates else ''))
        for arm in gates:
            if arm not in graphs:
                lines.append(f'L {ab_label(key)} {arm} FAIL:{gates[arm]}')
        del data, graphs
        gc.collect();torch.cuda.empty_cache()
    for b in (16, 32):
        ks = [(b, 8, 2048), (b, 8, 0)]
        if not all(k in results for k in ks):
            continue
        for n in sorted(set(results[ks[0]]) & set(results[ks[1]])):
            den = sum(w * results[k][n][3] for k, w in zip(ks, (46, 16)))
            num = sum(w * results[k][n][3] * results[k][n][0] for k, w in zip(ks, (46, 16)))
            lines.append(f'LOAD verify{b} {n} r={num/den:.5f}')
    lines.append(f'END shapes={len(results)}/4 elapsed={time.monotonic()-started:.1f}s left={deadline-time.monotonic():.1f}s')
    return lines
