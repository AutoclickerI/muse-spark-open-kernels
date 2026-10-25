"""Whole-layer paired A/B of the balanced split-K GEMVs on S11's prepared state (appended to dev/s10/ablate_lab.py.inc,
whose helpers it uses; this lab() replaces that one).

Variants at t <= 32 (each its own CUDA graph of kernel_fn on the same static inputs; functions rebound into a copy of
the host globals with its own _GEMV_CHOICES, so the host is never changed):
  A   S11 as prepared (qkv 64/7, pool_router 64/6 at t = 1/8/16/32, unpooler 64/2 at t = 16/32);  A2 a second capture
  V   every balanced row back to S10's V9 tile / cuBLAS choice (= S10's projections)
  Vq / Vp / Vu  only qkv / pool_router / unpooler back to S10's choice
  Qc / Qb / Ql  verify 1 only: the qkv projection forced to custom (now balanced) / F.linear / cuBLASLt algo 2
Context PRE (qkv -> attention -> O -> b1 -> pool_router, static inputs) for A vs V.
K <shape> <variant> bal=<balanced GEMV launches in one eager kernel_fn>: proves each arm runs what it names.
Lines as in ablate_lab: L, LOAD (46 sliding + 16 full weighted), C, P, END.
"""
BAL_CASES = tuple(k for k in AB_CASES if k[0] * k[1] <= 32)
BAL_ROLES = {'Vq': ('qkv_gate',), 'Vp': ('pool_router',), 'Vu': ('unpooler',), 'V': ('qkv_gate', 'pool_router', 'unpooler')}


def bal_choices(state):
    """(host choices, S10 choices): S10's = the V9 seed without the balanced rows."""
    g = kernel_fn.__globals__
    v = ab_namespace({'GB_ROWS': {}, '_GEMV_CHOICES': {}})
    v['_seed_v9_gemv_choices'](state)
    return dict(g['_GEMV_CHOICES']), dict(v['_GEMV_CHOICES'])


def bal_namespaces(state):
    host, s10 = bal_choices(state)
    ns = {'A': ab_namespace({'_GEMV_CHOICES': dict(host)})}
    for arm, roles in BAL_ROLES.items():
        ch = dict(host)
        for (t, role) in kernel_fn.__globals__['GB_ROWS']:
            if role in roles:
                n, k = state[role].shape
                ch[(t, n, k)] = s10[(t, n, k)]
        ns[arm] = ab_namespace({'_GEMV_CHOICES': ch})
    return ns


def bal_count(v, data, st):
    """Balanced GEMV launches in one eager kernel_fn of namespace v."""
    n = [0];f = v['_gb_gemv']
    def counted(*a, **k):
        n[0] += 1;return f(*a, **k)
    v['_gb_gemv'] = counted
    try:v['kernel_fn'](data, st);torch.cuda.synchronize()
    finally:v['_gb_gemv'] = f
    return n[0]


def lab(deadline, weights, state):
    started = time.monotonic();device = weights['qkv_gate'].device
    lines = [f'META host S11 state; seeds {len(AB_SEEDS)} from {AB_SEEDS[0]} (+7919); {AB_ROUNDS} rounds rotating order, '
             'lab_cold_l2 before each replay; r = mean per-seed time ratio vs A (<1 = faster than the host)',
             'META V=S10 projections (no balanced GEMV), Vq/Vp/Vu=one role back to S10, A2=A:A, Q*=verify-1 qkv choice; '
             'C ctx PRE=qkv..pool_router']
    ns = bal_namespaces(state)
    results = {};chains = [];incomplete = []
    for key in BAL_CASES:
        if time.monotonic() > deadline - 6:incomplete.append(ab_label(key));continue
        b, q, w = key;t = b * q;plan = state['best7_plans'][key]
        lines.append(ab_plan_line(key, state))
        mk = lambda seed: _ap_generate_input(batch=b, ctx=80000, layer='sliding' if w else 'full', q_len=q,
                                             spread='agentic' if b > 1 else 'fixed', dist='soft', page=1, seed=seed, device=device)
        data = mk(AB_SEEDS[0])
        variants = {'A': (ns['A'], state), 'A2': (ns['A'], state)}
        for arm in BAL_ROLES:
            if arm != 'Vu' or t > 8:variants[arm] = (ns[arm], state)
        if t == 8:
            cur = state['proj_plan'].get((8, 'qkv_gate'), 'custom')
            alts = {'Qc': 'custom', 'Qb': 'cublas'}
            if state.get('lt_mod') is not None and len(state.get('lt_algos', {}).get(('qkv_gate', 8), ())) > 2:alts['Ql'] = ('lt', 2)
            for n, c in alts.items():
                if c != cur:variants[n] = (ns['A'], dict(state, proj_plan={**state['proj_plan'], (8, 'qkv_gate'): c}))
        for n, (v, st) in variants.items():
            if n != 'A2':
                try:lines.append(f'K {ab_label(key)} {n} bal={bal_count(v, data, st)}')
                except Exception as exc:lines.append(f'K {ab_label(key)} {n} ERROR {type(exc).__name__}:{str(exc)[:60]}')
        graphs = {};gates = {}
        for n, (v, st) in variants.items():
            try:graphs[n] = ab_graph(lambda v=v, st=st: v['kernel_fn'](data, st))
            except Exception as exc:gates[n] = 'capture:' + type(exc).__name__ + ':' + str(exc)[:60]
        per_seed = []
        for seed in AB_SEEDS:
            if time.monotonic() > deadline - 4:incomplete.append(f'{ab_label(key)}@{seed}');break
            if seed != AB_SEEDS[0]:ab_copy(data, mk(seed))
            for g, _ in graphs.values():g.replay()
            torch.cuda.synchronize()
            for n, (g, out) in graphs.items():
                if n != 'A' and (r := ab_gate(out, graphs['A'][1])):gates.setdefault(n, r)
            times = ab_time(graphs, AB_ROUNDS)
            per_seed.append({n: statistics.mean(x) for n, x in times.items()})
        if per_seed:results[key] = (ab_summary(per_seed), gates)
        for n, (r, se, d, a_us, k) in sorted(results.get(key, ({}, {}))[0].items()):
            lines.append(f'L {ab_label(key)} {n} r={r:.4f} se={se:.4f} d={d:+.2f} A={a_us:.1f} n={k}' + (f' {gates[n]}' if n in gates else ''))
        for n in gates:
            if n not in results.get(key, ({}, {}))[0]:lines.append(f'L {ab_label(key)} {n} FAIL {gates[n]}')
        graphs = variants = None;gc.collect();torch.cuda.empty_cache()
        if time.monotonic() < deadline - 10:
            try:chains += bal_context(key, plan, ns, state, mk, deadline)
            except Exception as exc:chains.append(f'C {ab_label(key)} ERROR {type(exc).__name__}:{str(exc)[:80]}')
            gc.collect();torch.cuda.empty_cache()
    for (b, q), load in AB_LOADS.items():
        ks = [(b, q, 2048), (b, q, 0)]
        if not all(k in results for k in ks):continue
        names = set(results[ks[0]][0]) & set(results[ks[1]][0])
        for n in sorted(names):
            num = sum(wt * results[k][0][n][3] * results[k][0][n][0] for k, wt in zip(ks, (46, 16)))
            den = sum(wt * results[k][0][n][3] for k, wt in zip(ks, (46, 16)))
            lines.append(f'LOAD {load} {n} r={num / den:.4f}')
    lines += chains
    if incomplete:lines.append('INCOMPLETE ' + ' '.join(incomplete))
    lines.append(f'END shapes={len(results)}/{len(BAL_CASES)} elapsed={time.monotonic() - started:.1f}s left={deadline - time.monotonic():.1f}s')
    return lines


def bal_context(key, plan, ns, state, mk, deadline):
    """PRE (qkv .. pool_router) for A vs V on static inputs per seed."""
    data = mk(AB_SEEDS[0])
    graphs = {n: ab_graph(lambda v=ns[n]: ab_prefix(v, data, state, plan)) for n in ('A', 'V')}
    rows = []
    for seed in AB_SEEDS[:8]:
        if time.monotonic() > deadline - 4:break
        if seed != AB_SEEDS[0]:ab_copy(data, mk(seed))
        torch.cuda.synchronize()
        times = ab_time(graphs, 6)
        rows.append({n: statistics.mean(v) for n, v in times.items()})
    return [f'C {ab_label(key)} PRE {n} r={r:.4f} se={se:.4f} d={d:+.2f} A={a_us:.1f} n={k}'
            for n, (r, se, d, a_us, k) in ab_summary(rows).items()] if rows else []
