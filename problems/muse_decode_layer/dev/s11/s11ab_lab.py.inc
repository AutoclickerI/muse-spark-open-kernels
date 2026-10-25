"""S11 vs S10, whole layer, paired, leave-one-out on S11's prepared state (appended to dev/s10/ablate_lab.py.inc, whose
helpers it uses; this lab() replaces that one). Each arm is its own CUDA graph of kernel_fn on the same static inputs,
the host's functions rebound into a copy of its globals (the host never changes):
  A   S11 as prepared;  A2 a second capture (A:A)
  V   balanced split-K GEMV off: every balanced row back to S10's V9 tile / cuBLAS choice      (t <= 32)
  Q   vq8 off: the routed verify rows (1 sliding, 16 full, 32 full) back to S10's attention     (routed rows)
  K   lean K loop off: S10's N_full cell (NF_CUDA kinds 0/1, nf_plan, nf_call; embedded verbatim, compiled in a
      thread at lab start) in place of S11's                                                    (M = 128 / 256)
  S10 all of V, Q, K at once (S10's kernels on S11's pinned plan)
Seeds AB_SEEDS x AB_ROUNDS rounds, rotating order, lab_cold_l2 before each replay. Lines: L, LOAD (46 sliding + 16
full weighted), END; r = mean per-seed time ratio vs A (> 1: the reverted arm is slower, i.e. S11's change wins).
"""
import threading

S10_NF_BLOCK = __S10_NF_BLOCK__
_S11AB_NF = {}


def s11ab_k_namespace():
    v = ab_namespace({})
    exec(compile(S10_NF_BLOCK, 's10_nf_cell', 'exec'), v)
    return v


def s11ab_compile(v):
    try:v['nf_module']();_S11AB_NF['ok'] = True
    except Exception as exc:_S11AB_NF['err'] = f'{type(exc).__name__}:{str(exc)[:80]}'


def lab(deadline, weights, state):
    started = time.monotonic();device = weights['qkv_gate'].device
    lines = [f'META host S11 state; seeds {len(AB_SEEDS)} from {AB_SEEDS[0]} (+7919); {AB_ROUNDS} rounds rotating order, '
             'lab_cold_l2 before each replay; r = mean per-seed time ratio vs A (>1 = the S10 piece is slower)',
             'META V=no balanced GEMV, Q=no vq8, K=S10 N_full cell, S10=V+Q+K, A2=A:A']
    vk = s11ab_k_namespace()
    th = threading.Thread(target=s11ab_compile, args=(vk,), daemon=True);th.start()
    host, s10 = bal_choices(state)
    vchoices = dict(host)
    for (t, role) in kernel_fn.__globals__['GB_ROWS']:
        n, k = state[role].shape;vchoices[(t, n, k)] = s10[(t, n, k)]
    ns = {'A': ab_namespace({}), 'V': ab_namespace({'_GEMV_CHOICES': vchoices}), 'K': vk}
    vall = s11ab_k_namespace();vall['_GEMV_CHOICES'] = vchoices;ns['S10'] = vall
    noq = dict(state, vq8_rows={})
    lines.append(f'SETUP vq8_rows={state.get("vq8_rows")} pins={kernel_fn.__globals__.get("PROJ_PINS")} {time.monotonic() - started:.1f}s')
    results = {};incomplete = []
    order = [k for k in AB_CASES if k[0] * k[1] <= 32] + [k for k in AB_CASES if k[0] * k[1] > 32]
    for key in order:
        if time.monotonic() > deadline - 6:incomplete.append(ab_label(key));continue
        b, q, w = key;t = b * q
        routed = (b, q, bool(w)) in state.get('vq8_rows', {})
        heavy = t in (128, 256)
        if heavy and th.is_alive():th.join(max(0., deadline - 40 - time.monotonic()))
        kok = heavy and _S11AB_NF.get('ok', False)
        lines.append(ab_plan_line(key, state) + f' vq8={routed}')
        mk = lambda seed: _ap_generate_input(batch=b, ctx=80000, layer='sliding' if w else 'full', q_len=q,
                                             spread='agentic' if b > 1 else 'fixed', dist='soft', page=1, seed=seed, device=device)
        data = mk(AB_SEEDS[0])
        variants = {'A': (ns['A'], state), 'A2': (ns['A'], state)}
        if t <= 32:variants['V'] = (ns['V'], state)
        if routed:variants['Q'] = (ns['A'], noq)
        if kok:variants['K'] = (ns['K'], state)
        if t <= 32 or routed or kok:variants['S10'] = (ns['S10'] if kok else ns['V'], noq)
        if heavy and not kok:lines.append(f'K {ab_label(key)} unavailable {_S11AB_NF.get("err", "compile not done")}')
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
    for (b, q), load in AB_LOADS.items():
        ks = [(b, q, 2048), (b, q, 0)]
        if not all(k in results for k in ks):continue
        for n in ('A2', 'V', 'Q', 'K', 'S10'):
            # a load's arm: the shapes where it differs from A, the rest at r = 1 (the arm is A there)
            rs = [results[k][0].get(n, (1., 0., 0., results[k][0]['A2'][3], 0)) for k in ks]
            if all(n not in results[k][0] for k in ks):continue
            num = sum(wt * r[3] * r[0] for r, wt in zip(rs, (46, 16)));den = sum(wt * r[3] for r, wt in zip(rs, (46, 16)))
            lines.append(f'LOAD {load} {n} r={num / den:.4f}')
    if incomplete:lines.append('INCOMPLETE ' + ' '.join(incomplete))
    lines.append(f'END shapes={len(results)}/12 elapsed={time.monotonic() - started:.1f}s left={deadline - time.monotonic():.1f}s')
    return lines
