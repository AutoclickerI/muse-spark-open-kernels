

# ---- whole-layer paired A/B: the host's prepared state with and without the coded dense GEMVs ----
# Variants (each its own CUDA graph of kernel_fn on the same static inputs; every function of the host module rebound
# into a copy of its globals, so the host itself is never changed):
#   A   the host as prepared;  A2 a second capture of A (A:A noise)
#   C   every (tokens, role) in CL_ROWS on the coded GEMV (a coded copy of state's weights, made here and checked
#       bit for bit); decode-1 'stream' o overrides dropped where CL_ROWS codes o at t = 1
#   Cq / Cp / Co / Cu   only qkv_gate / pool_router / o / unpooler(_slice) coded (attribution)
# Each seed: copy the fixture into the static inputs, replay every graph once and gate it against A (route IDs exact
# up to 2% near-tie flips, relative L2 <= 1%), then LY_ROUNDS rounds, each replaying every graph once in a rotating
# order with lab_cold_l2() before each replay.
import gc
import math
import statistics
import time
import types

LY_SEEDS = tuple(172903 + 7919 * i for i in range(12))
LY_ROUNDS = 6
LY_CASES = ((1, 1, 2048), (1, 1, 0), (1, 8, 2048), (1, 8, 0), (16, 1, 2048), (16, 1, 0), (32, 1, 2048), (32, 1, 0))
LY_LOADS = {(1, 1): 'decode 1', (16, 1): 'decode 16', (32, 1): 'decode 32', (1, 8): 'verify 1'}
CL_ROWS = __CL_ROWS__   # (tokens, role) -> (NW, RT, PIPE, S, prew, V, C)
CL_INSTANCES = tuple(sorted({(1 if t <= 8 else 2 if t <= 16 else 4, rt, pipe, v)
                             for (t, _), (_, rt, pipe, _, _, v, _) in CL_ROWS.items()}))


def ly_label(key):
    b, q, w = key
    return ('d' if q == 1 else 'v') + str(b) + ('S' if w else 'F')


def ly_namespace(rows, bank):
    """Host functions rebound into a copy of its globals; gemv() routes rows to the coded GEMV."""
    g = kernel_fn.__globals__
    v = dict(g)
    for k, f in g.items():
        if isinstance(f, types.FunctionType) and f.__globals__ is g:
            h = types.FunctionType(f.__code__, v, f.__name__, f.__defaults__, f.__closure__)
            h.__kwdefaults__ = f.__kwdefaults__
            h.__dict__.update(f.__dict__)
            v[k] = h
    base_gemv = v['gemv']
    table = {}
    for (t, name), cfg in rows.items():
        c = bank[name]
        table[(t, c['N'], c['K'])] = (c, cfg)

    def gemv(x, w, out_dtype=torch.bfloat16):
        hit = table.get((x.shape[0], w.shape[0], w.shape[1]))
        if hit is None:
            return base_gemv(x, w, out_dtype=out_dtype)
        c, (NW, RT, PIPE, S, prew, V, C) = hit
        return cd_gemv(x, c, S, NW=NW, RT=RT, PIPE=PIPE, prew=prew, pdl=True, V=V, C=C)
    v['gemv'] = gemv
    return v


def ly_state(state, rows):
    plans = {}
    for key, plan in state['best7_plans'].items():
        p = dict(plan)
        if key != 'other' and key[0] * key[1] == 1 and (1, 'o') in rows:
            p['projection'] = {k: c for k, c in plan['projection'].items() if k != 'o'}
        plans[key] = p
    return dict(state, proj_plan={**state['proj_plan'], **{k: 'custom' for k in rows}}, best7_plans=plans)


def ly_copy(dst, src):
    if isinstance(dst, torch.Tensor):
        dst.copy_(src)
        return
    if isinstance(dst, dict):
        for k in dst:
            ly_copy(dst[k], src[k])
    elif isinstance(dst, (list, tuple)):
        for a, b in zip(dst, src):
            ly_copy(a, b)


def ly_graph(fn):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()
    graph.replay()
    torch.cuda.synchronize()
    return graph, out


def ly_tensors(out):
    if isinstance(out, torch.Tensor):
        return [out]
    if isinstance(out, (list, tuple)):
        return [t for o in out for t in ly_tensors(o)]
    return []


def ly_gate(got, want):
    got, want = ly_tensors(got), ly_tensors(want)
    flipped = None
    for a, b in zip(got, want):
        if a.dtype in (torch.int32, torch.int64):
            rows = (a.reshape(a.shape[0], -1).sort(-1).values != b.reshape(b.shape[0], -1).sort(-1).values).any(-1)
            n = int(rows.sum())
            if n > max(1, a.shape[0] // 50):
                return f'ids{n}/{a.shape[0]}'
            flipped = rows if flipped is None else flipped | rows
    worst = 0.
    for a, b in zip(got, want):
        if a.dtype in (torch.int32, torch.int64):
            continue
        if flipped is not None and a.shape[0] == flipped.shape[0] and bool(flipped.any()):
            a, b = a[~flipped], b[~flipped]
        if a.numel() == 0:
            continue
        af, bf = a.float(), b.float()
        if not bool(torch.isfinite(af).all()):
            return 'nonfinite'
        worst = max(worst, float((af - bf).norm() / bf.norm().clamp_min(1e-12)))
    return f'L2={worst:.4f}' if worst > .01 else ''


def ly_time(graphs, rounds):
    names = list(graphs)
    times = {n: [] for n in names}
    for r in range(rounds):
        order = names[r % len(names):] + names[:r % len(names)]
        if r % 2:
            order = order[::-1]
        evs = []
        for n in order:
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            lab_cold_l2()
            s.record()
            graphs[n][0].replay()
            e.record()
            evs.append((n, s, e))
        torch.cuda.synchronize()
        for n, s, e in evs:
            times[n].append(s.elapsed_time(e) * 1000.)
    return times


def lab(deadline, weights, state):
    started = time.monotonic()
    device = state['qkv_gate'].device
    cd_module(CL_INSTANCES)
    bank = {}
    for name in sorted({n for _, n in CL_ROWS}):
        w = state[name]
        c = cd_encode(w)
        if not torch.equal(cd_decode(c).view(torch.int16), w.view(torch.int16)):
            raise RuntimeError(f'coded {name}: decode differs')
        bank[name] = c
    lines = [f'META host state; seeds {len(LY_SEEDS)} from {LY_SEEDS[0]} (+7919); {LY_ROUNDS} rounds rotating order, '
             'lab_cold_l2 before each replay; r = mean per-seed time ratio vs A (<1 = faster than the host); C = all '
             'coded rows, Cq/Cp/Co/Cu = qkv / pool_router / o / unpooler only; bit-exact decode checked; setup '
             f'{time.monotonic() - started:.1f}s; escapes ' + ' '.join(f'{k}={c["nesc"]}' for k, c in bank.items())]
    role_of = {'Cq': ('qkv_gate',), 'Cp': ('pool_router',), 'Co': ('o',), 'Cu': ('unpooler', 'unpooler_slice')}
    ns = {'A': (ly_namespace({}, bank), state), 'C': (ly_namespace(CL_ROWS, bank), ly_state(state, CL_ROWS))}
    for n, roles in role_of.items():
        rows = {k: c for k, c in CL_ROWS.items() if k[1] in roles}
        if rows:
            ns[n] = (ly_namespace(rows, bank), ly_state(state, rows))
    results, incomplete = {}, []
    for key in LY_CASES:
        if time.monotonic() > deadline - 6:
            incomplete.append(ly_label(key))
            continue
        b, q, w = key
        mk = lambda seed: _ap_generate_input(batch=b, ctx=80000, layer='sliding' if w else 'full', q_len=q,  # noqa: E731
                                             spread='agentic' if b > 1 else 'fixed', dist='soft', page=1, seed=seed,
                                             device=device)
        data = mk(LY_SEEDS[0])
        variants = {'A': ns['A'], 'A2': ns['A'], **{n: x for n, x in ns.items() if n != 'A'}}
        graphs, gates = {}, {}
        for n, (v, st) in variants.items():
            try:
                graphs[n] = ly_graph(lambda v=v, st=st: v['kernel_fn'](data, st))
            except Exception as exc:
                gates[n] = 'capture:' + type(exc).__name__ + ':' + str(exc)[:60]
        per_seed = []
        for seed in LY_SEEDS:
            if time.monotonic() > deadline - 4:
                incomplete.append(f'{ly_label(key)}@{seed}')
                break
            if seed != LY_SEEDS[0]:
                ly_copy(data, mk(seed))
            for g, _ in graphs.values():
                g.replay()
            torch.cuda.synchronize()
            for n, (g, out) in graphs.items():
                if n != 'A' and (r := ly_gate(out, graphs['A'][1])):
                    gates.setdefault(n, r)
            times = ly_time(graphs, LY_ROUNDS)
            per_seed.append({n: statistics.mean(x) for n, x in times.items()})
        summ = {}
        for n in (per_seed[0] if per_seed else {}):
            if n == 'A':
                continue
            rs = [s[n] / s['A'] for s in per_seed]
            ds = [s[n] - s['A'] for s in per_seed]
            se = statistics.stdev(rs) / math.sqrt(len(rs)) if len(rs) > 1 else float('nan')
            summ[n] = (statistics.mean(rs), se, statistics.mean(ds), statistics.mean(s['A'] for s in per_seed), len(rs))
        results[key] = summ
        for n, (r, se, d, a_us, k) in sorted(summ.items()):
            lines.append(f'L {ly_label(key)} {n} r={r:.4f} se={se:.4f} d={d:+.2f} A={a_us:.1f} n={k}'
                         + (f' {gates[n]}' if n in gates else ''))
        for n in gates:
            if n not in summ:
                lines.append(f'L {ly_label(key)} {n} FAIL {gates[n]}')
        graphs = variants = None
        gc.collect()
        torch.cuda.empty_cache()
    for (b, q), load in LY_LOADS.items():
        ks = [(b, q, 2048), (b, q, 0)]
        if not all(k in results and results[k] for k in ks):
            continue
        for n in sorted(set(results[ks[0]]) & set(results[ks[1]])):
            num = sum(wt * results[k][n][3] * results[k][n][0] for k, wt in zip(ks, (46, 16)))
            den = sum(wt * results[k][n][3] for k, wt in zip(ks, (46, 16)))
            lines.append(f'LOAD {load} {n} r={num / den:.4f}')
    if incomplete:
        lines.append('INCOMPLETE ' + ' '.join(incomplete))
    lines.append(f'END elapsed={time.monotonic() - started:.1f}s left={deadline - time.monotonic():.1f}s')
    return lines
