"""Whole-layer paired A/B on the host's (S10's) prepared state, identical public inputs, judge-style graph replays.

Variants (each its own CUDA graph of kernel_fn on the same static inputs; every function of the host module rebound
into a copy of its globals with the overrides below, so the host itself is never changed):
  A   the host as prepared;  A2 a second capture of A (A:A noise)
  F   small M (t <= 32): S8's one-LOP3 NT1 (moe_fused<1>, nt1_call(.., 'lop3', 2, 264)) on the same recoded bank
      instead of the flag-templated cell (95 at M=1, 69 at M=8/16/32)
  N   BF16 qkv front off: _attn_sumcast is the identity, so attn_main rows sum + cast with torch and the layer ext
      reads the FP32 splits, as in S8
  FN  both (S8's small-M layer)
  Qc/Ql/Qb  verify 1 only: the qkv projection forced to custom GEMV / cuBLASLt algo 2 / F.linear (the prepare tuner
      picked ('lt', 2) in S8's run and 'custom' in S9's)
Mechanism contexts at t <= 32 (static inputs from the seed's own prefix): RM = route + MoE graph, M = MoE alone,
QAO = qkv projection + attention + O.
Each seed: copy the fixture into the static inputs, replay every graph once and gate it against A (route IDs exact
up to 2% near-tie flips, relative L2 <= 1%; a failing cell is reported, still timed), then AB_ROUNDS rounds, each
replaying every graph once in a rotating order with lab_cold_l2() (1 GiB read) before each replay.
Lines: L <shape> <variant> r=<mean of per-seed time ratios vs A> se=<std error> d=<mean us delta> A=<A us> n=<seeds>
[gate]; LOAD <load> <variant> r=<46 sliding + 16 full weighted time ratio vs A>; C <shape> <context> ...;
P <shape> <plan>.
"""
import gc
import math
import statistics
import time
import types
import torch
from torch.utils.cpp_extension import load_inline

AB_NT1_CUDA = ''
AB_NT1_CPP = ''
AB_SEEDS = tuple(172903 + 7919 * i for i in range(16))
AB_ROUNDS = 8
AB_CASES = ((1, 1, 2048), (1, 1, 0), (1, 8, 2048), (1, 8, 0), (16, 1, 2048), (16, 1, 0), (32, 1, 2048), (32, 1, 0),
            (16, 8, 2048), (16, 8, 0), (32, 8, 2048), (32, 8, 0))
AB_LOADS = {(1, 1): 'decode 1', (16, 1): 'decode 16', (32, 1): 'decode 32', (1, 8): 'verify 1', (16, 8): 'verify 16',
            (32, 8): 'verify 32'}
_AB_NT1 = []


def ab_label(key):
    b, q, w = key
    return ('d' if q == 1 else 'v') + str(b) + ('S' if w else 'F')


def ab_nt1_module():
    if not _AB_NT1:
        import hashlib
        tag = hashlib.sha256(AB_NT1_CUDA.encode()).hexdigest()[:12]
        target = '-gencode=arch=compute_90a,code=sm_90a' if torch.cuda.get_device_capability() == (9, 0) else '-gencode=arch=compute_120,code=sm_120'
        mod = load_inline(name='muse_nt1_limit_lop3' + tag, cpp_sources=AB_NT1_CPP, cuda_sources=AB_NT1_CUDA,
                          functions=['max_smem_attr', 'smem_needed', 'resident_blocks', 'run'], no_implicit_headers=True,
                          with_pytorch_error_handling=False, extra_cuda_cflags=['-O3', '-std=c++20', '-lineinfo', target], verbose=False)
        mod.smem_optin = mod.max_smem_attr()
        _AB_NT1.append(mod)
    return _AB_NT1[0]


def ab_fused(data, st, flags, stages=2, grid=264):
    """S8's nt1_call(data, s4_nt1_bank, 'lop3', 2, 264) on the same bank (flags ignored)."""
    x = data['hidden_states'].contiguous()
    ids = data['topk_ids'].int().contiguous(); tw = data['topk_weights'].float().contiguous()
    m = x.shape[0]; p = m * 8; dev = x.device
    mod = ab_nt1_module()
    if mod.smem_needed(1, 2, m) > mod.smem_optin: raise ValueError('lop3 stages2 shared memory exceeds device capacity')
    grid = min(264, st['sms'] * mod.resident_blocks(1, 2, m))
    bank = st['banks']['lop3']
    inter = torch.empty((p, 4096), device=dev, dtype=torch.bfloat16)
    y = torch.empty((p, 4096), device=dev); ss = torch.empty((p,), device=dev); out = torch.empty_like(x)
    cnt = st.setdefault('ab_lop3_counter', torch.zeros(64, device=dev, dtype=torch.int32))
    tensors = [x, ids, tw, st['pnw'], bank['w13p'], bank['w2p'], st['gs13'], st['gs2'], inter, y, ss, cnt, out]
    mod.run([v.data_ptr() for v in tensors], m, data['config']['local_start'], 2, grid, torch.cuda.current_stream(dev).cuda_stream)
    return out


def ab_namespace(overrides):
    """Every plain function of the host module rebound into a copy of its globals, then the overrides."""
    g = kernel_fn.__globals__
    v = dict(g)
    for k, f in g.items():
        if isinstance(f, types.FunctionType) and f.__globals__ is g:
            h = types.FunctionType(f.__code__, v, f.__name__, f.__defaults__, f.__closure__)
            h.__kwdefaults__ = f.__kwdefaults__
            h.__dict__.update(f.__dict__)
            v[k] = h
    v.update(overrides)
    return v


def ab_copy(dst, src):
    if isinstance(dst, torch.Tensor):
        if dst.shape != src.shape:raise RuntimeError(f'fixture shape {tuple(src.shape)} != {tuple(dst.shape)}')
        dst.copy_(src);return
    if isinstance(dst, dict):
        for k in dst:ab_copy(dst[k], src[k])
    elif isinstance(dst, (list, tuple)):
        for a, b in zip(dst, src):ab_copy(a, b)


def ab_graph(fn):
    for _ in range(2):fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):out = fn()
    graph.replay();torch.cuda.synchronize()
    return graph, out


def ab_tensors(out):
    if isinstance(out, torch.Tensor):return [out]
    if isinstance(out, (list, tuple)):return [t for o in out for t in ab_tensors(o)]
    return []


def ab_gate(got, want):
    """'' if close; else a short reason. Near-tie route flips (<= 2% of tokens) leave the float comparison."""
    got, want = ab_tensors(got), ab_tensors(want)
    flipped = None
    for a, b in zip(got, want):
        if a.dtype in (torch.int32, torch.int64):
            rows = (a.reshape(a.shape[0], -1).sort(-1).values != b.reshape(b.shape[0], -1).sort(-1).values).any(-1)
            n = int(rows.sum())
            if n > max(1, a.shape[0] // 50):return f'ids{n}/{a.shape[0]}'
            flipped = rows if flipped is None else flipped | rows
    worst = 0.
    for a, b in zip(got, want):
        if a.dtype in (torch.int32, torch.int64):continue
        if flipped is not None and a.shape[0] == flipped.shape[0] and bool(flipped.any()):a, b = a[~flipped], b[~flipped]
        if a.numel() == 0:continue
        af, bf = a.float(), b.float()
        if not bool(torch.isfinite(af).all()):return 'nonfinite'
        worst = max(worst, float((af - bf).norm() / bf.norm().clamp_min(1e-12)))
    return f'L2={worst:.4f}' if worst > .01 else ''


def ab_time(graphs, rounds):
    """name -> per-round us; every round replays each graph once in a rotating order, cold L2 before each."""
    names = list(graphs);times = {n: [] for n in names}
    for r in range(rounds):
        order = names[r % len(names):] + names[:r % len(names)]
        if r % 2:order = order[::-1]
        evs = []
        for n in order:
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            lab_cold_l2();s.record();graphs[n][0].replay();e.record();evs.append((n, s, e))
        torch.cuda.synchronize()
        for n, s, e in evs:times[n].append(s.elapsed_time(e) * 1000.)
    return times


def ab_summary(per_seed, base='A'):
    """per_seed: [{name: mean us}] -> {name: (mean ratio, se, mean delta us, A us, n)}."""
    out = {}
    for n in per_seed[0]:
        if n == base:continue
        rs = [s[n] / s[base] for s in per_seed if n in s];ds = [s[n] - s[base] for s in per_seed if n in s]
        se = statistics.stdev(rs) / math.sqrt(len(rs)) if len(rs) > 1 else float('nan')
        out[n] = (statistics.mean(rs), se, statistics.mean(ds), statistics.mean(s[base] for s in per_seed), len(rs))
    return out


def ab_prefix(v, data, state, plan):
    qkv = v['_best7_project'](data['attn_input'], 'qkv_gate', state, plan)
    attn = v['_best7_attention'](qkv, data, state, plan)
    a = v['_best7_project'](attn, 'o', state, plan)
    h, _ = v['_best7_boundary'](a, data['stream'], state, plan, 'b1')
    return v['_best7_project'](h, 'pool_router', state, plan, out_dtype=torch.float32)


def ab_rm(v, pr, data, state, plan):
    x, _, wts, lids = v['_best7_route'](pr, state, data, plan)
    return v['_best7_expert'](pr, x, wts, lids, state, plan)


def ab_qao(v, data, state, plan):
    qkv = v['_best7_project'](data['attn_input'], 'qkv_gate', state, plan)
    attn = v['_best7_attention'](qkv, data, state, plan)
    return v['_best7_project'](attn, 'o', state, plan)


def ab_plan_line(key, state):
    plan = state['best7_plans'][key];t = key[0] * key[1]
    roles = ('qkv_gate', 'o', 'pool_router', 'unpooler_slice' if t <= 8 else 'unpooler')
    proj = {r: plan['projection'].get(r, 'auto') for r in roles}
    proj = {r: state['proj_plan'].get((t, r), 'custom') if c == 'auto' else c for r, c in proj.items()}
    attn = plan['attention'] if plan['attention'] != 'auto' else state['attention_plan'].get((key[0], key[1], bool(key[2])), ('layer', None))
    short = lambda c: c if isinstance(c, str) else '/'.join(map(str, c))
    return (f'P {ab_label(key)} moe={plan["moe"]} attn={short(attn)} glue={plan["glue"]} '
            + ' '.join(f'{r[:4]}={short(c)}' for r, c in proj.items()))


def lab(deadline, weights, state):
    started = time.monotonic();device = weights['qkv_gate'].device
    lines = [f'META host S10 state; seeds {len(AB_SEEDS)} from {AB_SEEDS[0]} (+7919); {AB_ROUNDS} rounds rotating order, '
             'lab_cold_l2 before each replay; r = mean per-seed time ratio vs A (<1 = faster than the host)',
             'META F=S8 moe_fused small M, N=no BF16 front, FN=both, A2=A:A, Q*=verify-1 qkv choice; C ctx RM=route+MoE M=MoE QAO=qkv+attn+O']
    ab_nt1_module()
    lines.append(f'SETUP nt1 {time.monotonic() - started:.1f}s left={deadline - time.monotonic():.0f}s')
    ns = {'A': ab_namespace({}), 'F': ab_namespace({'smallm_experts': ab_fused}),
          'N': ab_namespace({'_attn_sumcast': lambda p, s: (p, s)}),
          'FN': ab_namespace({'smallm_experts': ab_fused, '_attn_sumcast': lambda p, s: (p, s)})}
    results = {};chains = [];incomplete = []
    for key in AB_CASES:
        if time.monotonic() > deadline - 6:incomplete.append(ab_label(key));continue
        b, q, w = key;t = b * q;plan = state['best7_plans'][key]
        lines.append(ab_plan_line(key, state))
        mk = lambda seed: _ap_generate_input(batch=b, ctx=80000, layer='sliding' if w else 'full', q_len=q,
                                             spread='agentic' if b > 1 else 'fixed', dist='soft', page=1, seed=seed, device=device)
        data = mk(AB_SEEDS[0])
        variants = {'A': (ns['A'], state), 'A2': (ns['A'], state), 'N': (ns['N'], state)}
        if t <= 32:variants.update(F=(ns['F'], state), FN=(ns['FN'], state))
        if t == 8:
            cur = state['proj_plan'].get((8, 'qkv_gate'), 'custom')
            alts = {'Qc': 'custom', 'Qb': 'cublas'}
            if state.get('lt_mod') is not None and len(state.get('lt_algos', {}).get(('qkv_gate', 8), ())) > 2:alts['Ql'] = ('lt', 2)
            for n, c in alts.items():
                if c != cur:variants[n] = (ns['A'], dict(state, proj_plan={**state['proj_plan'], (8, 'qkv_gate'): c}))
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
        # Mechanism contexts on the same fixture family (t <= 32).
        if t <= 32 and time.monotonic() < deadline - 10:
            try:chains += ab_contexts(key, plan, ns, state, mk, deadline)
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
    lines.append(f'END shapes={len(results)}/12 elapsed={time.monotonic() - started:.1f}s left={deadline - time.monotonic():.1f}s')
    return lines


def ab_contexts(key, plan, ns, state, mk, deadline):
    """RM (route+MoE) and M (MoE alone) for A vs F; QAO (qkv+attn+O) for A vs N; static inputs per seed."""
    out = [];va, vf, vn = ns['A'], ns['F'], ns['N']
    data = mk(AB_SEEDS[0])
    pr = ab_prefix(va, data, state, plan);pr_s = (pr[0].clone(), pr[1])
    x, ids, wts, lids = va['_best7_route'](pr_s, state, data, plan)
    stat = [t.clone() for t in (x, wts, lids)]
    ctx = {
        'RM': {n: ab_graph(lambda v=v: ab_rm(v, pr_s, data, state, plan)) for n, v in (('A', va), ('F', vf))},
        'M': {n: ab_graph(lambda v=v: v['_best7_expert'](pr_s, stat[0], stat[1], stat[2], state, plan)) for n, v in (('A', va), ('F', vf))},
        'QAO': {n: ab_graph(lambda v=v: ab_qao(v, data, state, plan)) for n, v in (('A', va), ('N', vn))},
    }
    seeds = {c: [] for c in ctx}
    for seed in AB_SEEDS[:8]:
        if time.monotonic() > deadline - 4:break
        if seed != AB_SEEDS[0]:
            ab_copy(data, mk(seed))
            p2 = ab_prefix(va, data, state, plan);pr_s[0].copy_(p2[0])
            x2, _, w2, l2 = va['_best7_route'](pr_s, state, data, plan)
            for d, s in zip(stat, (x2, w2, l2)):d.copy_(s)
        torch.cuda.synchronize()
        for c, graphs in ctx.items():
            times = ab_time(graphs, 6)
            seeds[c].append({n: statistics.mean(v) for n, v in times.items()})
    for c, rows in seeds.items():
        if not rows:continue
        for n, (r, se, d, a_us, k) in ab_summary(rows).items():
            out.append(f'C {ab_label(key)} {c} {n} r={r:.4f} se={se:.4f} d={d:+.2f} A={a_us:.1f} n={k}')
    return out


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
