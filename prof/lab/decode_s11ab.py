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


"""S11 vs S10, whole layer, paired, leave-one-out on S11's prepared state (appended to dev/s10/ablate_lab.py.inc, whose
helpers it uses; this lab() replaces that one). Each arm is its own CUDA graph of kernel_fn on the same static inputs,
the host's functions rebound into a copy of its globals (the host never changes):
  A   S11 as prepared;  A2 a second capture (A:A)
  V   balanced split-K GEMV off: every balanced row back to S10's V9 tile / cuBLAS choice      (t <= 32)
  Q   vq8 off: the routed verify rows (1 sliding, 16 full, 32 full) back to S10's attention     (routed rows)
  K   S11's N_full heavy kernel off: S10's N_full cell (NF_CUDA kinds 0/1, nf_plan, nf_call; embedded verbatim,
      compiled in a thread at lab start) on S10's s5_hh_base (issue_prepare without nf_ilv's W13 row interleave),
      in place of S11's                                                                         (M = 128 / 256)
  S10 all of V, Q, K at once (S10's kernels on S11's pinned plan)
Seeds AB_SEEDS x AB_ROUNDS rounds, rotating order, lab_cold_l2 before each replay. Lines: L, LOAD (46 sliding + 16
full weighted), END; r = mean per-seed time ratio vs A (> 1: the reverted arm is slower, i.e. S11's change wins).
"""
import threading

S10_NF_BLOCK = '# MoE lane H200 c01b1916e2c74af8 / 14f942d57e764ebf / 880c388117ed4e16: N_full at M=128 (paired 0.8842 vs S7\'s H_H,\n# bit-equal) and M=256 (0.8701 vs mixed_parallel); heavy on the side stream, launched first: a further 0.9799 / 0.9882.\nNF_CUDA=\'#define BR_KINDS(F) F(0,32,true,true,0,0,0) F(1,32,false,true,0,0,0)\\n// Branch/remainder lab kernels. Banks stay packed E2M1 plus uint8 exponent/mantissa scale codes\\n// (sign/zero folded into the codes); scales convert to BF16 inside the timed kernel.\\n// One task = one 32-row group of one expert x one 64-channel M tile (route NT=32 numbering).\\n// A kernel instance owns the groups whose expert count (FM=1) or rows-in-group (FM=2) lies in (LO,HI].\\n#include <cuda_runtime.h>\\n#include <cuda_bf16.h>\\n#include <stdint.h>\\n#include <vector>\\n#include <stdexcept>\\n#define BR_CUDA_CHECK(expr) do{cudaError_t br_s=(expr);if(br_s!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(br_s));}while(0)\\n#define BR_REQUIRE(c) do{if(!(c))throw std::runtime_error("branch lab argument check");}while(0)\\n#if defined(__CUDA_ARCH__) && __CUDA_ARCH__==900 && defined(__CUDA_ARCH_FEAT_SM90_ALL)\\n#define BR_HOPPER 1\\n#else\\n#define BR_HOPPER 0\\n#endif\\n\\nnamespace mwbr {\\nconstexpr int BK=64,KT=128,S=4,BIG=1<<30;\\n__device__ __forceinline__ unsigned smaddr(const void*p){return (unsigned)__cvta_generic_to_shared(p);}\\n__device__ __forceinline__ void async_copy(void*d,const void*s,int bytes,int valid){\\n unsigned dst=smaddr(d);\\n if(bytes==16)asm volatile("cp.async.cg.shared.global [%0],[%1],16,%2;"::"r"(dst),"l"(s),"r"(valid?16:0):"memory");\\n else if(bytes==8)asm volatile("cp.async.ca.shared.global [%0],[%1],8;"::"r"(dst),"l"(s):"memory");\\n else asm volatile("cp.async.ca.shared.global [%0],[%1],4;"::"r"(dst),"l"(s):"memory");\\n}\\n__device__ __forceinline__ unsigned mul2(unsigned a,unsigned b){unsigned c;asm("mul.rn.bf16x2 %0,%1,%2;":"=r"(c):"r"(a),"r"(b));return c;}\\n__device__ __forceinline__ unsigned rr(unsigned a,int n){return __funnelshift_r(a,a,n);}\\n__device__ __forceinline__ void dq(unsigned a,unsigned b,unsigned raw,unsigned*out){\\n unsigned w;asm("lop3.b32 %0,%1,0x60306030,%2,0xe2;":"=r"(w):"r"(a),"r"(b));unsigned mask=0x81c081c0u;\\n unsigned s0=__byte_perm(raw,0u,0x4040)*16u+0x70007000u,s1=__byte_perm(raw,0u,0x4141)*16u+0x70007000u;\\n unsigned s2=__byte_perm(raw,0u,0x4242)*16u+0x70007000u,s3=__byte_perm(raw,0u,0x4343)*16u+0x70007000u;\\n out[0]=mul2(a&mask,s0);out[1]=mul2(rr(a,4)&mask,s1);out[2]=mul2(rr(a,10)&mask,s0);out[3]=mul2(rr(b,2)&mask,s1);\\n out[4]=mul2(rr(b,6)&mask,s2);out[5]=mul2(rr(b,12)&mask,s3);out[6]=mul2(rr(w,24)&mask,s2);out[7]=mul2(rr(w,30)&mask,s3);\\n}\\n__device__ __forceinline__ void keep(const unsigned*a){\\n asm volatile(""::"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(a[4]),"r"(a[5]),"r"(a[6]),"r"(a[7]),\\n  "r"(a[8]),"r"(a[9]),"r"(a[10]),"r"(a[11]),"r"(a[12]),"r"(a[13]),"r"(a[14]),"r"(a[15]));\\n}\\ntemplate<int NT>__device__ __forceinline__ void mma(float*c,const unsigned*a,const __nv_bfloat16*xs,int b){\\n#if BR_HOPPER\\n uint64_t d=(uint64_t)((smaddr(xs)&0x3ffff)>>4)|((uint64_t)64<<32)|((uint64_t)1<<62);\\n d+=b*2;\\n if constexpr(NT==8){\\n asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 "\\n "{%0,%1,%2,%3},{%4,%5,%6,%7},%8,p,1,1,0;}"\\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(d));\\n }else if constexpr(NT==16){\\n asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 "\\n "{%0,%1,%2,%3,%4,%5,%6,%7},{%8,%9,%10,%11},%12,p,1,1,0;}"\\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7])\\n :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(d));\\n }else{\\n static_assert(NT==32,"tile width");\\n asm volatile("{ .reg .pred p; setp.ne.b32 p,1,0; wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 "\\n "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15},{%16,%17,%18,%19},%20,p,1,1,0;}"\\n :"+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3]),"+f"(c[4]),"+f"(c[5]),"+f"(c[6]),"+f"(c[7]),\\n  "+f"(c[8]),"+f"(c[9]),"+f"(c[10]),"+f"(c[11]),"+f"(c[12]),"+f"(c[13]),"+f"(c[14]),"+f"(c[15])\\n :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"l"(d));\\n }\\n#else\\n // Portable stand-in: identical shared layout and fragment ownership, immediate mma.sync.\\n int lane=threadIdx.x&31;\\n #pragma unroll\\n for(int j=0;j<NT/8;++j){int r=j*8+(lane>>2),k=b*16+(lane&3)*2;\\n  unsigned b0=*(const unsigned*)(xs+r*BK+(k^(((r+(smaddr(xs)>>7))&7)*8)));\\n  unsigned b1=*(const unsigned*)(xs+r*BK+((k+8)^(((r+(smaddr(xs)>>7))&7)*8)));\\n  float*p=c+4*j;\\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"\\n  :"+f"(p[0]),"+f"(p[1]),"+f"(p[2]),"+f"(p[3]):"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b0),"r"(b1));\\n }\\n#endif\\n}\\ntemplate<int NT>__device__ __forceinline__ void mma_group(float*c,const unsigned*a,const __nv_bfloat16*xs){\\n#pragma unroll\\n for(int b=0;b<4;++b)mma<NT>(c,a+b*4,xs,b);\\n#if BR_HOPPER\\n asm volatile("wgmma.commit_group.sync.aligned;":::"memory");\\n asm volatile("wgmma.wait_group.sync.aligned 1;":::"memory");\\n#endif\\n}\\ntemplate<int NT,bool VEC>__device__ __forceinline__ void load_stage(\\n unsigned char*q,unsigned char*s,__nv_bfloat16*bs,const int*Q,const unsigned char*SC,\\n const __nv_bfloat16*X,const int*src,int e,int mt,int N,int tile){\\n constexpr int NJ=NT>=16?NT/16:1;\\n int t=threadIdx.x,w=t>>5,l=t&31;\\n if constexpr(VEC){\\n  if(t<64){unsigned group=t>>4,lane=t&15;\\n   unsigned base=((unsigned)e*(N/16)+mt*4+group)*128*64+tile*(KT/32)*64+lane*4;\\n   #pragma unroll\\n   for(int g=0;g<KT/32;++g){unsigned dst=smaddr(q+g*1024+t*16);const void*sp=Q+base+g*64;\\n    asm volatile("cp.async.ca.shared.global [%0],[%1],16;"::"r"(dst),"l"(sp):"memory");}\\n  }\\n  if(t>=64&&t<64+KT/4){unsigned i=t-64,g=i/8,group=(i%8)/2,half=i%2;\\n   unsigned srcidx=(((unsigned)e*(N/16)+mt*4+group)*128+tile*(KT/32)+g)*32+half*16;\\n   unsigned dst=smaddr(s+g*128+group*32+half*16);const void*sp=SC+srcidx;\\n   asm volatile("cp.async.ca.shared.global [%0],[%1],16;"::"r"(dst),"l"(sp):"memory");\\n  }\\n }else{\\n  #pragma unroll\\n  for(int g=0;g<KT/32;++g){size_t z=((size_t)e*(N/16)+mt*4+w)*128+tile*(KT/32)+g;\\n   async_copy(q+(g*128+t)*8,Q+z*64+l*2,8,1);\\n   if(l<8)async_copy(s+g*128+w*32+l*4,SC+z*32+l*4,4,1);\\n  }\\n }\\n #pragma unroll\\n for(int c=0;c<KT/64;++c){\\n  #pragma unroll\\n  for(int j=0;j<NJ;++j){int r=t/8+j*16,k=(t&7)*8;\\n   if(NT>=16||r<NT)async_copy(bs+c*NT*64+r*64+(k^((r&7)*8)),X+(size_t)max(src[j],0)*4096+tile*KT+c*64+k,16,src[j]>=0);\\n  }\\n }\\n asm volatile("cp.async.commit_group;":::"memory");\\n}\\ntemplate<int NS>__device__ __forceinline__ void cp_wait(int k){\\n if(k==NS-1)asm volatile("cp.async.wait_group 0;":::"memory");\\n else if(k==NS-2)asm volatile("cp.async.wait_group 1;":::"memory");\\n else asm volatile("cp.async.wait_group %0;"::"n"(S-2):"memory");\\n}\\ntemplate<int NT>constexpr int smem_bytes(){return 1023+((S*(64*KT/2+64*KT/16)+1023)&~1023)+S*NT*KT*2;}\\n\\ntemplate<int NT,bool FIRST,bool VEC,int FM,int LO,int HI>\\n__global__ void __launch_bounds__(128,2) gemm(const int*Q,const unsigned char*SC,const float*GS,\\n const __nv_bfloat16*X,const int*C,const int*OFF,const int*ORDER,const int*TASK,const int*SLOTS,\\n const __nv_bfloat16*PNW,__nv_bfloat16*ACT,float*Y,float*SS,int M){\\n constexpr int QB=64*KT/2,SB=64*KT/16,NS=4096/KT,N=FIRST?8192:4096,MT=N/64,NJ=NT>=16?NT/16:1;\\n constexpr int BS=((S*(QB+SB)+1023)&~1023);\\n extern __shared__ __align__(1024) unsigned char typed_sm[];\\n unsigned char*qs=typed_sm,*ss=qs+S*QB;auto*bs=(__nv_bfloat16*)(typed_sm+BS);\\n __shared__ __nv_bfloat16 epi[FIRST?64*NT:1];__shared__ float rowsum[FIRST?1:4*NT];\\n int t=threadIdx.x,w=t>>5,l=t&31;\\n int total=TASK[32]*MT;\\n for(int unit=blockIdx.x;unit<total;unit+=gridDim.x){\\n  int group=unit/MT,mt=unit%MT,ix=0;\\n  #pragma unroll\\n  for(int d=16;d;d>>=1)if(TASK[ix+d]<=group)ix+=d;\\n  int e=ORDER[ix],row=(group-TASK[ix])*32,count=C[e],offset=OFF[e];\\n  if constexpr(FM==1){if(count<=LO||count>HI)continue;}\\n  if constexpr(FM==2){int rem=min(count-row,32);if(rem<=LO||rem>HI)continue;}\\n  int src[NJ];\\n  #pragma unroll\\n  for(int j=0;j<NJ;++j){int rl=t/8+j*16,r=row+rl;src[j]=(rl<NT&&r<count)?(FIRST?SLOTS[e*M+r]/8:offset+r):-1;}\\n  float acc[NT/2]={};unsigned af[2][16]={};\\n  #pragma unroll\\n  for(int j=0;j<NT/2;++j)asm volatile("":"+f"(acc[j]));\\n  #pragma unroll\\n  for(int i=0;i<S-1;++i)load_stage<NT,VEC>(qs+i*QB,ss+i*SB,bs+i*NT*KT,Q,SC,X,src,e,mt,N,i);\\n  #pragma unroll 1\\n  for(int k=0;k<NS;++k){\\n   int st=k%S;cp_wait<NS>(k);__syncthreads();\\n#if BR_HOPPER\\n   asm volatile("fence.proxy.async.shared::cta;":::"memory");\\n#endif\\n   #pragma unroll\\n   for(int sub=0;sub<2;++sub){\\n    #pragma unroll\\n    for(int g=0;g<2;++g){int z=sub*2+g;uint2 q=*(const uint2*)(qs+st*QB+(z*128+t)*8);\\n     unsigned sf=*(const unsigned*)(ss+st*SB+z*128+w*32+(l>>2)*4);dq(q.x,q.y,sf,af[sub]+g*8);}\\n    keep(af[sub]);\\n#if BR_HOPPER\\n    asm volatile("wgmma.fence.sync.aligned;":::"memory");\\n#endif\\n    mma_group<NT>(acc,af[sub],bs+st*NT*KT+sub*NT*64);keep(af[sub^1]);\\n    if(sub==0&&k+S-1<NS){int dst=(k+S-1)%S;\\n     load_stage<NT,VEC>(qs+dst*QB,ss+dst*SB,bs+dst*NT*KT,Q,SC,X,src,e,mt,N,k+S-1);}\\n   }\\n  }\\n#if BR_HOPPER\\n  asm volatile("wgmma.wait_group.sync.aligned 0;":::"memory");\\n#endif\\n  keep(af[0]);keep(af[1]);\\n  if constexpr(FIRST){\\n   #pragma unroll\\n   for(int j=0;j<NT/2;++j){int r=(l&3)*2+(j&1)+(j>>2)*8;\\n    int c=w*16+(l>>2)+((j>>1)&1)*8;float gs=GS[e+(c>=32?32:0)]*256.f;\\n    epi[c*NT+r]=__float2bfloat16_rn(acc[j]*gs);\\n   }\\n   __syncthreads();\\n   #pragma unroll\\n   for(int i=t;i<NT*32;i+=128){int r=i/32,c=i%32;\\n    float g=__bfloat162float(epi[c*NT+r]),u=__bfloat162float(epi[(c+32)*NT+r]);\\n    if(row+r<count)ACT[(size_t)(offset+row+r)*4096+mt*32+c]=__float2bfloat16_rn(g/(1.f+__expf(-g))*u);\\n   }\\n  }else{\\n   float gs=GS[e]*256.f;\\n   #pragma unroll\\n   for(int j=0;j<NT/2;++j){int r=row+(l&3)*2+(j&1)+(j>>2)*8;\\n    int c=mt*64+w*16+(l>>2)+((j>>1)&1)*8;\\n    float z=acc[j]*gs*(1.f+__bfloat162float(PNW[c]));acc[j]=z;\\n    if(r<count)Y[(size_t)(offset+r)*4096+c]=z;\\n   }\\n   #pragma unroll\\n   for(int i=0;i<NT/8;++i){float a=acc[4*i]*acc[4*i]+acc[4*i+2]*acc[4*i+2];\\n    float b=acc[4*i+1]*acc[4*i+1]+acc[4*i+3]*acc[4*i+3];\\n    #pragma unroll\\n    for(int d=4;d<=16;d*=2){a+=__shfl_xor_sync(~0u,a,d);b+=__shfl_xor_sync(~0u,b,d);}\\n    if(l<4){rowsum[w*NT+i*8+2*l]=a;rowsum[w*NT+i*8+2*l+1]=b;}\\n   }\\n   __syncthreads();\\n   if(t<NT&&row+t<count){float sum=0;\\n    #pragma unroll\\n    for(int ww=0;ww<4;++ww)sum+=rowsum[ww*NT+t];\\n    SS[(size_t)(offset+row+t)*64+mt]=sum;\\n   }\\n  }\\n  __syncthreads();\\n }\\n}\\n\\n// Routing only: no output fill, no readiness state. One CTA owns every pair (same as the\\n// original CTA 0); with META it also writes the heavy-only (count>16) 32-row task prefix T32 and\\n// the one-group-per-expert prefix T8 of experts with 8<count<=16.\\ntemplate<bool META>__global__ void __launch_bounds__(256) route_live(const int*ids,int*C,int*OFF,int*ORDER,int*TASK,int*SLOTS,int*POS,int*T32,\\n int*T8,int M,int NT,int start){\\n int t=threadIdx.x;\\n __shared__ int count[32];if(t<32)count[t]=0;__syncthreads();\\n // All eight cold id loads per thread are issued before the first shared atomic.\\n for(int base=0;base<M*8;base+=256*8){int v[8];\\n  #pragma unroll\\n  for(int k=0;k<8;++k){int i=base+k*256+t;v[k]=i<M*8?ids[i]:start-1;}\\n  #pragma unroll\\n  for(int k=0;k<8;++k){int i=base+k*256+t,e=v[k]-start,rank=-1;\\n   if(e>=0&&e<32){rank=atomicAdd(count+e,1);SLOTS[e*M+rank]=i;}\\n   if(i<M*8)POS[i]=rank;}\\n }\\n __syncthreads();\\n if(t<32){int n=count[t],pad=(n+NT-1)/NT*NT;C[t]=n;\\n  int scan=pad;\\n  #pragma unroll\\n  for(int d=1;d<32;d*=2){int a=__shfl_up_sync(~0u,scan,d);if(t>=d)scan+=a;}\\n  OFF[t]=scan-pad;if(t==31)OFF[32]=scan;\\n  int key=n*32+31-t;\\n  #pragma unroll\\n  for(int k=2;k<=32;k*=2){\\n   #pragma unroll\\n   for(int j=k/2;j;j/=2){int other=__shfl_xor_sync(~0u,key,j);\\n    bool ascending=(t&k)==0,low=(t&j)==0;key=(ascending==low)?min(key,other):max(key,other);\\n   }\\n  }\\n  key=__shfl_sync(~0u,key,31-t);ORDER[t]=31-key%32;\\n  int blocks=(key/32+NT-1)/NT,ends=blocks;\\n  #pragma unroll\\n  for(int d=1;d<32;d*=2){int a=__shfl_up_sync(~0u,ends,d);if(t>=d)ends+=a;}\\n  TASK[t+1]=ends;if(t==0)TASK[0]=0;\\n  if constexpr(META){int ns=key/32,v=ns>16?(ns+31)/32:0,u=ns>8&&ns<=16;\\n   #pragma unroll\\n   for(int d=1;d<32;d*=2){int a=__shfl_up_sync(~0u,v,d),b=__shfl_up_sync(~0u,u,d);if(t>=d){v+=a;u+=b;}}\\n   T32[t+1]=v;T8[t+1]=u;if(t==0){T32[0]=0;T8[0]=0;}}\\n }\\n}\\n\\n__global__ void finish(const int*__restrict__ OFF,const int*__restrict__ IDS,const float*__restrict__ TW,const int*__restrict__ POS,const float*__restrict__ Y,const float*__restrict__ SS,__nv_bfloat16*__restrict__ OUT,int M,int start){\\n int tok=blockIdx.x,t=threadIdx.x,w=t/32,l=t%32;__shared__ int base[8];__shared__ float alpha[8];\\n int e=IDS[tok*8+w]-start,p=-1;float sum=0;\\n if(e>=0&&e<32){p=OFF[e]+POS[tok*8+w];sum=SS[(size_t)p*64+l]+SS[(size_t)p*64+l+32];}\\n for(int d=16;d;d/=2)sum+=__shfl_down_sync(~0u,sum,d);\\n if(l==0){base[w]=p;alpha[w]=p>=0?rsqrtf(sum/4096.f+1.e-8f)*TW[tok*8+w]:0.f;}\\n __syncthreads();int pp[8];float aa[8];\\n #pragma unroll\\n for(int j=0;j<8;++j){pp[j]=base[j];aa[j]=alpha[j];}\\n for(int c=t*4;c<4096;c+=1024){float4 sum=make_float4(0,0,0,0);\\n  #pragma unroll\\n  for(int j=0;j<8;++j)if(pp[j]>=0){float4 v=*(const float4*)(Y+(size_t)pp[j]*4096+c);float a=aa[j];sum.x+=v.x*a;sum.y+=v.y*a;sum.z+=v.z*a;sum.w+=v.w*a;}\\n  __nv_bfloat162 a=__floats2bfloat162_rn(sum.x,sum.y),b=__floats2bfloat162_rn(sum.z,sum.w);\\n  *(uint2*)(OUT+(size_t)tok*4096+c)=make_uint2(*(unsigned*)&a,*(unsigned*)&b);\\n }\\n}\\n}  // namespace mwbr\\n\\n// Pointer ABI for gemm: Q,SC,GS,X,C,OFF,ORDER,TASK,SLOTS,PNW,ACT,Y,SS.\\n#define BR_ARGS(v) (const int*)v[0],(const unsigned char*)v[1],(const float*)v[2],(const __nv_bfloat16*)v[3],(const int*)v[4],(const int*)v[5],(const int*)v[6],(const int*)v[7],(const int*)v[8],(const __nv_bfloat16*)v[9],(__nv_bfloat16*)v[10],(float*)v[11],(float*)v[12]\\n#define BR_INIT(ID,NT,F,V,FM,LO,HI) BR_CUDA_CHECK(cudaFuncSetAttribute(mwbr::gemm<NT,F,V,FM,LO,HI>,cudaFuncAttributeMaxDynamicSharedMemorySize,mwbr::smem_bytes<NT>()));\\n#define BR_GO(ID,NT,F,V,FM,LO,HI) case ID:mwbr::gemm<NT,F,V,FM,LO,HI><<<grid,128,mwbr::smem_bytes<NT>(),(cudaStream_t)stream>>>(BR_ARGS(v),(int)M);break;\\n#define BR_INFO(ID,NT,F,V,FM,LO,HI) case ID:{cudaFuncAttributes a{};int n;BR_CUDA_CHECK(cudaFuncGetAttributes(&a,mwbr::gemm<NT,F,V,FM,LO,HI>));BR_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,mwbr::gemm<NT,F,V,FM,LO,HI>,128,mwbr::smem_bytes<NT>()));return {a.numRegs,(int64_t)a.localSizeBytes,(int64_t)a.sharedSizeBytes,(int64_t)mwbr::smem_bytes<NT>(),n};}\\nint64_t br_init(){BR_KINDS(BR_INIT) return 0;}\\nstd::vector<int64_t> br_info(int64_t kind){switch(kind){BR_KINDS(BR_INFO) default:BR_REQUIRE(false);}return {};}\\nvoid br_run(std::vector<int64_t>v,int64_t kind,int64_t grid,int64_t M,int64_t stream){\\n BR_REQUIRE(v.size()==13);\\n switch(kind){BR_KINDS(BR_GO) default:BR_REQUIRE(false);}\\n BR_CUDA_CHECK(cudaGetLastError());\\n}\\n// v: ids,C,OFF,ORDER,TASK,SLOTS,POS,T32,T8 (T32/T8 written only with meta)\\nvoid br_route(std::vector<int64_t>v,int64_t M,int64_t start,int64_t meta,int64_t stream){\\n BR_REQUIRE(v.size()==9);\\n if(meta)mwbr::route_live<true><<<1,256,0,(cudaStream_t)stream>>>((const int*)v[0],(int*)v[1],(int*)v[2],(int*)v[3],(int*)v[4],(int*)v[5],(int*)v[6],(int*)v[7],(int*)v[8],(int)M,32,(int)start);\\n else mwbr::route_live<false><<<1,256,0,(cudaStream_t)stream>>>((const int*)v[0],(int*)v[1],(int*)v[2],(int*)v[3],(int*)v[4],(int*)v[5],(int*)v[6],(int*)v[7],(int*)v[8],(int)M,32,(int)start);\\n BR_CUDA_CHECK(cudaGetLastError());\\n}\\n// v: OFF,IDS,TW,POS,Y,SS,OUT\\nvoid br_finish(std::vector<int64_t>v,int64_t M,int64_t start,int64_t stream){\\n BR_REQUIRE(v.size()==7);\\n mwbr::finish<<<M,256,0,(cudaStream_t)stream>>>((const int*)v[0],(const int*)v[1],(const float*)v[2],(const int*)v[3],(const float*)v[4],(const float*)v[5],(__nv_bfloat16*)v[6],(int)M,(int)start);\\n BR_CUDA_CHECK(cudaGetLastError());\\n}\\n\\nnamespace pmz {\\n\\nconstexpr int HD = 4096;\\nconstexpr int NLOC = 32;\\nconstexpr int TILE_BYTES = 36864;\\nconstexpr int TILE_W_BYTES = 32768;\\nconstexpr int NT1 = 512;\\nconstexpr int NT2 = 256;\\nconstexpr int NCW = 8;                 // consumer warps\\nconstexpr int NCT = NCW * 32;          // consumer threads\\nconstexpr int NTHR = NCT;             // thread 0 also prefetches\\nconstexpr uint32_t DQ_MASK = 0x81C081C0u;\\nconstexpr uint32_t LA_MASK = 0x60306030u;\\n// persistent counters\\nenum { C_DONE = 0, C_ZERO = 32, C_BAR = 33, C_GEN = 34 };\\n\\n__device__ __forceinline__ uint32_t smem_u32(const void* p) {\\n  return static_cast<uint32_t>(__cvta_generic_to_shared(p));\\n}\\n__device__ __forceinline__ void mbar_init(uint64_t* b, int cnt) {\\n  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_u32(b)), "r"(cnt));\\n}\\n__device__ __forceinline__ bool mbar_try(uint64_t* b, uint32_t parity) {\\n  uint32_t ok;\\n  asm volatile("{\\\\n .reg .pred p;\\\\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\\\\n selp.u32 %0, 1, 0, p;\\\\n}\\\\n"\\n               : "=r"(ok) : "r"(smem_u32(b)), "r"(parity) : "memory");\\n  return ok != 0;\\n}\\n__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t parity) {\\n  while (!mbar_try(b, parity)) {}\\n}\\n__device__ __forceinline__ void mbar_arrive(uint64_t* b) {\\n  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_u32(b)) : "memory");\\n}\\n__device__ __forceinline__ void mbar_expect_tx(uint64_t* b, uint32_t bytes) {\\n  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_u32(b)), "r"(bytes) : "memory");\\n}\\n__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* b, uint64_t pol) {\\n  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"\\n               :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(b)), "l"(pol) : "memory");\\n}\\n__device__ __forceinline__ void bar_consumers() {\\n  asm volatile("bar.sync 1, %0;" :: "n"(NCT) : "memory");\\n}\\n__device__ __forceinline__ int ld_acquire(const int* p) {\\n  int v;\\n  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");\\n  return v;\\n}\\n__device__ __forceinline__ uint32_t rotr(uint32_t x, int s) { return __funnelshift_r(x, x, s); }\\n__device__ __forceinline__ uint32_t hmul2(uint32_t a, uint32_t b) {\\n  uint32_t d;\\n  asm("mul.rn.bf16x2 %0, %1, %2;" : "=r"(d) : "r"(a), "r"(b));\\n  return d;\\n}\\n__device__ __forceinline__ void mma16816(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,\\n                                         uint32_t b0, uint32_t b1) {\\n  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"\\n               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])\\n               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));\\n}\\n// 16 E2M1 codes (two words) -> 8 BF16x2 registers holding value * 2^-126 (exact).\\n__device__ __forceinline__ void dq8(uint32_t A, uint32_t B, uint32_t* o) {\\n  o[0] = A & DQ_MASK;\\n  o[1] = rotr(A, 4) & DQ_MASK;\\n  o[2] = rotr(A, 10) & DQ_MASK;\\n  o[3] = rotr(B, 2) & DQ_MASK;\\n  o[4] = rotr(B, 6) & DQ_MASK;\\n  o[5] = rotr(B, 12) & DQ_MASK;\\n  uint32_t W;asm("lop3.b32 %0,%1,0x60306030,%2,0xe2;":"=r"(W):"r"(A),"r"(B));\\n  o[6] = rotr(W, 24) & DQ_MASK;\\n  o[7] = rotr(W, 30) & DQ_MASK;\\n}\\n\\nstruct PrefetchState {\\n  volatile int index, stage, phase, pass;\\n  volatile int e, nb, m, mb, me;\\n  volatile uint64_t pol_first, pol_normal;\\n};\\n\\nstruct Route {        // shared-memory routing summary\\n  int n[NLOC];        // pairs per local expert\\n  int off[NLOC];      // first sorted position\\n  int nblk[NLOC];     // token blocks\\n  int us1[NLOC + 1];  // phase-1 unit prefix\\n  int us2[NLOC + 1];  // phase-2 unit prefix\\n  int c1, c2;         // m-chunk sizes\\n  int run[NLOC];\\n  int wc[NCW][NLOC];\\n  PrefetchState pre;\\n};\\n\\n// unit -> (expert, token block, m tile); order (e, m-chunk, block, m)\\n__device__ __forceinline__ void decode_unit(int u, const int* us, const int* nblk, int cc, int ntile,\\n                                            int& e, int& nb, int& m) {\\n  e = 0;\\n#pragma unroll\\n  for (int d = 16; d > 0; d >>= 1) {\\n    if (us[e + d] <= u) e += d;\\n  }\\n  const int rel = u - us[e];\\n  const int nbk = nblk[e];\\n  const int nfull = ntile / cc;\\n  const int span = cc * nbk;\\n  if (rel < nfull * span) {\\n    const int mc = rel / span;\\n    const int r2 = rel - mc * span;\\n    nb = r2 / cc;\\n    m = mc * cc + (r2 - nb * cc);\\n  } else {\\n    const int L = ntile - nfull * cc;\\n    const int r3 = rel - nfull * span;\\n    nb = r3 / L;\\n    m = nfull * cc + (r3 - nb * L);\\n  }\\n}\\n\\n__device__ __forceinline__ void advance_unit(int& e, int& nb, int& m, int& mb, int& me,\\n                                               const int* nblk, int cc, int ntile) {\\n  if (++m == me) {\\n    if (++nb == nblk[e]) {\\n      nb = 0;\\n      mb += cc;\\n      if (mb >= ntile) {\\n        ++e;\\n        while (e < NLOC && nblk[e] == 0) ++e;\\n        mb = 0;\\n      }\\n      me = min(ntile, mb + cc);\\n    }\\n    m = mb;\\n  }\\n}\\n\\n\\n__device__ __forceinline__ void prefetch_one(\\n    Route* rt, int total, int first, int a2, int nstage,\\n    uint8_t* stages, uint64_t* fullb, uint64_t* emptyb,\\n    const uint8_t* w13p, const uint8_t* w2p) {\\n  auto& p = rt->pre;\\n  int idx = p.index;\\n  if (idx >= total) return;\\n  int st=p.stage, ph=p.phase, pass=p.pass;\\n  int e=p.e, nb=p.nb, m=p.m, mb=p.mb, me=p.me;\\n  const int ntile=pass?NT2:NT1;\\n  const int cc=pass?rt->c2:rt->c1;\\n  const uint8_t* base=pass?w2p:w13p;\\n  const uint8_t* src=base+(static_cast<size_t>(e)*ntile+m)*TILE_BYTES;\\n  const uint64_t pol=rt->nblk[e]>1?p.pol_normal:p.pol_first;\\n  mbar_wait(&emptyb[st], ph^1u);\\n  mbar_expect_tx(&fullb[st], TILE_BYTES);\\n  bulk_g2s(stages+st*TILE_BYTES, src, TILE_BYTES, &fullb[st], pol);\\n  ++idx;\\n  if (idx < total) {\\n    if (idx == first) {\\n      pass=1;\\n      decode_unit(a2,rt->us2,rt->nblk,rt->c2,NT2,e,nb,m);\\n      mb=(m/rt->c2)*rt->c2;\\n      me=min(NT2,mb+rt->c2);\\n    } else {\\n      advance_unit(e,nb,m,mb,me,rt->nblk,cc,ntile);\\n    }\\n  }\\n  if (++st == nstage) { st=0; ph^=1; }\\n  p.index=idx; p.stage=st; p.phase=ph; p.pass=pass;\\n  p.e=e; p.nb=nb; p.m=m; p.mb=mb; p.me=me;\\n}\\n\\ntemplate<int NT,bool FIRST,bool SMALL,bool ZSS,int THR>\\n__global__ void __launch_bounds__(NTHR,2) phase(\\n const __nv_bfloat16*x,const __nv_bfloat16*pnw,const uint8_t*w13p,const uint8_t*w2p,\\n const float*gs13,const float*gs2,const int*C,const int*OFF,const int*SLOTS,\\n __nv_bfloat16*inter,float*y,float*ss,int M,int nstage){\\n constexpr int NB=8*NT,RED=NT*4;\\n extern __shared__ __align__(128) uint8_t smem[];\\n uint8_t*stages=smem;float*red=(float*)(smem+nstage*TILE_BYTES);\\n uint64_t*fullb=(uint64_t*)(red+2*NCW*32*RED),*emptyb=fullb+nstage;\\n Route*rt=(Route*)(emptyb+nstage);\\n const int tid=threadIdx.x,lane=tid&31,warp=tid>>5,G=gridDim.x,c=blockIdx.x;\\n if(tid==0){for(int z=0;z<nstage;++z){mbar_init(fullb+z,1);mbar_init(emptyb+z,NCW);}\\n  asm volatile("fence.mbarrier_init.release.cluster;":::"memory");}\\n if(warp==0){int n=C[lane];int nb=(n+NB-1)/NB;if constexpr(SMALL)if(n>THR)nb=0;\\n  rt->n[lane]=n;rt->off[lane]=OFF[lane];rt->nblk[lane]=nb;\\n  int u1=nb*NT1,u2=nb*NT2;\\n  #pragma unroll\\n  for(int d=1;d<32;d*=2){int a=__shfl_up_sync(~0u,u1,d),b=__shfl_up_sync(~0u,u2,d);if(lane>=d){u1+=a;u2+=b;}}\\n  rt->us1[lane+1]=u1;rt->us2[lane+1]=u2;\\n  if(lane==0){rt->us1[0]=rt->us2[0]=0;}\\n  int U1=__shfl_sync(~0u,u1,31),U2=__shfl_sync(~0u,u2,31);\\n  if(lane==0){rt->c1=min(NT1,max(1,(U1+G/2)/G));rt->c2=min(NT2,max(1,(U2+G/2)/G));}\\n }\\n __syncthreads();\\n  const int U1 = rt->us1[NLOC], U2 = rt->us2[NLOC];\\n  const int a1 = static_cast<int>((long long)U1 * c / G), b1 = static_cast<int>((long long)U1 * (c + 1) / G);\\n  const int a2 = static_cast<int>((long long)U2 * c / G), b2 = static_cast<int>((long long)U2 * (c + 1) / G);\\n\\n  const int first_count=FIRST?b1-a1:0,total_count=FIRST?first_count:b2-a2;\\n  if (tid == 0) {\\n    uint64_t pf,pn;\\n    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pf));\\n    asm volatile("createpolicy.fractional.L2::evict_normal.b64 %0, 1.0;" : "=l"(pn));\\n    rt->pre.pol_first=pf; rt->pre.pol_normal=pn;\\n    rt->pre.index=0; rt->pre.stage=0; rt->pre.phase=0;\\n    if (total_count) {\\n      int e,nb,m;\\n      const int pass=first_count?0:1;\\n      const int cc=pass?rt->c2:rt->c1, ntile=pass?NT2:NT1;\\n      decode_unit(pass?a2:a1,pass?rt->us2:rt->us1,rt->nblk,cc,ntile,e,nb,m);\\n      rt->pre.pass=pass; rt->pre.e=e; rt->pre.nb=nb; rt->pre.m=m;\\n      rt->pre.mb=(m/cc)*cc; rt->pre.me=min(ntile,(m/cc)*cc+cc);\\n    }\\n    for (int i=0;i<nstage;++i)\\n      prefetch_one(rt,total_count,first_count,a2,nstage,stages,fullb,emptyb,w13p,w2p);\\n  }\\n  __syncthreads();\\n\\n  const int q = lane & 3;\\n  uint32_t Bf[NT][8][8];\\n  int st = 0;\\n  uint32_t ph = 0;\\n  int buf = 0;\\n  int cur_pass = -1, cur_e = -1, cur_nb = -1, seg_tiles = 0;\\n  float fw[2];   // (1 + post_norm_w) for this thread\\\'s phase-2 epilogue row, refreshed per tile\\n\\n  for (int pass = FIRST?0:1; pass < (FIRST?1:2); ++pass) {\\n    const int ua = pass ? a2 : a1, ub = pass ? b2 : b1;\\n    const int* us = pass ? rt->us2 : rt->us1;\\n    const int cc = pass ? rt->c2 : rt->c1;\\n    const int ntile = pass ? NT2 : NT1;\\n    int e=0, nb=0, m=0, mb=0, me=0;\\n    if (ua < ub) {\\n      decode_unit(ua, us, rt->nblk, cc, ntile, e, nb, m);\\n      mb = (m / cc) * cc;\\n      me = min(ntile, mb + cc);\\n    }\\n    for (int u = ua; u < ub; ++u) {\\n      const int ne = rt->n[e], off = rt->off[e];\\n      if (pass != cur_pass || e != cur_e || nb != cur_nb) {\\n        cur_pass=pass;cur_e=e;cur_nb=nb;seg_tiles=0;\\n#pragma unroll\\n        for (int j = 0; j < NT; ++j) {\\n          const int idx = nb * NB + j * 8 + (lane >> 2);\\n          const __nv_bfloat16* row = nullptr;\\n          if (idx < ne) {\\n            const int pos = off + idx;\\n            row = pass ? inter + static_cast<size_t>(pos) * HD : x + static_cast<size_t>(SLOTS[e*M+idx] >> 3) * HD;\\n          }\\n#pragma unroll\\n          for (int g = 0; g < 8; ++g) {\\n            uint4 v0 = make_uint4(0, 0, 0, 0), v1 = v0;\\n            if (row) {\\n              const uint4* p = reinterpret_cast<const uint4*>(row + warp * 512 + g * 64 + q * 16);\\n              if (pass) { v0 = __ldcg(p); v1 = __ldcg(p + 1); }\\n              else { v0 = __ldg(p); v1 = __ldg(p + 1); }\\n            }\\n            Bf[j][g][0] = v0.x; Bf[j][g][1] = v0.y; Bf[j][g][2] = v0.z; Bf[j][g][3] = v0.w;\\n            Bf[j][g][4] = v1.x; Bf[j][g][5] = v1.y; Bf[j][g][6] = v1.z; Bf[j][g][7] = v1.w;\\n          }\\n        }\\n      }\\n      if (pass == 1 && tid < 16 * NB) fw[0] = 1.f + __bfloat162float(pnw[m * 16 + (tid & 15)]);\\n\\n      mbar_wait(&fullb[st], ph);\\n      const uint8_t* tile = stages + st * TILE_BYTES;\\n      float acc[NT][4];\\n#pragma unroll\\n      for (int j = 0; j < NT; ++j) acc[j][0] = acc[j][1] = acc[j][2] = acc[j][3] = 0.f;\\n      {\\n        const uint4* wq = reinterpret_cast<const uint4*>(tile) + warp * 8 * 32 + lane;\\n        const uint4 scv = *(reinterpret_cast<const uint4*>(tile + TILE_W_BYTES) + warp * 32 + lane);\\n        const uint32_t scw[4] = {scv.x, scv.y, scv.z, scv.w};\\n#pragma unroll\\n        for (int g = 0; g < 8; ++g) {\\n          const uint4 wv = wq[g * 32];\\n          const uint32_t sw = scw[g >> 1];\\n          const uint32_t Sr = __byte_perm(sw, 0u, (g & 1) ? 0x4242u : 0x4040u) * 16u + 0x70007000u;\\n          const uint32_t Sr8 = __byte_perm(sw, 0u, (g & 1) ? 0x4343u : 0x4141u) * 16u + 0x70007000u;\\n          uint32_t o[16];\\n          dq8(wv.x, wv.y, o);\\n          dq8(wv.z, wv.w, o + 8);\\n#pragma unroll\\n          for (int s = 0; s < 4; ++s) {\\n            const uint32_t* os = o + (s >> 1) * 8 + (s & 1) * 4;\\n            const uint32_t a0 = hmul2(os[0], Sr), a1 = hmul2(os[1], Sr8);\\n            const uint32_t a2 = hmul2(os[2], Sr), a3 = hmul2(os[3], Sr8);\\n#pragma unroll\\n            for (int j = 0; j < NT; ++j) mma16816(acc[j], a0, a1, a2, a3, Bf[j][g][2 * s], Bf[j][g][2 * s + 1]);\\n          }\\n        }\\n      }\\n      __syncwarp();\\n      if (lane == 0) mbar_arrive(&emptyb[st]);\\n      if (++st == nstage) { st = 0; ph ^= 1u; }\\n\\n      float* rp = red + ((buf * NCW + warp) * 32 + lane) * RED;\\n#pragma unroll\\n      for (int j = 0; j < NT; ++j) reinterpret_cast<float4*>(rp)[j] = make_float4(acc[j][0], acc[j][1], acc[j][2], acc[j][3]);\\n      bar_consumers();\\n      if (tid == 0) prefetch_one(rt,total_count,first_count,a2,nstage,stages,fullb,emptyb,w13p,w2p);\\n      const float* rb = red + buf * NCW * 32 * RED;\\n      if (pass == 0) {\\n        const float sc = gs13[e] * 256.f;\\n        if (tid < 8 * NB) {\\n          const int i = tid & 7, n = tid >> 3, j = n >> 3, col = n & 7;\\n          const int ls = i * 4 + (col >> 1), rg = j * 4 + (col & 1);\\n          float G1 = 0.f, U = 0.f;\\n#pragma unroll\\n          for (int w = 0; w < NCW; ++w) {\\n            G1 += rb[(w * 32 + ls) * RED + rg];\\n            U += rb[(w * 32 + ls) * RED + rg + 2];\\n          }\\n          const int idx = nb * NB + n;\\n          if (idx < ne) {\\n            G1 *= sc;\\n            U *= gs13[e+32]*256.f;\\n            G1=__bfloat162float(__float2bfloat16_rn(G1));\\n            U=__bfloat162float(__float2bfloat16_rn(U));\\n            const float h = G1 / (1.f + __expf(-G1)) * U;\\n            inter[static_cast<size_t>(off + idx) * HD + m * 8 + i] = __float2bfloat16(h);\\n          }\\n        }\\n        if constexpr (ZSS) {\\n          if (m == 0) {\\n            for (int z = tid; z < NB * 64; z += NTHR) {\\n              const int zi = nb * NB + (z >> 6);\\n              if (zi < ne) ss[static_cast<size_t>(off + zi) * 64 + (z & 63)] = 0.f;\\n            }\\n          }\\n        }\\n      } else {\\n        const float sc = gs2[e] * 256.f;\\n        if (tid < 16 * NB) {\\n          const int i = tid & 15, n = tid >> 4, j = n >> 3, col = n & 7;\\n          const int ls = (i & 7) * 4 + (col >> 1), rg = j * 4 + (col & 1) + ((i >> 3) << 1);\\n          float v = 0.f;\\n#pragma unroll\\n          for (int w = 0; w < NCW; ++w) v += rb[(w * 32 + ls) * RED + rg];\\n          v *= sc;\\n          const int idx = nb * NB + n;\\n          const bool ok = idx < ne;\\n          if (ok) y[static_cast<size_t>(off + idx) * HD + m * 16 + i] = v*fw[0];\\n          float t2 = ok ? (v * fw[0]) * (v * fw[0]) : 0.f;\\n#pragma unroll\\n          for (int d = 8; d > 0; d >>= 1) t2 += __shfl_xor_sync(~0u, t2, d);   // 16 rows of one token\\n          if (ok && i == 0) atomicAdd(ss + (size_t)(off+idx)*64+(m>>2), t2);\\n        }\\n      }\\n      buf ^= 1;\\n      ++seg_tiles;\\n      advance_unit(e, nb, m, mb, me, rt->nblk, cc, ntile);\\n    }\\n  }\\n}\\n\\nsize_t smem_bytes(int nt, int nstage, int M) {\\n  const int npairs = M * 8;\\n  return static_cast<size_t>(nstage) * TILE_BYTES + 2 * NCW * 32 * nt * 4 * sizeof(float) + 2 * nstage * 8 +\\n         sizeof(Route) + ((npairs + 15) & ~15) + 4 * npairs;\\n}\\n\\n}  // namespace\\n\\nint64_t pmz_init(){int dev,opt;BR_CUDA_CHECK(cudaGetDevice(&dev));BR_CUDA_CHECK(cudaDeviceGetAttribute(&opt,cudaDevAttrMaxSharedMemoryPerBlockOptin,dev));\\n auto set=[&]<bool F,bool Z,int T>(){cudaFuncAttributes a{};BR_CUDA_CHECK(cudaFuncGetAttributes(&a,pmz::phase<1,F,true,Z,T>));BR_CUDA_CHECK(cudaFuncSetAttribute(pmz::phase<1,F,true,Z,T>,cudaFuncAttributeMaxDynamicSharedMemorySize,opt-a.sharedSizeBytes));};\\n set.template operator()<true,true,16>();set.template operator()<false,false,16>();set.template operator()<false,false,8>();return opt;}\\nint64_t pmz_smem(int64_t stages,int64_t M){return pmz::smem_bytes(1,stages,M);}\\nint64_t pmz_resident(int64_t stages,int64_t M,int64_t first){int n;\\n if(first)BR_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,pmz::phase<1,true,true,true,16>,pmz::NTHR,pmz::smem_bytes(1,stages,M)));\\n else BR_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,pmz::phase<1,false,true,false,16>,pmz::NTHR,pmz::smem_bytes(1,stages,M)));return n;}\\n// v: x,pnw,w13p,w2p,gs13,gs2,C,OFF,SLOTS,act,y,ss. Experts with count<=thr only; W13 (thr 16) zeroes their SS rows.\\nvoid pmz_run(std::vector<int64_t>v,int64_t first,int64_t thr,int64_t M,int64_t stages,int64_t grid,int64_t stream){\\n BR_REQUIRE(v.size()==12);BR_REQUIRE(first?thr==16:(thr==16||thr==8));\\n auto launch=[&]<bool F,bool Z,int T>(){pmz::phase<1,F,true,Z,T><<<grid,pmz::NTHR,pmz::smem_bytes(1,stages,M),(cudaStream_t)stream>>>(\\n (__nv_bfloat16*)v[0],(__nv_bfloat16*)v[1],(uint8_t*)v[2],(uint8_t*)v[3],(float*)v[4],(float*)v[5],(int*)v[6],(int*)v[7],(int*)v[8],(__nv_bfloat16*)v[9],(float*)v[10],(float*)v[11],M,stages);};\\n if(first)launch.template operator()<true,true,16>();else if(thr==16)launch.template operator()<false,false,16>();else launch.template operator()<false,false,8>();\\n BR_CUDA_CHECK(cudaGetLastError());}\\n\'\nNF_CPP=\'#include <pybind11/pybind11.h>\\n#include <pybind11/stl.h>\\n#include <vector>\\n#include <cstdint>\\nint64_t br_init();std::vector<int64_t> br_info(int64_t);void br_run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t);void br_route(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t);void br_finish(std::vector<int64_t>,int64_t,int64_t,int64_t);int64_t pmz_init();int64_t pmz_smem(int64_t,int64_t);int64_t pmz_resident(int64_t,int64_t,int64_t);void pmz_run(std::vector<int64_t>,int64_t,int64_t,int64_t,int64_t,int64_t,int64_t);\'\nNF_OPTS={}\n@functools.lru_cache(None)\ndef nf_module():\n    target=\'-gencode=arch=compute_90a,code=sm_90a\' if torch.cuda.get_device_capability()==(9,0) else \'-gencode=arch=compute_120,code=sm_120\'\n    mod=load_inline(name=\'nf_m256_\'+hashlib.sha256(NF_CUDA.encode()).hexdigest()[:12],cpp_sources=NF_CPP,cuda_sources=NF_CUDA,\n        functions=[\'br_init\',\'br_info\',\'br_run\',\'br_route\',\'br_finish\',\'pmz_init\',\'pmz_smem\',\'pmz_resident\',\'pmz_run\'],\n        no_implicit_headers=True,with_pytorch_error_handling=False,extra_cuda_cflags=[\'-O3\',\'-std=c++20\',\'-lineinfo\',target],verbose=False)\n    mod.br_init();NF_OPTS[\'pmz\']=mod.pmz_init()\n    return mod\n\ndef nf_plan(data,base,native,streams):\n    x=data[\'hidden_states\'];m=x.shape[0];dev=x.device;cap=m*8+32*128;mod=nf_module()\n    c=torch.empty(32,device=dev,dtype=torch.int32);off=torch.empty(33,device=dev,dtype=torch.int32);order=torch.empty_like(c)\n    tasks=[torch.empty_like(off) for _ in range(3)];slots=torch.empty((32,m),device=dev,dtype=torch.int32)\n    ids=data[\'topk_ids\'].int().contiguous();tw=data[\'topk_weights\'].float().contiguous();pos=torch.empty_like(ids)\n    act=torch.empty((cap,4096),device=dev,dtype=torch.bfloat16);y=torch.empty((cap,4096),device=dev);ss=torch.empty((cap,64),device=dev)\n    out=torch.empty_like(x);start=data[\'config\'].get(\'local_start\',0)\n    h13=[base[\'w13\'],base[\'w13_sf\'],base[\'w13_gs\'],x,c,off,order,tasks[1],slots,base[\'pnw\'],act,y,ss]\n    h2=[base[\'w2\'],base[\'w2_sf\'],base[\'w2_gs\'],act,c,off,order,tasks[1],slots,base[\'pnw\'],act,y,ss]\n    nv=[x,base[\'pnw\'],native[\'w13p\'],native[\'w2p\'],base[\'w13_gs\'],base[\'w2_gs\'],c,off,slots,act,y,ss]\n    p13,p2,pn=([t.data_ptr() for t in v] for v in (h13,h2,nv))\n    rp=[t.data_ptr() for t in (ids,c,off,order,tasks[0],slots,pos,tasks[1],tasks[2])]\n    fp=[t.data_ptr() for t in (off,ids,tw,pos,y,ss,out)]\n    owners=(h13,h2,nv,ids,tw,pos,tasks)\n    stages=2\n    while stages>1 and mod.pmz_smem(stages,m)>NF_OPTS[\'pmz\']:stages-=1\n    sms=torch.cuda.get_device_properties(x.device).multi_processor_count\n    grids=[sms*mod.pmz_resident(stages,m,f) for f in (0,1)]\n    side,fork,join=streams;grid=1056 if m<=128 else 2112  # heavy-branch grids the H200 labs timed\n    def route():\n        _=owners;mod.br_route(rp,m,start,1,torch.cuda.current_stream().cuda_stream)\n    def body():\n        _=owners;parent=torch.cuda.current_stream()\n        fork.record(parent);side.wait_event(fork)\n        with torch.cuda.stream(side):  # heavy first: launched ahead of NT1 it takes the SMs it needs (880c3881 N_swap)\n            mod.br_run(p13,0,grid,m,side.cuda_stream);mod.br_run(p2,1,grid,m,side.cuda_stream);join.record()\n        mod.pmz_run(pn,1,16,m,stages,grids[1],parent.cuda_stream);mod.pmz_run(pn,0,16,m,stages,grids[0],parent.cuda_stream)\n        parent.wait_event(join)\n    def finish():\n        _=owners;mod.br_finish(fp,m,start,torch.cuda.current_stream().cuda_stream);return out\n    return dict(steps=(route,body,finish),out=out,owners=owners)\n\ndef nf_call(data,state):\n    for f in nf_plan(data,state[\'s5_hh_base\'],state[\'moe_bank\'],state[\'nf_streams\'])[\'steps\']:out=f()\n    return out\n'
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
    kstate = dict(state, s5_hh_base=kernel_fn.__globals__['issue_prepare'](dict(weights, post_norm_w=weights['post_expert_norm'])))
    kstate_noq = dict(kstate, vq8_rows={})
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
        if kok:variants['K'] = (ns['K'], kstate)
        if t <= 32 or routed or kok:variants['S10'] = (ns['S10'], kstate_noq) if kok else (ns['V'], noq)
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
