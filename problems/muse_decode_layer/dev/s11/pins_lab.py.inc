"""Whole-layer paired sweep of every prepare-tuned choice on S11's prepared state, one factor at a time (appended to
dev/s10/ablate_lab.py.inc, whose helpers it uses; this lab() replaces that one). The winners become PROJ_PINS and
BEST7_SETTINGS attention pins (dev/s11/patch_pin.py) instead of per-run tuning.

Per shape, around the host's plan (A; A2 = second capture):
  projection roles left 'auto' in BEST7_SETTINGS: every variant the tuners could pick at that (tokens, role) --
    custom, cublas, ('lt', i) for each cuBLASLt algo prepared, VDSK shortlist, the _MEASURED_LT recipe (its layout
    rebuilt here when prepare dropped it) -- each admitted only if its projection output is within 1% (relative L2)
    of custom on a random input, the tuner's own guard; arm name '<qkv|o|pool|uslc|unp>=<choice>'
  attention, where BEST7_SETTINGS says 'auto': ('layer', None), ('attn_main', None) and the tuner's four fastest
    public-draw trials; arm name 'att=<choice>'
Seeds PIN_SEEDS x PIN_ROUNDS rounds, rotating order, lab_cold_l2 before each replay.
Lines (compact for the 9000-character lab log): S <shape> A=<host us> se=<median per-arm se> <arm>:<r> ... sorted by r
(r < 1: the arm beats the host; '!' marks a capture failure, non-finite output or route-ID change); PINA <shape>
host=<choice> best=<choice> r=<shape ratio>; PINP <tokens> <role> host=<choice> best=<choice> r=<46/16 LOAD ratio>.
"""
PIN_SEEDS = AB_SEEDS[:12]
PIN_ROUNDS = 6
_PIN_SHORT = lambda c: c if isinstance(c, str) else '/'.join(map(str, c)).replace(' ', '').replace('short_lt', 'slt')
_PIN_ROLE = {'qkv_gate': 'qkv', 'o': 'o', 'pool_router': 'pool', 'unpooler_slice': 'uslc', 'unpooler': 'unp'}


def pin_measured_lt(state, device):
    """Layouts for every _MEASURED_LT recipe (prepare keeps only the chosen ones)."""
    g = kernel_fn.__globals__
    if 'measured_lt' not in state:
        import hashlib as _h
        major, minor = torch.cuda.get_device_capability(device);arch = '90a' if major == 9 else f'{major}{minor}'
        tag = _h.sha256(g['PG_RAW'].encode()).hexdigest()[:12]
        path = load_inline(name=f'muse_decode_measured_lt_{tag}_{arch}', cpp_sources='', cuda_sources=g['PG_RAW'],
                           functions=None, is_python_module=False, no_implicit_headers=True,
                           extra_cuda_cflags=['-O3', '-std=c++20', f'-gencode=arch=compute_{arch},code=sm_{arch}'],
                           extra_ldflags=['-lcublasLt'], verbose=False)
        state['measured_lt'] = {'mod': g['_PgNative'](path), 'layouts': {}, 'workspace': torch.empty(4 << 30, device=device, dtype=torch.uint8)}
    bank = state['measured_lt'];out = {}
    for (t, name), (nn, ordinal) in g['_MEASURED_LT'].items():
        key = (t, name, nn)
        if key not in bank['layouts']:
            w = state[name];n, k = w.shape
            codes, _ = bank['mod'].options(t, n, k, nn, name == 'pool_router', bank['workspace'].numel())
            if ordinal >= len(codes):continue
            bank['layouts'][key] = (w.T.contiguous() if nn else w, codes)
        out[(t, name)] = ('short_lt', nn, ordinal)
    return out


def pin_alternatives(t, role, state, short_lt):
    g = kernel_fn.__globals__
    alts = ['custom', 'cublas']
    if role != 'pool_router':alts += [('lt', i) for i in range(int(len(state.get('lt_algos', {}).get((role, t), ()))))]
    alts += [('vdsk', cfg, s) for cfg, s in g['VDSK_SHORTLIST'].get((t, role), ())]
    if (t, role) in short_lt:alts.append(short_lt[(t, role)])
    w = state[role];dtype = torch.float32 if role == 'pool_router' else torch.bfloat16
    gen = torch.Generator(device=w.device).manual_seed(1009 + t)
    x = torch.randn((t, w.shape[1]), device=w.device, dtype=torch.bfloat16, generator=gen)
    want = g['_materialize'](g['_project_variant'](x, w, role, state, 'custom', dtype)).float()
    ok, rejected = [], []
    for c in alts:
        try:
            y = g['_materialize'](g['_project_variant'](x, w, role, state, c, dtype)).float()
            rel = float((y - want).norm() / want.norm().clamp_min(1e-12))
            (ok if bool(torch.isfinite(y).all()) and rel <= .01 else rejected).append(c if rel <= .01 else (c, round(rel, 4)))
        except Exception as exc:
            rejected.append((c, type(exc).__name__))
    return ok, rejected


def pin_attention_alternatives(key, state):
    b, q, w = key
    rows = [r for r in state.get('attention_public_rows', []) if (r['batch'], r['q_len'], bool(r['window'])) == (b, q, bool(w))]
    trials = sorted(rows[0]['trials'], key=lambda r: r['mean_us']) if rows else []
    alts = [('layer', None)] + ([('attn_main', None)] if q == 1 else [])
    for tr in trials:
        c = tuple(tr['choice']) if isinstance(tr['choice'], list) else tr['choice']
        if c not in alts and len(alts) < 6:alts.append(c)
    return alts


def lab(deadline, weights, state):
    started = time.monotonic();device = weights['qkv_gate'].device
    lines = [f'META host S11 state; one-factor sweep of tuned choices; seeds {len(PIN_SEEDS)} from {PIN_SEEDS[0]} (+7919); '
             f'{PIN_ROUNDS} rounds rotating order, lab_cold_l2 before each replay; r = mean per-seed time ratio vs A (<1 = arm beats the host)']
    try:short_lt = pin_measured_lt(state, device)
    except Exception as exc:
        short_lt = {};lines.append(f'SETUP measured_lt ERROR {type(exc).__name__}:{str(exc)[:80]}')
    lines.append(f'SETUP {time.monotonic() - started:.1f}s left={deadline - time.monotonic():.0f}s')
    ns = ab_namespace({})
    results = {};incomplete = [];host_proj = {};arm_choice = {};alt_cache = {};pina = []
    order = [k for k in AB_CASES if k[0] * k[1] <= 32] + [k for k in AB_CASES if k[0] * k[1] > 32]
    for key in order:
        if time.monotonic() > deadline - 8:incomplete.append(ab_label(key));continue
        b, q, w = key;t = b * q;plan = state['best7_plans'][key]
        mk = lambda seed: _ap_generate_input(batch=b, ctx=80000, layer='sliding' if w else 'full', q_len=q,
                                             spread='agentic' if b > 1 else 'fixed', dist='soft', page=1, seed=seed, device=device)
        data = mk(PIN_SEEDS[0])
        variants = {'A': state, 'A2': state}
        roles = ('qkv_gate', 'o', 'pool_router', 'unpooler_slice' if t <= 8 else 'unpooler')
        for role in roles:
            if plan['projection'].get(role, 'auto') != 'auto':continue
            cur = state['proj_plan'].get((t, role), 'custom');host_proj[(t, role)] = cur
            if (t, role) not in alt_cache:
                alt_cache[(t, role)] = pin_alternatives(t, role, state, short_lt)
                if alt_cache[(t, role)][1]:lines.append(f'REJECT t={t} {role} {alt_cache[(t, role)][1]}')
            for c in alt_cache[(t, role)][0]:
                if c == cur:continue
                name = f'{_PIN_ROLE[role]}={_PIN_SHORT(c)}';arm_choice[name] = (role, c)
                variants[name] = dict(state, proj_plan={**state['proj_plan'], (t, role): c})
        if plan['attention'] == 'auto':
            k3 = (b, q, bool(w));cur = state['attention_plan'].get(k3, ('layer', None))
            for c in pin_attention_alternatives(key, state):
                if c == cur:continue
                name = f'att={_PIN_SHORT(c)}';arm_choice[name] = ('attention', c)
                variants[name] = dict(state, attention_plan={**state['attention_plan'], k3: c})
        graphs = {};gates = {}
        for n, st in variants.items():
            try:graphs[n] = ab_graph(lambda st=st: ns['kernel_fn'](data, st))
            except Exception as exc:gates[n] = 'capture:' + type(exc).__name__ + ':' + str(exc)[:60]
        per_seed = []
        for seed in PIN_SEEDS:
            if time.monotonic() > deadline - 5:incomplete.append(f'{ab_label(key)}@{seed}');break
            if seed != PIN_SEEDS[0]:ab_copy(data, mk(seed))
            for g, _ in graphs.values():g.replay()
            torch.cuda.synchronize()
            for n, (g, out) in graphs.items():
                if n != 'A' and (r := ab_gate(out, graphs['A'][1])):gates.setdefault(n, r)
            times = ab_time(graphs, PIN_ROUNDS)
            per_seed.append({n: statistics.mean(x) for n, x in times.items()})
        if per_seed:results[key] = (ab_summary(per_seed), gates)
        summ = results.get(key, ({}, {}))[0]
        if summ:
            bad = lambda n: '!' if n in gates and not gates[n].startswith('L2=') else ''
            a_us = next(iter(summ.values()))[3];ses = sorted(v[1] for v in summ.values())
            lines.append(f'S {ab_label(key)} A={a_us:.1f} se={ses[len(ses) // 2]:.4f} n={len(per_seed)} '
                         + ' '.join(f'{n}:{v[0]:.4f}{bad(n)}' for n, v in sorted(summ.items(), key=lambda z: z[1][0])))
        fails = [n for n in gates if n not in summ]
        if fails:lines.append(f'F {ab_label(key)} ' + ' '.join(f'{n}:{gates[n][:40]}' for n in fails))
        if plan['attention'] == 'auto' and key in results:
            arms = {n: v[0] for n, v in results[key][0].items() if n.startswith('att=')}
            best = min(arms.items(), key=lambda z: z[1], default=None)
            cur = state['attention_plan'].get((b, q, bool(w)), ('layer', None))
            if best is None or best[1] >= 1.:pina.append(f'PINA {ab_label(key)} host={_PIN_SHORT(cur)} best={_PIN_SHORT(cur)} r=1.0000')
            else:pina.append(f'PINA {ab_label(key)} host={_PIN_SHORT(cur)} best={best[0][4:]} r={best[1]:.4f}')
        graphs = variants = None;gc.collect();torch.cuda.empty_cache()
    loads = {}
    for (b, q), load in AB_LOADS.items():
        ks = [(b, q, 2048), (b, q, 0)]
        if not all(k in results for k in ks):continue
        names = set(results[ks[0]][0]) & set(results[ks[1]][0])
        for n in sorted(names):
            num = sum(wt * results[k][0][n][3] * results[k][0][n][0] for k, wt in zip(ks, (46, 16)))
            den = sum(wt * results[k][0][n][3] for k, wt in zip(ks, (46, 16)))
            loads[(b * q, n)] = num / den
    lines += pina
    for (t, role), cur in sorted(host_proj.items(), key=lambda z: (z[0][0], z[0][1])):
        arms = {n: r for (tt, n), r in loads.items() if tt == t and n in arm_choice and arm_choice[n][0] == role}
        best = min(arms.items(), key=lambda z: z[1], default=None)
        if best is None or best[1] >= 1.:lines.append(f'PINP {t} {role} host={_PIN_SHORT(cur)} best={_PIN_SHORT(cur)} r=1.0000')
        else:lines.append(f'PINP {t} {role} host={_PIN_SHORT(cur)} best={_PIN_SHORT(arm_choice[best[0]][1])} r={best[1]:.4f}')
    if incomplete:lines.append('INCOMPLETE ' + ' '.join(incomplete))
    lines.append(f'END shapes={len(results)}/12 elapsed={time.monotonic() - started:.1f}s left={deadline - time.monotonic():.1f}s')
    return lines
