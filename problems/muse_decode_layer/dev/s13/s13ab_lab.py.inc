"""S13 leave-one-out. r=arm/A; >1 means S13 wins. All seeds required for promotion."""
import linecache
import threading
import ast
import hashlib

S13_VARIANTS = __S13_VARIANTS__


def s13_restore(spec):
    lines = S13_HOST_SOURCE.splitlines(keepends=True)
    for a, b, text in reversed(spec['edits']):lines[a:b] = [text]
    source = ''.join(lines)
    assert hashlib.sha256(source.encode()).hexdigest() == spec['sha256']
    return source


def s13_assignments(source):
    found = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):found[target.id] = ast.dump(node)
    return found


def s13_namespace(name, spec, state):
    fname = '<s13-arm-' + name + '-' + spec['sha256'][:12] + '>'
    source = spec['source']
    linecache.cache[fname] = (len(source), None, source.splitlines(True), fname)
    ns = dict(__name__=__name__, __file__=fname)
    exec(compile(source, fname, 'exec'), ns)
    # prepare selects portable attention locally and initializes native caches. Carry over unchanged
    # assignments, while changed CUDA/layout constants and every function keep the arm's own globals.
    g = kernel_fn.__globals__
    host_assign = s13_assignments(S13_HOST_SOURCE)
    for key, code in s13_assignments(source).items():
        if key not in ('_GEMV_CHOICES', 'NF_OPTS', '_CD_MOD') and host_assign.get(key) == code and key in g and not callable(g[key]):
            ns[key] = g[key]
    # Seed the source's actual projection choices, independently of the S13 host's mutable choices.
    ns['_seed_v9_gemv_choices'](state)
    return ns


def lab(deadline, weights, state):
    started = time.monotonic()
    device = weights['qkv_gate'].device
    generator = _ap_generate_input
    if torch.cuda.get_device_capability(device) != (9, 0):
        # local_eval disables FA3 scheduler metadata on its reference inputs. Apply that same portable
        # fixture behavior here; the judge uses the original scheduler on H200.
        generator = ab_namespace({'_ap_fa3_scheduler_metadata': lambda *a, **kw: None})['_ap_generate_input']
    lines = ['META S13 leave-one-out; 16 seeds (+7919), 8 rotating rounds, cold L2 before every replay; r=arm/A (>1 S13 wins)',
             'META arms=' + ','.join(S13_VARIANTS) + '; A2=A:A; S12=all new pieces removed; unchanged loads are controls']
    ns = {'A': ab_namespace({})}
    states = {'A': state}
    compiled, errors, threads = {}, {}, {}
    def native(name, v):
        try:
            v['nf_module']()
            if 'cd_module' in v:v['cd_module']()
            compiled[name] = True
        except Exception as exc:errors[name] = f'{type(exc).__name__}:{str(exc)[:100]}'
    for name, spec in S13_VARIANTS.items():
        spec['source'] = s13_restore(spec)
        ns[name] = s13_namespace(name, spec, state)
        th = threading.Thread(target=native, args=(name, ns[name]), daemon=True)
        threads[name] = th
        th.start()
    # A reverted packed layout needs its own bank built from the original codes, not S13's wide layout.
    # Compare preparation code, so finalizer-only reversions reuse the exact host bank.
    def bank_prepare_code(source):
        tree = ast.parse(source)
        return ast.dump(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_s4_prepare_banks'))
    host_bank_code = bank_prepare_code(S13_HOST_SOURCE)
    bank_states = {host_bank_code: state}
    for name, spec in S13_VARIANTS.items():
        code = bank_prepare_code(spec['source'])
        if code not in bank_states:
            st = dict(state)
            ns[name]['_s4_prepare_banks'](st, weights)
            # The changed representation belongs only to s5_hh_base. Share the unchanged small-M bank,
            # including its exact counter buffers, so control loads do not compare unrelated allocations.
            st['s4_nt1_bank'] = state['s4_nt1_bank']
            bank_states[code] = st
        states[name] = bank_states[code]
        # A reverted coded implementation may use a different packed-row layout and ABI.
        # Rebuild its lossless bank from the identical original BF16 state, outside timing.
        if name in ('cpall', 'S12'):
            threads[name].join(max(0., deadline - 30 - time.monotonic()))
            if not compiled.get(name):raise RuntimeError(name + ': ' + errors.get(name, 'compile pending'))
            states[name] = dict(states[name])
            ns[name]['_cd_prepare'](states[name])
    results, seed_times, incomplete = {}, {}, []
    for key in AB_CASES:
        if time.monotonic() > deadline - 6:
            incomplete.append(ab_label(key));continue
        b, q, w = key;t = b * q
        mk = lambda seed: generator(batch=b, ctx=80000, layer='sliding' if w else 'full', q_len=q,
            spread='agentic' if b > 1 else 'fixed', dist='soft', page=1, seed=seed, device=device)
        data = mk(AB_SEEDS[0])
        variants = {'A': (ns['A'], state), 'A2': (ns['A'], state)}
        for name, spec in S13_VARIANTS.items():
            if name == 'S12' or t in spec['active_tokens']:
                if t in (1, 128, 256):
                    threads[name].join(max(0., deadline - 30 - time.monotonic()))
                    if not compiled.get(name):
                        incomplete.append(name + ':' + ab_label(key) + ':' + errors.get(name, 'compile pending'));continue
                variants[name] = (ns[name], states[name])
        graphs, gates = {}, {}
        for name, (v, st) in variants.items():
            try:graphs[name] = ab_graph(lambda v=v, st=st: v['kernel_fn'](data, st))
            except Exception as exc:gates[name] = 'capture:' + type(exc).__name__ + ':' + str(exc)[:80]
        if 'A' not in graphs:
            lines.append('ERROR ' + ab_label(key) + ' A capture failed ' + str(gates));continue
        per_seed = []
        for seed in AB_SEEDS:
            if time.monotonic() > deadline - 4:
                incomplete.append(f'{ab_label(key)}@{seed}');break
            if seed != AB_SEEDS[0]:ab_copy(data, mk(seed))
            for g, _ in graphs.values():g.replay()
            torch.cuda.synchronize()
            for name, (_, output) in graphs.items():
                if name != 'A' and (reason := ab_gate(output, graphs['A'][1])):gates.setdefault(name, reason)
            times = ab_time(graphs, AB_ROUNDS)
            per_seed.append({name: statistics.mean(values) for name, values in times.items()})
        if per_seed:
            results[key] = (ab_summary(per_seed), gates)
            seed_times[key] = per_seed
        for name, (ratio, se, delta, a_us, count) in sorted(results.get(key, ({}, {}))[0].items()):
            lines.append(f'L {ab_label(key)} {name} r={ratio:.4f} se={se:.4f} d={delta:+.2f} A={a_us:.1f} n={count}'
                         + (f' FAIL {gates[name]}' if name in gates else ''))
        for name, reason in gates.items():
            if name not in results.get(key, ({}, {}))[0]:lines.append(f'L {ab_label(key)} {name} FAIL {reason}')
        graphs = variants = None
        gc.collect();torch.cuda.empty_cache()
    for (b, q), load in AB_LOADS.items():
        keys = [(b, q, 2048), (b, q, 0)]
        if not all(k in results for k in keys):continue
        aa = []
        for left, right in zip(*(seed_times[k] for k in keys)):
            aa.append((46 * left['A2'] + 16 * right['A2']) / (46 * left['A'] + 16 * right['A']))
        aa_se = statistics.stdev(aa) / math.sqrt(len(aa)) if len(aa) > 1 else float('nan')
        lines.append(f'AA {load} mean={statistics.mean(aa):.6f} lo={min(aa):.6f} hi={max(aa):.6f} '
                     f'se={aa_se:.6f} n={len(aa)}')
        for name in ['A2'] + list(S13_VARIANTS):
            if all(name not in results[k][0] for k in keys):continue
            rows = [results[k][0].get(name, (1., 0., 0., results[k][0]['A2'][3], 0)) for k in keys]
            num = sum(wt * row[3] * row[0] for row, wt in zip(rows, (46, 16)))
            den = sum(wt * row[3] for row, wt in zip(rows, (46, 16)))
            lines.append(f'LOAD {load} {name} r={num / den:.6f}')
            if all(name in seed_times[k][0] for k in keys):
                ratios=[(46*left[name]+16*right[name])/(46*left['A']+16*right['A'])
                        for left,right in zip(*(seed_times[k] for k in keys))]
                normalized=[r/a for r,a in zip(ratios,aa)]
                lines.append(f'PAIR {load} {name} mean={statistics.mean(ratios):.6f} '
                             f'se={statistics.stdev(ratios)/math.sqrt(len(ratios)):.6f} '
                             f'norm={statistics.mean(normalized):.6f} '
                             f'normse={statistics.stdev(normalized)/math.sqrt(len(normalized)):.6f} n={len(ratios)}')
    if incomplete:lines.append('INCOMPLETE ' + ' '.join(incomplete))
    lines.append(f'END shapes={len(results)}/12 elapsed={time.monotonic() - started:.1f}s left={deadline - time.monotonic():.1f}s')
    return lines
