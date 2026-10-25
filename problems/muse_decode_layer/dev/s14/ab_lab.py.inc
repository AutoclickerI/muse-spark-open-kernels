"""S14 leave-one-out. r=arm/A; >1 means S14 wins. All seeds required for promotion."""
import linecache
import threading
import ast
import hashlib

S14_VARIANTS = __S14_VARIANTS__


def s14_restore(spec):
    lines = S14_HOST_SOURCE.splitlines(keepends=True)
    for a, b, text in reversed(spec['edits']):lines[a:b] = [text]
    source = ''.join(lines)
    assert hashlib.sha256(source.encode()).hexdigest() == spec['sha256']
    return source


def s14_assignments(source):
    found = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):found[target.id] = ast.dump(node)
    return found


def s14_namespace(name, spec, state):
    fname = '<s14-arm-' + name + '-' + spec['sha256'][:12] + '>'
    source = spec['source']
    linecache.cache[fname] = (len(source), None, source.splitlines(True), fname)
    ns = dict(__name__=__name__, __file__=fname)
    exec(compile(source, fname, 'exec'), ns)
    # prepare selects portable attention locally and initializes native caches. Carry over unchanged
    # assignments, while changed CUDA/layout constants and every function keep the arm's own globals.
    g = kernel_fn.__globals__
    host_assign = s14_assignments(S14_HOST_SOURCE)
    for key, code in s14_assignments(source).items():
        if key not in ('_GEMV_CHOICES', 'NF_OPTS', '_CD_MOD', '_DCP_MOD') and host_assign.get(key) == code and key in g and not callable(g[key]):
            ns[key] = g[key]
    # All these experiments preserve the balanced GEMV / split-QKV choices. Reseeding only V9
    # drops _gate_split_prepare's QKV3 entry and makes unchanged decode arms slower.
    for key in ('GB_ROWS', 'GATE_SPLIT_QKV3', 'GATE_SPLIT_ROWS'):
        assert ns[key] == g[key], (name, key)
    ns['_GEMV_CHOICES'] = dict(g['_GEMV_CHOICES'])
    assert ns['_GEMV_CHOICES'] == g['_GEMV_CHOICES']
    return ns


def lab(deadline, weights, state):
    started = time.monotonic()
    device = weights['qkv_gate'].device
    generator = _ap_generate_input
    if torch.cuda.get_device_capability(device) != (9, 0):
        # local_eval disables FA3 scheduler metadata on its reference inputs. Apply that same portable
        # fixture behavior here; the judge uses the original scheduler on H200.
        generator = ab_namespace({'_ap_fa3_scheduler_metadata': lambda *a, **kw: None})['_ap_generate_input']
    lines = ['META S14 leave-one-out; 16 seeds (+7919), 32 rounds at decode1 / 8 elsewhere, cold L2 before every replay; r=arm/A (>1 S14 wins)',
             'META arms=' + ','.join(S14_VARIANTS) + '; A2=A:A; S13=all new pieces removed; unchanged loads are controls']
    ns = {'A': ab_namespace({})}
    states = {'A': state}
    compiled, errors, threads = {}, {}, {}
    def native(name, v):
        try:
            v['nf_module']()
            if 'nf128_module' in v:v['nf128_module']()
            if 'cd_module' in v:v['cd_module']()
            if 'dcp_module' in v:v['dcp_module']()
            compiled[name] = True
        except Exception as exc:errors[name] = f'{type(exc).__name__}:{str(exc)[:100]}'
    for name, spec in S14_VARIANTS.items():
        spec['source'] = s14_restore(spec)
        ns[name] = s14_namespace(name, spec, state)
        th = threading.Thread(target=native, args=(name, ns[name]), daemon=True)
        threads[name] = th
        th.start()
    # A reverted packed layout needs its own bank built from the original codes, not S14's wide layout.
    # Compare preparation code, so finalizer-only reversions reuse the exact host bank.
    def bank_prepare_code(source):
        tree = ast.parse(source)
        return ast.dump(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_s4_prepare_banks'))
    host_bank_code = bank_prepare_code(S14_HOST_SOURCE)
    bank_states = {host_bank_code: state}
    for name, spec in S14_VARIANTS.items():
        code = bank_prepare_code(spec['source'])
        if code not in bank_states:
            st = dict(state)
            ns[name]['_s4_prepare_banks'](st, weights)
            # The changed representation belongs only to s5_hh_base. Share the unchanged small-M bank,
            # including its exact counter buffers, so control loads do not compare unrelated allocations.
            st['s4_nt1_bank'] = state['s4_nt1_bank']
            bank_states[code] = st
        states[name] = bank_states[code]
        # Shape plans are prepared state: restore stream-O when a direct-O piece is removed.
        # Other plan settings are identical and keep the exact admitted host gates/geometry.
        st=dict(states[name]);st['best7_plans']={k:dict(v,projection=dict(v['projection'])) for k,v in state['best7_plans'].items()}
        st['best7_stream_ops']=dict(state['best7_stream_ops'])
        for key,setting in ns[name]['BEST7_SETTINGS'].items():
            assert setting[:3]==ns['A']['BEST7_SETTINGS'][key][:3],(name,key)
            projection=setting[3] if isinstance(setting[3],dict) else {}
            st['best7_plans'][key]['projection']={k:v for k,v in projection.items() if k!='l2_prefetch'}
            if key=='other':continue
            tokens=key[0]*key[1]
            for role,choice in projection.items():
                if isinstance(choice,tuple) and choice[0]=='stream':
                    opkey=(tokens,role,choice[1],choice[2])
                    if opkey not in st['best7_stream_ops']:
                        st['best7_stream_ops'][opkey],_=ns[name]['_best7_stream_op'](tokens,role,choice,st,device)
        plan=next(n for n in ast.parse(spec['source']).body if isinstance(n,ast.FunctionDef) and n.name=='nf_plan')
        unpack=next(n.targets[0] for n in ast.walk(plan) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Tuple)
                    and getattr(n.targets[0].elts[0],'id','')=='side')
        st['nf_streams']=state['nf_streams'][:len(unpack.elts)]
        assert len(st['nf_streams'])==len(unpack.elts)
        states[name]=st
    results, seed_times, incomplete = {}, {}, []
    for key in AB_CASES:
        if time.monotonic() > deadline - 6:
            incomplete.append(ab_label(key));continue
        b, q, w = key;t = b * q
        mk = lambda seed: generator(batch=b, ctx=80000, layer='sliding' if w else 'full', q_len=q,
            spread='agentic' if b > 1 else 'fixed', dist='soft', page=1, seed=seed, device=device)
        data = mk(AB_SEEDS[0])
        variants = {'A': (ns['A'], state), 'A2': (ns['A'], state)}
        for name, spec in S14_VARIANTS.items():
            if True:  # every off arm also measures its untouched loads
                if True:
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
            times = ab_time(graphs, 32 if t == 1 else AB_ROUNDS)
            per_seed.append({name: statistics.mean(values) for name, values in times.items()})
        if per_seed:
            results[key] = (ab_summary(per_seed), gates)
            seed_times[key] = per_seed
        for name, (ratio, se, delta, a_us, count) in sorted(results.get(key, ({}, {}))[0].items()):
            if name not in ('A2','S13') and t not in S14_VARIANTS[name]['active_tokens'] and name not in gates:continue
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
        for name in ['A2'] + list(S14_VARIANTS):
            if all(name not in results[k][0] for k in keys):continue
            if not all(name in seed_times[k][0] for k in keys):continue
            ratios=[(46*left[name]+16*right[name])/(46*left['A']+16*right['A'])
                    for left,right in zip(*(seed_times[k] for k in keys))]
            lines.append(f'LOAD {load} {name} r={statistics.mean(ratios):.6f}')
            t=b*q
            if name=='S13' or (name!='A2' and t in S14_VARIANTS[name]['active_tokens']):
                normalized=[r/a for r,a in zip(ratios,aa)]
                lines.append(f'PAIR {load} {name} mean={statistics.mean(ratios):.6f} '
                             f'se={statistics.stdev(ratios)/math.sqrt(len(ratios)):.6f} '
                             f'norm={statistics.mean(normalized):.6f} '
                             f'normse={statistics.stdev(normalized)/math.sqrt(len(normalized)):.6f} n={len(ratios)}')
    if incomplete:lines.append('INCOMPLETE ' + ' '.join(incomplete))
    lines.append(f'END shapes={len(results)}/12 elapsed={time.monotonic() - started:.1f}s left={deadline - time.monotonic():.1f}s')
    return lines
