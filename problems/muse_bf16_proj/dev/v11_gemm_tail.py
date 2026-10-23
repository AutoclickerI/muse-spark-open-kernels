"""V11 integration: exact incumbent producer/consumer remains the group control."""
_pg_v11_prepare = prepare


def _pg_v11_dense(x, state, role):
    if role == 'router':
        pair = torch.mm(x, state['router_cat'].T, out_dtype=torch.float32)
        return pair[:, :256] + pair[:, 256:]
    return torch.mm(x, state['qkv' if role == 'qkv_gate' else role].T)


def _pg_layer_linear(x, state, role):
    plan = state.get('pg')
    return plan(x, role) if plan is not None else _pg_v11_dense(x, state, role)


def _pg_layer_group(h, state, local_start, choice=None):
    m = h.shape[0]
    choice = choice or state.get('pg_groups', {}).get(m, ('v11',))
    if choice[0] in ('v11', 'v11_router'):
        logits = torch.mm(h, state['router_cat'].T, out_dtype=torch.float32)
        route = _routing_kernel
        pooled = None
    elif choice[0] in ('separate', 'v11_pooler'):
        logits = state['pg'](h, 'router')
        route = _pg_routing_256
        pooled = None
    elif choice[0] == 'mixed':
        pooled, logits = state['pg'](h, 'pool_router')
        route = _pg_routing_256
    else:
        raise ValueError('unknown V11 projection group choice')
    ids = torch.empty((m, 8), device=h.device, dtype=torch.int32)
    topk_w = torch.empty((m, 8), device=h.device, dtype=torch.float32)
    slot_e, slot_pos = torch.empty_like(ids), torch.empty_like(ids)
    counts = torch.zeros(32, device=h.device, dtype=torch.int32)
    route[(m,)](logits, state['router_bias'], local_start, ids, topk_w,
                 slot_e, slot_pos, counts, num_warps=2)
    # Preserve V11's router -> routing -> pooler order in the incumbent.
    if pooled is None:
        pooled = (torch.mm(h, state['pooler'].T) if choice[0] in ('v11', 'v11_pooler')
                  else state['pg'](h, 'pooler'))
    return pooled, ids, topk_w, slot_e, slot_pos, counts, logits


def _pg_group_logits(output):
    raw = output[-1]
    return raw if raw.shape[1] == 256 else raw[:, :256] + raw[:, 256:]


def _pg_group_agrees(got, reference, state):
    if not _pg_equal(got[0], reference[0]): return False
    lg, expected = _pg_group_logits(got), _pg_group_logits(reference)
    if not _pg_equal(lg, expected): return False
    # Match the layer's allowed near-tie routing differences, rather than
    # requiring identical ids or nondeterministic atomic slot positions.
    score = torch.sigmoid(expected * .1767766952966369) + state['router_bias']
    values, indices = torch.topk(score, 9, dim=1)
    chosen = got[1].long()
    if not bool((score.gather(1, chosen) >= values[:, 7:8] - .05).all()): return False
    required = values[:, :8] > values[:, 8:9] + .04
    included = (indices[:, :8, None] == chosen[:, None, :]).any(2)
    return bool((included | ~required).all()) and bool(torch.isfinite(got[2]).all())


def _pg_group_pair(base, trial, h, banks, flush, state, repeats, deadline):
    samples = [[], []]
    for rep in range(repeats):
        if time.monotonic() >= deadline: raise TimeoutError('group trial budget exhausted')
        h.copy_(banks[(rep // 2) % len(banks)])
        for index in ((0, 1) if rep % 2 == 0 else (1, 0)):
            flush.add_(1)
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record(); (base if index == 0 else trial).graph.replay(); b.record(); b.synchronize()
            samples[index].append(a.elapsed_time(b) * 1000)
        if not _pg_group_agrees(trial.out, base.out, state):
            raise ValueError('pooler/router/consumer admission failed')
    ratios = [((samples[1][i] / samples[0][i]) * (samples[1][i+1] / samples[0][i+1]))**.5
              for i in range(0, repeats, 2)]
    return statistics.median(ratios), statistics.median(samples[0]), statistics.median(samples[1])


def _pg_tune_v11_groups(state, deadline, lines):
    if state.get('pg') is None: return
    device = state['pooler'].device
    flush = torch.zeros(128 << 20, device=device, dtype=torch.uint8)
    local_start = torch.zeros((), device=device, dtype=torch.int32)
    for m in (16384, 3012, 753):
        if time.monotonic() + 1 >= deadline:
            lines.append(f'PGC_GROUP m={m} choice=v11 budget'); continue
        try:
            h = torch.randn((m, 8192), device=device, dtype=torch.bfloat16)
            banks = [torch.randn_like(h) for _ in range(2)]
            probe = torch.zeros_like(h); probe[:, 127] = .5; banks.append(probe)
            best = (1., ('v11',), None)
            with _PgGraph(lambda: _pg_layer_group(h, state, local_start, ('v11',))) as base:
                if not _pg_group_agrees(base.out, base.out, state):
                    raise RuntimeError('V11 incumbent group must hold')
                for choice in (('v11_router',), ('v11_pooler',), ('separate',), ('mixed',)):
                    if time.monotonic() + 1 >= deadline: break
                    try:
                        with _PgGraph(lambda c=choice: _pg_layer_group(h, state, local_start, c)) as graph:
                            for fresh in banks:
                                h.copy_(fresh); base.graph.replay(); graph.graph.replay()
                                if not _pg_group_agrees(graph.out, base.out, state): raise ValueError('three-bank group guard')
                            ratio, old, new = _pg_group_pair(base, graph, h, banks, flush, state, 20, deadline)
                            lines.append(f'PGC_TRIAL m={m} choice={choice[0]} old_us={old:.3f} new_us={new:.3f} ratio={ratio:.6f}')
                            if ratio < best[0]: best = (ratio, choice, (old, new))
                    except Exception as exc:
                        lines.append(f'PGC_REJECT group m={m} choice={choice[0]} {type(exc).__name__}: {str(exc)[:110]}')
            state['pg_groups'][m] = best[1]
            lines.append(f'PGC_GROUP m={m} choice={best[1][0]} ratio={best[0]:.6f} us={best[2]}')
        except Exception as exc:
            # A diagnostic failure cannot displace the known V11 group.
            state['pg_groups'][m] = ('v11',)
            lines.append(f'PGC_REJECT group m={m} retained=v11 {type(exc).__name__}: {str(exc)[:110]}')


def _pg_profile_v11(state, deadline, lines):
    # Confirm the inferred profile-name -> projection mapping on H200 itself.
    # Profile only our own synthetic untimed GEMMs; no harness state is read.
    if torch.cuda.get_device_capability()[0] != 9: return
    for role in ('qkv_gate', 'o', 'pooler', 'unpooler', 'router'):
        if time.monotonic() + 2 >= deadline: break
        try:
            w = state['router_cat' if role == 'router' else 'qkv' if role == 'qkv_gate' else role]
            x = torch.randn((16384, w.shape[1]), device=w.device, dtype=torch.bfloat16)
            def call():
                return torch.mm(x, w.T, **({'out_dtype':torch.float32} if role == 'router' else {}))
            call(); torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                call()
            events = [e for e in prof.events() if str(e.device_type).endswith('CUDA')]
            events.sort(key=lambda e: e.self_device_time_total, reverse=True)
            for event in events[:2]:
                lines.append(f'PGC_KERNEL {role} m=16384 us={event.self_device_time_total:.3f} name={event.name[:190]}')
        except Exception as exc:
            lines.append(f'PGC_REJECT profiler {role}: {type(exc).__name__}: {str(exc)[:100]}')


def prepare(weights):
    started = time.monotonic()
    state = _pg_v11_prepare(weights)
    state['pg'], state['pg_groups'] = None, {}
    lines = []
    try: _pg_profile_v11(state, min(started+210, time.monotonic()+12), lines)
    except Exception as exc:
        lines.append(f'PGC_REJECT profiler {type(exc).__name__}: {str(exc)[:130]}')
    try:
        # The layer's own proven matmuls are the dense controls. Group selection
        # additionally compares its exact concatenated router + real consumer.
        state['pg'] = prefill_gemm_prepare(weights, tokens=(16384, 3012, 753),
            budget_s=max(0, 210 - (time.monotonic()-started)), deadline=started+210,
            incumbent=lambda x, role: _pg_v11_dense(x, state, role))
    except Exception as exc:
        state['pg'] = None
        lines.append(f'PGC_REJECT block retained=v11 {type(exc).__name__}: {str(exc)[:160]}')
    try: _pg_tune_v11_groups(state, started+225, lines)
    except Exception as exc:
        lines.append(f'PGC_REJECT groups retained=v11 {type(exc).__name__}: {str(exc)[:130]}')
    # All compact selections print after compiler/rejection/profiler chatter.
    print('PGC_BEGIN V11 incumbent; three fresh probes; cold128MiB; paired20', flush=True)
    if state['pg'] is not None: prefill_gemm_summary(state['pg'])
    for line in lines: print(line, flush=True)
    print(f'PGC_END prepare_s={time.monotonic()-started:.3f} dense_deadline=210 group_deadline=225', flush=True)
    return state
