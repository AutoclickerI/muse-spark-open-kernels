def br_timed(graph, ev0, ev1):
    lab_cold_l2()
    ev0.record()
    graph.replay()
    ev1.record()
    ev1.synchronize()
    return ev0.elapsed_time(ev1) * 1000.


def br_pair(a, b, ev0, ev1, repeats=6):
    aa, bb = [], []
    for r in range(repeats):
        br_need(.3, 'cold pair')
        for g, s in (((a, aa), (b, bb)) if r % 2 == 0 else ((b, bb), (a, aa))):
            s.append(br_timed(g, ev0, ev1))
    return statistics.median(aa), statistics.median(bb)


def br_phases(data, bank, arm, ev0, ev1):
    steps = br_plan(data, bank, arm)['steps']
    for f in steps:
        f()
    vals = []
    for k, fn in enumerate(steps):
        br_need(1., 'phase graph')
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            fn()
        ss = []
        for _ in range(4):
            for pre in steps[:k]:
                pre()
            ss.append(br_timed(g, ev0, ev1))
        vals.append(statistics.median(ss))
    return vals


BR_META_TUNE = ('META heavy-kernel lab at M%d vs LI_swap (701d2867 winner: N_full, heavy N32 lean K loop, W13 gate/up rows '
                'interleaved by 8 in prepare with SwiGLU in registers, grid %d, launched first on the side stream || NT1<=16; live '
                'route; one join). F<s>: finalizer split over s blocks per token (bit-equal). Probes, timing only: LIF_swap = '
                'LI_swap without the per-stage fence.proxy.async (bit = race diagnostic); P_heavy / P_nt1 = one branch alone; '
                'P_nofin = no finalizer (TASK FAIL by design). 12 seeds x3 fresh routes, lab_cold_l2, AB/BA x6, A:A.')
BR_META = {
    128: 'META M128 vs HH = exact S7 H_H (u8 codes; route128 zero-fill + pm_meta SS clear; NT1<=16 then WGMMA32>16 per phase; finish). '
         'U16 = exact u16-word H_H (S6). ',
    256: 'META M256 vs S6 = exact S6 mixed_parallel. HH = H_H policy on u8 (NT1<=16, WGMMA32>16). R_s6 = 110599ba R_full (S6 buffers). ',
}
BR_META_TAIL = ('HH_u8: H_H with this lab\'s typed-smem WGMMA. Other arms: live route 1CTA (T32/T8 lists), small W13 zeroes its SS rows, '
                'n=NT1<=16; N: heavy N32; NR: heavy remainder rows<=16 on N16 beside NT1; NRm: remainder on the heavy stream; '
                'B8: W2 NT1<=8 + N16 for 8<count<=16 beside it. phase=fork/join per phase, full=fork once, join at finish. '
                '12 seeds x3 fresh routes, judge-style read flush (lab_cold_l2), AB/BA x6 vs control, A:A.')


def lab(deadline, weights, state=None):
    global BR_DEADLINE
    BR_DEADLINE = deadline
    start = time.monotonic()
    names = BR_NAMES
    ctl = BR_CONTROL
    lines = [BR_META_TUNE % (BR_M, BR_GRID) if BR_SET == 'tune' else BR_META[BR_M] + BR_META_TAIL,
             f'GATE TASK FP32 L2<=.015,row<=.075,finite before+after timing (failures recorded, still timed); bit=equal to {ctl} output (diagnostic). Phase means are diagnostics.']
    rows, phases, aa = [], [], []
    try:
        br_need(25., 'before compile')
        br_modules()
        lines.append(f'BUILD s={time.monotonic() - start:.1f} left={deadline - time.monotonic():.1f}')
        dev = next(t.device for t in weights.values() if torch.is_tensor(t))
        ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        peak = 0.
        for seed in range(12):
            br_need(max(8., peak * 1.3 + 1.), f'before seed{seed}')
            tick = time.monotonic()
            w = dc_weights(5353 + 7919 * seed, dev)
            bank = br_prepare(w)
            ds = dc_fixtures(w, BR_M)
            wants = dc_oracles(ds, w)
            graphs = {}
            for arm in names:
                br_need(1., 'graph')
                graphs[arm] = MWGraph(lambda d, a=arm: br_call(d, bank, a), ds[0])
            per = []
            for fi, (data, want) in enumerate(zip(ds, wants)):
                for g in graphs.values():
                    g.refresh(data)
                base = graphs[ctl].replay().clone()
                torch.cuda.synchronize()
                err = {a: br_error(graphs[a].replay().clone(), want) for a in names}
                bit = {a: bool(torch.equal(base, graphs[a].replay())) for a in names}
                a0, a1 = br_pair(graphs[ctl], graphs[ctl], ev0, ev1)
                aa.append(a1 - a0)
                us = {ctl: a0}
                ratio = {ctl: 1.}
                for a in names[1:]:
                    c, x = br_pair(graphs[ctl], graphs[a], ev0, ev1)
                    us[a] = x
                    ratio[a] = x / c
                for a in names:
                    e2 = br_error(graphs[a].replay().clone(), want)
                    err[a] = (max(err[a][0], e2[0]), max(err[a][1], e2[1]))
                rows.append(dict(seed=seed, fixture=fi, us=us, ratio=ratio, err=err, bit=bit))
                per.append(ratio)
            br_need(4., 'phases')
            phases.append({a: br_phases(ds[0], bank, a, ev0, ev1) for a in names})
            lines.append(f'S{seed} ' + ' '.join(f'{statistics.mean(p[a] for p in per):.3f}' for a in names[1:]))
            peak = max(peak, time.monotonic() - tick)
            del graphs, bank
    except BRBudget as e:
        lines.append('BUDGET ' + str(e))
    if rows:
        lines.append('ORDER ' + ' '.join(names[1:]))
        for a in names:
            e = max(r['err'][a][0] for r in rows)
            rw = max(r['err'][a][1] for r in rows)
            rr = [r['ratio'][a] for r in rows]
            lines.append(f'MEAN {a} n={len(rows)} us={statistics.mean(r["us"][a] for r in rows):.2f} med={statistics.median(r["us"][a] for r in rows):.2f} '
                         f'pair={statistics.mean(rr):.4f} lo/hi={min(rr):.3f}/{max(rr):.3f} L2/row={e:.5f}/{rw:.5f} TASK={"PASS" if e <= .015 and rw <= .075 else "FAIL"} bit={sum(r["bit"][a] for r in rows)}')
        if phases:
            for a in names:
                k = len(phases[0][a])
                lines.append(f'PH {a} ' + '/'.join(f'{statistics.mean(p[a][i] for p in phases):.1f}' for i in range(k)))
        lines.append(f'AA signed/abs/max={statistics.mean(aa):.3f}/{statistics.mean(abs(x) for x in aa):.3f}/{max(abs(x) for x in aa):.3f}')
    lines.append(f'END M={BR_M} seeds={len(rows) // 3}/12 used={time.monotonic() - start:.1f} left={deadline - time.monotonic():.1f}')
    return lines
