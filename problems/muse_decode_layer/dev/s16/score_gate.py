"""One promotion rule: held-out paired six-load log-geomean and TASK correctness."""
import math
import statistics

LOADS = ((1, 1), (16, 1), (32, 1), (1, 8), (16, 8), (32, 8))
# Two-sided 95% Student-t critical values, df 1..30. Above 30 use the
# next smaller tabulated df (conservative, independent of observed results).
T95 = (0, 12.7062, 4.3027, 3.1825, 2.7764, 2.5706, 2.4469, 2.3646,
       2.3060, 2.2622, 2.2281, 2.2010, 2.1788, 2.1604, 2.1448, 2.1315,
       2.1199, 2.1098, 2.1009, 2.0930, 2.0860, 2.0796, 2.0739, 2.0687,
       2.0639, 2.0595, 2.0555, 2.0518, 2.0484, 2.0452, 2.0423)


def interval(ratios, aa=None):
    """Geometric mean with a t interval in log space; A:A estimates noise.

    Keep the larger paired/A:A SE, without imposing a per-load threshold or
    subtracting A:A's mean. Samples are independent seed pairs, not replays.
    """
    assert len(ratios) >= 2
    assert all(math.isfinite(x) and x > 0 for x in ratios)
    logs = list(map(math.log, ratios)); n = len(logs)
    se = statistics.stdev(logs) / math.sqrt(n)
    aa_se = 0.
    if aa is not None:
        assert len(aa) == n and all(math.isfinite(x) and x > 0 for x in aa)
        aa_se = statistics.stdev(map(math.log, aa)) / math.sqrt(n)
    error = T95[min(n - 1, 30)] * max(se, aa_se)
    center = statistics.mean(logs)
    return dict(R=math.exp(center), CI95=[math.exp(center-error), math.exp(center+error)],
                n=n, log_se=se, AA_log_se=aa_se, method='95% Student-t on paired seed log R; max(paired, A:A) SE')


def scores(seed_times, old, new):
    """Same seed index across twelve shapes, with 46/16 weighted load sums."""
    n = len(next(iter(seed_times.values())))
    assert len(seed_times) == 12 and all(len(v) == n for v in seed_times.values())
    per_seed = []; by_load = {}
    for b, q in LOADS:
        label = ('d' if q == 1 else 'v') + str(b)
        s, f = seed_times[label+'S'], seed_times[label+'F']
        by_load[label] = [(46*x[old]+16*y[old])/(46*x[new]+16*y[new]) for x,y in zip(s,f)]
    for i in range(n):per_seed.append(math.exp(statistics.mean(math.log(v[i]) for v in by_load.values())))
    return per_seed, by_load


def promotion(raw, old, new, *, official_pass, selection_seeds):
    seeds = raw['seed_ids']
    assert len(seeds) == raw['seeds'] and len(set(seeds)) == len(seeds)
    assert not set(seeds).intersection(selection_seeds), 'selection and confirmation seeds overlap'
    samples={k:v['samples'] for k,v in raw['loads'].items()}
    assert len(samples)==6 and all(len(v)==len(seeds) for v in samples.values())
    loads={k:[r[old]/r[new] for r in rows] for k,rows in samples.items()}
    rr=[math.exp(statistics.mean(math.log(v[i]) for v in loads.values())) for i in range(len(seeds))]
    aa=[math.exp(statistics.mean(math.log(rows[i]['A2']/rows[i]['A']) for rows in samples.values())) for i in range(len(seeds))]
    result = interval(rr, aa)
    arms = set(next(iter(samples.values()))[0])
    expected = 12 * len(seeds) * len(arms)
    correct = bool(official_pass and raw['complete'] and not raw['failures']
                   and raw['task_checks'] == expected)
    result.update(promote=correct and result['CI95'][0] > 1, TASK_pass=correct,
                  seed_ids=seeds, selection_seeds=list(selection_seeds),
                  per_seed_R=rr, loads={k:interval(v) for k,v in loads.items()})
    return result


def legacy_summary(loads, arm):
    """Descriptive only: old receipts discarded the seed pairs/covariances.

    R of load means and a worst-correlation delta-method t interval are useful
    for auditing old decisions, never inputs to promotion().
    """
    values = [row['r'][arm] for row in loads.values()]
    assert len(values) == 6
    center = statistics.mean(math.log(v[0]) for v in values)
    se = statistics.mean(v[1]/v[0] for v in values)
    error = T95[15] * se
    return dict(R=math.exp(center), CI95_approx=[math.exp(center-error),math.exp(center+error)],
                CI95=None, promote=False, reason='paired seed samples and per-arm TASK checks absent; descriptive summary only')
