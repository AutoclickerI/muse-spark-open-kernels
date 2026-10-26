"""Host-independent promotion from paired native evidence and the official TASK result.

New integrations emit arm_status with source identity, native, complete, seed_ids,
TASK/nearby counts and failures per arm. Failures have scope (arm/device/run), arms
and reason. A device failure invalidates every arm listed, or all arms if unscoped.
The historical complete S18 archive is supported through its verified wrapper manifest;
incomplete older archives lack enough attribution and cannot establish a measurement.
"""
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



LABELS = {"decode 1", "decode 16", "decode 32", "verify 1", "verify 16", "verify 32"}


def _protocol_arm_valid(raw, manifest, name, seed_count):
    """Require the prospectively declared sampling plan, without changing archived gates."""
    plan = manifest.get('measurement_protocol')
    if plan is None:
        return True
    if (raw.get('measurement_protocol') != plan
            or plan.get('name') != 'palindromic-cohort-v1'
            or plan.get('aggregation') != 'arithmetic-mean-all-rounds'
            or plan.get('selective_retiming') is not False
            or plan.get('seed_count') != seed_count):
        return False
    rounds = plan.get('rounds')
    if not isinstance(rounds, int) or rounds <= 0 or rounds % 2:
        return False
    labels = manifest['arms'].get(name, {}).get('rows') or [
        ('d' if q == 1 else 'v') + str(b) + suffix
        for b, q in LOADS for suffix in ('S', 'F')]
    counts = raw.get('timing_round_counts', {})
    for label in labels:
        values = counts.get(label, {}).get(name)
        if not isinstance(values, list) or len(values) != seed_count or any(v != rounds for v in values):
            return False
    return True


def evaluate(receipt, raw, manifest):
    """The host selects transport/quota only. Callers verify wrapper and official-source hashes."""
    old = manifest.get('baseline_arm', 'INC')
    identities = {'A': manifest['sha256'], 'A2': manifest['sha256'],
                  **{name: spec['sha256'] for name, spec in manifest['arms'].items()}}
    if manifest.get('base_sha256'):
        assert identities[old] == manifest['base_sha256'], 'incumbent identity mismatch'
    assert raw['host_sha256'] == identities['A'], 'candidate identity mismatch'
    assert raw['checker_sha256'] == manifest['checker_sha256'], 'TASK checker identity mismatch'
    seeds = raw['seed_ids']
    seed_ok = (len(seeds) >= 2 and len(seeds) == raw['seeds'] == len(set(seeds))
               and seeds == manifest['confirmation_seeds']
               and not set(seeds).intersection(manifest['selection_seeds']))
    samples = {k: v.get('samples', []) for k, v in raw['loads'].items()}
    loads_ok = set(samples) == LABELS and all(len(v) == len(seeds) for v in samples.values())
    expected_task = {name: len(manifest['arms'].get(name, {}).get('rows') or range(12))*len(seeds)
                     for name in identities}
    nearby_arms = manifest.get('nearby_arms', ['A', old])
    nearby_per_arm = manifest.get('nearby_expected_checks', 260)//len(nearby_arms)
    expected_nearby = {name: nearby_per_arm if name in nearby_arms else 0 for name in identities}
    statuses = raw.get('arm_status')
    if statuses is None:
        # Complete archived S18 evidence: wrapper contains all source hashes, native TASK checks,
        # counts and paired samples. Never infer per-arm health from an incomplete old archive.
        valid = bool(raw['complete'] and not raw['failures']
                     and raw['task_checks'] == manifest['expected_TASK_checks'])
        nearby = raw.get('nearby_task_checks') == manifest.get('nearby_expected_checks', 260)
        statuses = {name: dict(sha256=sha, native=raw.get('native', receipt['result'].get('gpu') == 'H200'), complete=valid,
                              seed_ids=seeds, task_checks=expected_task[name], expected_task_checks=expected_task[name],
                              nearby_task_checks=expected_nearby[name] if nearby else -1,
                              nearby_expected_checks=expected_nearby[name], failures=[])
                    for name, sha in identities.items()}
    invalid = set()
    for failure in raw.get('failures', []):
        if not isinstance(failure, dict):
            invalid.update(identities)  # unattributed failure is not isolated optional evidence
        else:
            invalid.update(failure.get('arms') or identities)
    def health(name):
        s = statuses.get(name, {})
        timing = bool(seed_ok and loads_ok and name not in invalid and not s.get('failures')
                      and s.get('sha256') == identities.get(name) and s.get('native') is True
                      and s.get('complete') is True and s.get('seed_ids') == seeds
                      and _protocol_arm_valid(raw, manifest, name, len(seeds)))
        for rows in samples.values():
            for row in rows:
                value = row.get(name)
                timing &= isinstance(value, (float, int)) and math.isfinite(value) and value > 0
        task = bool(timing and s.get('task_checks') == s.get('expected_task_checks') == expected_task[name]
                    and expected_task[name] > 0)
        nearby = (s.get('nearby_task_checks') == s.get('nearby_expected_checks') == expected_nearby[name])
        return timing and task, task and nearby
    arm_health = {name: health(name) for name in identities}
    official = receipt['result'].get('passed') is True
    def compare(older):
        needed = {older, 'A', 'A2'}
        measured = all(arm_health.get(n, (False, False))[0] for n in needed)
        if not measured:
            return dict(R=None, CI95=None, measured=False, TASK_pass=False, promote=False,
                        reason='Missing or invalid native paired evidence/control for '+','.join(sorted(needed)))
        loads = {k: [r[older]/r['A'] for r in rows] for k, rows in samples.items()}
        rr = [math.exp(statistics.mean(math.log(v[i]) for v in loads.values())) for i in range(len(seeds))]
        aa = [math.exp(statistics.mean(math.log(rows[i]['A2']/rows[i]['A']) for rows in samples.values()))
              for i in range(len(seeds))]
        out = interval(rr, aa)
        correct = official and all(arm_health[n][1] for n in needed)
        out.update(measured=True, TASK_pass=correct, promote=correct and out['CI95'][0] > 1,
                   seed_ids=seeds, selection_seeds=list(manifest['selection_seeds']), per_seed_R=rr,
                   loads={k: interval(v) for k, v in loads.items()})
        return out
    score = compare(old)
    return dict(job=receipt['id'], source_sha256=manifest['sha256'], incumbent_sha256=identities[old],
                freeze_gate=score['promote'], measured=score['measured'], score=score,
                pieces={name: compare(name) for name in manifest['arms'] if name != old},
                arm_status=statuses, raw=raw,
                piece_interpretation='Leave-out/A diagnostic R; >1 favors retaining the piece. No per-piece veto. '
                'A composition selected using these confirmations needs new disjoint confirmation seeds.')
