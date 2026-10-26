"""Audit a P3a four-arm H200 receipt without treating failed/incomplete controls as losses."""
import argparse
import ast
import hashlib
import json
import math
import re
from pathlib import Path

D = Path(__file__).resolve().parent
R = D.parents[3]
ARMS = ('FLIGHT', 'ILP', 'K16', 'LUT')
LEAVES_OPEN = {
    'FLIGHT': 'Other lookahead distances, issuing agents and schedules that avoid competing with useful TMA work.',
    'ILP': 'Other chain assignments, decode/activation dependency cuts and register budgets; combination with PAIR is unmeasured.',
    'K16': 'Other K partitions and occupancy layouts without this shared B-cache/epilogue cost; higher occupancy generally is not closed.',
    'LUT': 'Other lossless code layouts and decoder instruction reductions without this table/packing/shared-service cost.',
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def audit(path):
    receipt = json.loads(path.read_text())
    result = receipt['result']
    source = path.with_suffix('.py').read_bytes()
    expected_wrapper = json.loads((D / 'followup_queue.json').read_text())['sha256']
    assert sha(source) == expected_wrapper
    entries = next(ast.literal_eval(n.value) for n in ast.parse(source).body
                   if isinstance(n, ast.Assign) and any(getattr(t, 'id', '') == '_LAB_PAYLOADS' for t in n.targets))
    name, short_hash, payload = next(e for e in entries if e[0] == 'moe_nt1_pipe_rebuild')
    expected = json.loads((D / 'pipe_rebuild_validation.json').read_text())['sha256']
    assert sha(payload.encode()) == expected and short_hash == expected[:16]
    manifest = json.loads((D / 'pipe_ready_manifest.json').read_text())
    for entry in manifest.values():
        cell = (R / entry['cell_path']).read_text()
        assert sha(cell.encode()) == entry['cell_sha256'] and cell.strip() in payload
    log = result.get('log_tail', '')
    sections = re.findall(r'^LAB_BEGIN moe_nt1_pipe_rebuild .*?(?:^LAB_END.*?$|\Z)', log, re.M | re.S)
    assert len(sections) == 1, 'missing or duplicate lab section'
    text = sections[0]
    gates = {arm: (float(total), float(row)) for arm, total, row in re.findall(
        r'(?:^GATE |,)([A-Z_0-9]+):([\d.]+)/([\d.]+)', text, re.M)}
    aa = {int(m): float(us) for m, us in re.findall(r'^AA M(\d+) abs_us=([\d.]+)', text, re.M)}
    failures = {arm: reason for arm, reason in re.findall(r'^FAIL (\w+) (.*)$', text, re.M)}
    end = re.search(r'^END seeds=(\d+)\b', text, re.M)
    global_reasons = []
    if result.get('gpu') != 'H200':
        global_reasons.append('not an H200 receipt')
    if receipt.get('status') != 'done' or not result.get('passed') or not result.get('checks', {}).get('ok'):
        global_reasons.append('official host did not pass')
    if not end or int(end[1]) != 32:
        global_reasons.append('not all 32 seeds completed')
    if re.search(r'LAB_ERROR|LAB_TRUNCATED|^BUDGET ', text, re.M):
        global_reasons.append('lab error, truncation or exhausted budget')
    if 'S14' in failures or 'S14' not in gates or gates['S14'][0] > .015 or gates['S14'][1] > .075:
        global_reasons.append('S14 correctness control failed or absent')
    if set(aa) != {8, 16, 32} or not all(math.isfinite(v) and v >= 0 for v in aa.values()):
        global_reasons.append('A:A control incomplete or invalid')
    rows = []
    pattern = (r'^PAIR M(\d+) (\w+) S14_us=([\d.]+) arm_us=([\d.]+) '
               r'ratio=([\d.]+) se=([\d.]+) wins=(\d+)/(\d+) delta_us=([\d.+-]+)')
    for m, arm, control, candidate, ratio, se, wins, n, delta in re.findall(pattern, text, re.M):
        assert arm in ARMS
        m, n = int(m), int(n)
        assert m in (8, 16, 32)
        control, candidate, ratio, se, delta = map(float, (control, candidate, ratio, se, delta))
        reasons = list(global_reasons)
        if arm in failures:
            reasons.append(failures[arm])
        if arm not in gates or gates[arm][0] > .015 or gates[arm][1] > .075:
            reasons.append('arm TASK gate failed or absent')
        if n != 32:
            reasons.append('arm has fewer than 32 paired seeds')
        assert control > 0 and candidate > 0 and ratio > 0 and se >= 0
        radius = aa[m] / control if m in aa else None
        upper = ratio + 1.96 * se + radius if radius is not None else None
        lower = ratio - 1.96 * se - radius if radius is not None else None
        decision = ('unmeasured' if reasons else 'nominate to P4' if upper < 1
                    else 'close exact recipe at this M; no measured gain')
        rows.append(dict(M=m, arm=arm, control_us=control, arm_us=candidate, ratio=ratio, se=se,
            n=n, wins=int(wins), delta_us=delta, AA_mean_abs_us=aa.get(m), AA_fraction=radius,
            upper95_plus_AA=upper, lower95_minus_AA=lower, decision=decision, invalid_reasons=reasons,
            leaves_open=LEAVES_OPEN[arm]))
    assert len({(row['M'], row['arm']) for row in rows}) == len(rows), 'duplicate paired row'
    for arm in ARMS:
        for m in (8, 16, 32):
            if not any(row['arm'] == arm and row['M'] == m for row in rows):
                rows.append(dict(M=m, arm=arm, decision='unmeasured',
                    invalid_reasons=global_reasons + [failures.get(arm, 'paired row absent')],
                    leaves_open=LEAVES_OPEN[arm]))
    return dict(job=receipt['id'], receipt=str(path.relative_to(R)), receipt_sha256=sha(path.read_bytes()),
        wrapper_sha256=sha(source), payload_sha256=expected, official_passed=result.get('passed', False),
        global_control_reasons=global_reasons, completed_seeds=int(end[1]) if end else None,
        gates=gates, AA_mean_abs_us=aa, arm_failures=failures, rows=rows,
        nominated_M_by_arm={arm: [row['M'] for row in rows if row['arm'] == arm and row['decision'] == 'nominate to P4']
                           for arm in ARMS},
        whole_layer_verified=False, comparison_with_PAIR_verified=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipt', type=Path)
    args = parser.parse_args()
    report = audit(args.receipt.resolve())
    (D / 'rebuild_results.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
