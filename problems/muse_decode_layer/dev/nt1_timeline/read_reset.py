"""Audit the three P3a reset receipts and reproduce the stage-only nominations."""
import ast
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

from read_pipe import unpack

D = Path(__file__).resolve().parent
R = D.parents[3]
JOBS = {
    'epilogue': ('d650c04ead114119', '917e79f37f28b5c1', '2b0356d740f0b2ca', 16),
    'epilogue_next': ('29ad388de976432c', 'e23854aca7a11da1', '642d6ae01ccd9194', 32),
    'pipe': ('16c180784f414576', '875038058dfb252d', 'd53196031b271014', 16),
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    report = {'ratio_convention': 'candidate time / control time; below one wins', 'runs': {}}
    logs = {}
    for name, (job, wrapper, payload, seeds) in JOBS.items():
        paths = list((R / 'runs').glob(f'*/*_{job}.json'))
        assert len(paths) == 1, (job, paths)
        path = paths[0]
        receipt = json.loads(path.read_text())
        result = receipt['result']
        assert receipt['id'] == job and receipt['status'] == 'done'
        assert result['passed'] and result['checks']['ok'] and not result['error']
        assert result['gpu'] == 'H200'
        source = path.with_suffix('.py').read_bytes()
        assert digest(source).startswith(wrapper)
        entries = next(ast.literal_eval(n.value) for n in ast.parse(source).body
                       if isinstance(n, ast.Assign) and any(getattr(t, 'id', '') == '_LAB_PAYLOADS' for t in n.targets))
        assert len(entries) == 1 and entries[0][1] == payload
        assert digest(entries[0][2].encode()).startswith(payload)
        log = result['log_tail']
        assert f'LAB_BEGIN {entries[0][0]} {payload}' in log and 'LAB_END' in log
        assert not re.search(r'LAB_ERROR|LAB_TRUNCATED|^FAIL |^BUDGET ', log, re.M)
        assert re.search(rf'END seeds={seeds}\b', log)
        gates = re.findall(r'(?:^GATE |,)([A-Z_0-9]+):([\d.]+)/([\d.]+)', log, re.M)
        if name == 'pipe':
            gate = re.search(r'GATE S14 worst=\(([\d.e+-]+), ([\d.e+-]+)\)', log)
            assert gate
            gates = [('S14', *gate.groups())]
        assert gates and all(float(total) < .015 and float(row) < .075 for _, total, row in gates)
        aa = {int(m): float(us) for m, us in re.findall(r'^AA M(\d+) abs=([\d.]+) us', log, re.M)}
        if name == 'pipe':
            aa = {int(m): float(us) for m, us in re.findall(r'^CELL M(\d+).*AAabs=([\d.]+)', log, re.M)}
        assert set(aa) == ({1, 8, 16, 32} if name == 'epilogue' else {8, 16, 32})
        rows = []
        for match in re.finditer(r'^(PAIR|VS_S14|VS_PAIR) M(\d+) (\w+) us=([\d.]+) (?:r|ratio)=([\d.]+) se=([\d.]+) wins=(\d+)/(\d+)', log, re.M):
            label, m, arm, us, ratio, se, wins, count = match.groups()
            m, us, ratio, se = int(m), float(us), float(ratio), float(se)
            assert int(count) == seeds
            radius = aa[m] / (us / ratio)
            rows.append(dict(M=m, arm=arm, control='PAIR' if label == 'VS_PAIR' else 'S14',
                             ratio=ratio, se=se, n=int(count), wins=int(wins), arm_us=us,
                             AA_mean_abs_us=aa[m], AA_fraction=radius,
                             upper95_plus_AA=ratio + 1.96 * se + radius,
                             clears_summary_control=ratio + 1.96 * se + radius < 1))
        if name == 'pipe':
            for m, ratio in re.findall(r'^CELL M(\d+).*?ratio=([\d.]+)', log, re.M):
                rows.append(dict(M=int(m), arm='PDIA', control='S14', ratio=float(ratio),
                                 n=seeds, correct=False, promotable=False, interpretation='wrong-math diagnostic only'))
        report['runs'][name] = dict(job=job, receipt=str(path.relative_to(R)),
            receipt_sha256=digest(path.read_bytes()), wrapper_sha256=digest(source), payload_sha256=digest(entries[0][2].encode()),
            official_passed=True, seeds=seeds, gates={a: [float(x), float(y)] for a, x, y in gates},
            AA_mean_absolute_us=aa, rows=rows)
        logs[name] = log
    detail, = unpack(logs['pipe'])
    assert sum(line.startswith('PIPE ') for line in detail.splitlines()) == 96
    assert sum(line.startswith('WORK ') for line in detail.splitlines()) == 66
    assert sum(line.startswith('LATENCY ') for line in detail.splitlines()) == 6
    (D / 'h200_pipe.txt').write_text(detail + '\n')
    work = {(int(m), name, int(cap)): float(t) for m, name, cap, t in re.findall(
        r'^WORK M(\d+) (\w+) cap(\d+).*?node_us=([\d.]+)', detail, re.M)}
    rates = {(int(m), name, int(cap)): float(t) for m, name, cap, t in re.findall(
        r'^WORK M(\d+) (\w+) cap(\d+).*?packed_TBs=([\d.]+)', detail, re.M)}
    report['pipe_contrasts'] = {m: dict(
        ILP2_over_shared_W8=work[m, 'MMA_ILP2', 2] / work[m, 'MATH_SMEM_W8', 2],
        W16_over_W8=work[m, 'MATH_SMEM_W16', 2] / work[m, 'MATH_SMEM_W8', 2],
        ILP2_over_shared_W8_work_normalized=rates[m, 'MATH_SMEM_W8', 2] / rates[m, 'MMA_ILP2', 2],
        W16_over_W8_work_normalized=rates[m, 'MATH_SMEM_W8', 2] / rates[m, 'MATH_SMEM_W16', 2],
        L2_over_HBM=work[m, 'MATH_L2', 2] / work[m, 'MATH_HBM', 2],
        copy_S4_over_S2=work[m, 'COPY_S4', 2] / work[m, 'COPY_S2', 2]) for m in (8, 16, 32)}
    report['pipe_decision'] = ('ILP2 gives a small shared-math improvement at identical 2-CTA/16-warp residency. '
        'This supports a dependency hypothesis, not a proven full-kernel win; microprobes lack independent A:A. '
        'Send the prepared packed four-arm batch for paired TASK-gated measurement. '
        'More flight, more warps, and LUT are not identified fixes; PDIA cannot be promoted.')
    candidates=[]
    for batch,run in report['runs'].items():
        for row in run['rows']:
            if row['arm'] not in ('S14','PDIA') and row.get('ratio',1)<1:
                candidates.append(dict(job=run['job'],batch=batch,**row))
    report['P4_point_estimate_candidates']=candidates
    report['nomination_gate']='valid TASK-gated per-shape point estimate below one; SE and A:A describe uncertainty'
    report['promotion_gate']='P4 held-out per-seed six-load geomean lower confidence bound above one'
    (D / 'h200_results.json').write_text(json.dumps(report, indent=2) + '\n')
    print('3 official-passing receipts; 96 pipe / 66 work / 6 latency rows; all per-shape point winners handed to P4')


if __name__ == '__main__':
    main()
