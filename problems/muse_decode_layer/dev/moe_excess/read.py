"""Decode and validate the matched S19 ladder; report seed-paired contrasts.

No diagnostic arm is a promotion candidate. Native H200 evidence is mandatory
unless --local is supplied; local reports are written only below ignored out/.
"""
import argparse
import base64
import json
import lzma
import math
import statistics as st
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / 'tools'))
from lab_packet import read_packets
from promotion import T95, interval

ARMS = ('BASE', 'AA', 'MATCH', 'SHADOW', 'L0', 'L1', 'L2', 'L3',
        'MATH', 'INIT', 'STREAM', 'HIST')
CONTRASTS = {
    'decode/scale | loaded operands': ('L1', 'L0'),
    'MMA | loaded decoded operands': ('L2', 'L1'),
    'epilogue/readiness | MMA': ('L3', 'L2'),
    'load issue/reuse/residency | unique stream': ('L0', 'STREAM'),
    'on-chip math incl. epilogue, initialization removed': ('MATH', 'INIT'),
    'sink/control tax': ('L3', 'MATCH'),
    'clone control': ('SHADOW', 'MATCH'),
    'allocation control': ('MATCH', 'BASE'),
    'instrumentation control': ('HIST', 'SHADOW'),
}


def decode(blob):
    data = lzma.decompress(base64.b85decode(blob))
    size, = struct.unpack_from('<I', data)
    schema = json.loads(data[4:4 + size])
    position = 4 + size

    def integer():
        nonlocal position
        result = shift = 0
        while True:
            assert position < len(data), 'truncated integer stream'
            byte = data[position]
            position += 1
            result |= (byte & 127) << shift
            if byte < 128:
                break
            shift += 7
            assert shift < 70, 'invalid integer'
        return -(result // 2) - 1 if result & 1 else result // 2

    def vector(n):
        result = []
        previous = 0
        for _ in range(n):
            previous += integer()
            result.append(previous)
        return result

    def walk(value):
        if isinstance(value, dict):
            return {k: walk(v) for k, v in value.items()}
        if isinstance(value, list):
            if value and value[0] == '$int':
                return vector(value[1])
            if value and value[0] == '$matrix':
                columns = [vector(value[1]) for _ in range(value[2])]
                return list(map(list, zip(*columns)))
            return [walk(v) for v in value]
        return value

    result = walk(schema)
    assert position == len(data), 'unconsumed integer stream'
    return result


def mean_ci(values):
    n = len(values)
    center = st.mean(values)
    error = T95[min(n - 1, 30)] * st.stdev(values) / math.sqrt(n)
    return dict(us=center, CI95=[center - error, center + error], n=n)


def read(path, local=False):
    text = path.read_text()
    receipt = None
    if path.suffix == '.json':
        receipt = json.loads(text)
        assert receipt['result']['gpu'] == 'H200', 'native H200 receipt required'
        queue = json.loads((HERE / 'queue.json').read_text())
        assert receipt['content_hash'] == queue['wrapper_sha256'], 'wrapper identity mismatch'
        text = receipt['result']['log_tail']
    else:
        assert local, 'receipt JSON required; raw lines are local evidence only'
    records = read_packets(text)
    if receipt:
        assert records, 'missing checksummed transport'
        assert receipt['status'] == 'done', receipt['status']
    manifest = json.loads((HERE / 'manifest.json').read_text())
    lines = []
    for record in records:
        if record['name'] not in ('large_moe_excess', 'ladder'):
            continue
        assert record['source_sha256'] == manifest['payload_sha256'][:16]
        assert not record['packet'].get('transport_errors'), record['packet']
        lines.extend(record['packet'].get('lines', []))
    if not records:
        lines = text.splitlines()
    blobs = [line.split(' ', 1)[1] for line in lines
             if line.startswith('LADDER_RESULT_BIN85 ')]
    assert len(blobs) == 1, 'missing or ambiguous ladder result'
    raw = decode(blobs[0])
    assert raw['complete'] is True
    assert raw['seeds'] == manifest['seeds']
    assert raw['weight_seeds'] == manifest['weight_seeds']
    assert raw['identities_sha256'] == manifest['identities_sha256']
    assert raw['telemetry_error'] is None, raw['telemetry_error']
    labels = {f'{m}:{phase}' for m in (128, 256) for phase in ('w13', 'w2')}
    assert set(raw['round_ns']) == labels
    for seeds in raw['round_ns'].values():
        assert len(seeds) == 8
        for row in seeds:
            assert set(row) == set(ARMS)
            for values in row.values():
                assert len(values) == manifest['rounds'] == 4
                assert all(type(v) is int and v > raw['fixed_ns'] for v in values)
    expected = {(seed, label) for seed in range(8) for label in labels}
    assert {(s, label) for s, label, *_ in raw['telemetry']} == expected
    assert len(raw['telemetry']) == 32
    for _, _, readings, host in raw['telemetry']:
        assert len(readings) == 5
        assert all(len(row) == 5 and min(row) > 0 for row in readings)
        assert set(host) == {'BASE', 'AA'}
        assert all(len(row) == 4 for row in host.values())
    assert {(s, label) for s, label, _ in raw['histograms']} == expected
    for when in ('before', 'after'):
        assert raw['checks'][when] == {arm: 32 for arm in
                                      ('BASE', 'AA', 'MATCH', 'SHADOW', 'L3', 'HIST')}
    assert len(raw['work_counts']) == 16
    counts = {(s, m): c for s, m, c in raw['work_counts']}
    assert set(counts) == {(s, m) for s in range(8) for m in (128, 256)}
    assert all(len(c) == 32 and all(type(x) is int and 0 <= x <= m for x in c)
               for (s, m), c in counts.items())
    for seed, label, hist in raw['histograms']:
        m, phase = label.split(':')
        c = counts[seed, int(m)]
        ntile = 512 if phase == 'w13' else 256
        expected_tiles = [sum((n + 63) // 64 for n in c if n > 16) * ntile // 8,
                          sum(1 for n in c if 8 < n <= 16) * ntile // 4,
                          sum(1 for n in c if 0 < n <= 8) * ntile]
        grids = raw['resources'][label + ':HIST']['grids']
        for branch, (g, tiles) in enumerate(zip(grids, expected_tiles)):
            h = hist['branches'][branch]
            for marginal in h['CTA_hist_tiles_experts']:
                assert sum(n for _, n in marginal) == g
            assert sum(t * n for t, n in h['CTA_hist_tiles_experts'][0]) == tiles
            assert h['last_CTA'][1] >= h['last_CTA'][0] >= 0
        smhist = hist['SM_hist_CTAs_weighttiles_experts']
        assert all(sum(n for _, n in marginal) == hist['SMs'] for marginal in smhist)
        assert sum(t * n for t, n in smhist[0]) == sum(grids)
        assert sum(t * n for t, n in smhist[1]) == sum(t * scale for t, scale in
                                                                  zip(expected_tiles, (8, 4, 1)))
    return raw, receipt


def summarize(raw, receipt):
    samples = {label: {arm: [(st.mean(seed[arm]) - raw['fixed_ns']) / 1000
                            for seed in seeds] for arm in ARMS}
               for label, seeds in raw['round_ns'].items()}
    report = dict(native_H200=receipt is not None,
                  receipt=receipt['id'] if receipt else None,
                  method='95% t7 over eight seed means of all four retained rounds; paired contrasts',
                  ladder={}, contrasts={}, controls={}, work={}, ranking={},
                  interpretation='Conditional contrasts change overlap and instruction dependencies. '
                  'They are not independent hardware costs; no diagnostic can promote.')
    for label, arms in samples.items():
        report['ladder'][label] = {arm: mean_ci(x) for arm, x in arms.items()}
        report['contrasts'][label] = {
            name: mean_ci([a - b for a, b in zip(arms[hi], arms[lo])])
            for name, (hi, lo) in CONTRASTS.items()}
        report['controls'][label] = {
            name: interval([a / b for a, b in zip(arms[hi], arms[lo])])
            for name, (hi, lo) in dict(AA=('AA', 'BASE'), allocation=('MATCH', 'BASE'),
                                       clone=('SHADOW', 'MATCH'), sink=('L3', 'SHADOW'),
                                       observer=('HIST', 'SHADOW')).items()}
        report['controls'][label]['resources'] = {
            arm: raw['resources'][label + ':' + arm] for arm in ARMS
            if label + ':' + arm in raw['resources']}
        report['controls'][label]['occupancy_equal_BASE'] = all(
            info[3] == info[4] for arm in ARMS if label + ':' + arm in raw['resources']
            for info in raw['resources'][label + ':' + arm]['info'])
        report['controls'][label]['spills_present'] = any(
            info[6] for arm in ARMS if label + ':' + arm in raw['resources']
            for info in raw['resources'][label + ':' + arm]['info'])
    for m in (128, 256):
        phases = [samples[f'{m}:{phase}'] for phase in ('w13', 'w2')]
        ranked = {}
        for name, (hi, lo) in list(CONTRASTS.items())[:4]:
            values = [sum(phase[hi][i] - phase[lo][i] for phase in phases) for i in range(8)]
            ranked[name] = mean_ci(values)
        largest = max(ranked, key=lambda name: ranked[name]['us'])
        ranked[largest]['largest_conditional_contrast'] = True
        report['ranking'][str(m)] = ranked
        for phase in ('w13', 'w2'):
            label = f'{m}:{phase}'
            hs = [h for _, ll, h in raw['histograms'] if ll == label]
            n = 8192 if phase == 'w13' else 4096
            unique = [sum(x > 0 for x in c) * (n * 2304 + (8 if phase == 'w13' else 4))
                      + (0 if phase == 'w13' else 8192)
                      for _, mm, c in raw['work_counts'] if mm == m]
            report['work'][label] = dict(
                unique_weight_operand_bytes=unique,
                unique_stream_4_8TBps_us=[x / 4.8e6 for x in unique],
                last_CTA_by_seed=[h['branches'] for h in hs],
                SM_histograms_by_seed=[{k: v for k, v in h.items() if k != 'branches'} for h in hs],
                warning='256 ns histogram bins; exact last-CTA timestamps. One final HIST replay per seed. '
                        'Observed tail is not automatically removable or additive to ladder contrasts.')
    # Existing S19 timeline, 46 sliding +16 full weighted per-layer wall. This is
    # a transfer scenario, not a whole-layer paired measurement or promotion gate.
    wall = {128: 321.09, 256: 425.66}
    excess = {128: 32.63, 256: 67.58}
    ceilings = {}
    for m in (128, 256):
        key = max(report['ranking'][str(m)], key=lambda k: report['ranking'][str(m)][k]['us'])
        hi, lo = CONTRASTS[key]
        gains = [sum(samples[f'{m}:{p}'][hi][i] - samples[f'{m}:{p}'][lo][i]
                     for p in ('w13', 'w2')) for i in range(8)]
        rr = [(wall[m] / (wall[m] - max(0, min(gain, excess[m])))) ** (1 / 6)
              for gain in gains]
        ceilings[str(m)] = dict(term=key, ceiling=interval(rr),
                                cap_us=excess[m], wall_us=wall[m],
                                interpretation='Only this load changes; full transfer of positive contrast, '
                                'capped at timeline excess above 1.3x roof. Model; no G0 verdict.')
    report['six_load_scenarios'] = ceilings
    report['promotion'] = 'No candidate: excluded-work ladder cannot be evaluated as a speedup.'
    return report


def markdown(report):
    title = 'H200 matched S19 ladder' if report['native_H200'] else 'LOCAL sm120 ONLY: no H200 attribution'
    lines = ['# ' + title, '', report['method'], '',
             '| M / phase | arm | us [95% CI] |', '|---|---|---:|']
    def cell(value):
        return f"{value['us']:.3f} [{value['CI95'][0]:.3f}, {value['CI95'][1]:.3f}]"
    for label, rows in report['ladder'].items():
        lines += [f'| {label} | {arm} | {cell(rows[arm])} |' for arm in ARMS]
    lines += ['', '| M / phase | conditional contrast | us [95% CI] |', '|---|---|---:|']
    for label, rows in report['contrasts'].items():
        lines += [f'| {label} | {term} | {cell(value)} |' for term, value in rows.items()]
    lines += ['', report['interpretation'], '',
              'Resource/observer controls, full work histograms and transfer scenarios are in the JSON. '
              'Ranking requires those controls and the native receipt; a histogram tail alone does not '
              'establish a scheduling win. A fresh six-load paired candidate must pass tools/promotion.py.']
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipt', type=Path)
    parser.add_argument('--local', action='store_true')
    args = parser.parse_args()
    raw, receipt = read(args.receipt, args.local)
    report = summarize(raw, receipt)
    dest = HERE / 'out/local_read' if args.local else HERE / 'result'
    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'raw.json').write_text(json.dumps(raw, indent=2) + '\n')
    (dest / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    (dest / 'ladder.md').write_text(markdown(report))
    print(dest / 'ladder.md')
