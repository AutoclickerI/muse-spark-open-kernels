"""Decode and validate the S22 large-M load-path split (payloads J and I); report paired contrasts,
duplicated-byte reuse cost from the actual tiling, and each part's six-load transfer scenario.

Loads-only arms are excluded-work oracles and never promote. PF is full math (TASK-checked,
bit-compared with BASE); its phase ratio ranks it for a production whole-load build only.
Usage: read.py <J receipt.json> <I receipt.json> [--local]
"""
import argparse, hashlib, json, math, statistics as st, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT / 'tools')); sys.path.insert(0, str(HERE.parent))
from lab_packet import read_packets
from promotion import T95, interval
from read import decode  # the ladder's exact varint/lzma codec

LABELS = [f'{m}:{p}' for m in (128, 256) for p in ('w13', 'w2')]
# Ladder transfer scenario (1b75cd73): S19 per-layer walls at verify 16 / 32 (46 sliding + 16 full
# weighted); MoE is byte-identical in S22. Other four loads unchanged.
WALL = {128: 321.09, 256: 425.66}
EB = {'w13': 8192 * 2304, 'w2': 4096 * 2304}
CONTRASTS_J = {
    'ladder term L0 - STREAM': ('L0', 'STREAM'),
    'sink tax L0 - L0N': ('L0', 'L0N'),
    'loads-only term L0N - STREAM': ('L0N', 'STREAM'),
    'activation copies L0N - NOX': ('L0N', 'NOX'),
    'weight rereads L0N - REUSE': ('L0N', 'REUSE'),
    'per-thread issue L0N - ISSUE': ('L0N', 'ISSUE'),
    'heavy 1 vs 2 CTAs/SM (S4) RES1 - L0N': ('RES1', 'L0N'),
    'heavy ring 3 vs 4 at 2/SM RES2S3 - L0N': ('RES2S3', 'L0N'),
    'heavy 2 vs 3 CTAs/SM (S3) RES2S3 - RES3': ('RES2S3', 'RES3'),
    'weight bytes in flight L0N - PFN': ('L0N', 'PFN'),
    'observer HISTN - L0N': ('HISTN', 'L0N'),
    'full math BASE - PF': ('BASE', 'PF'),
    'A:A AA - BASE': ('AA', 'BASE'),
}
CONTRASTS_I = {
    'heavy alone L0N - STREAM': ('H_L0N', 'H_ST'),
    'medium alone L0N - STREAM': ('M_L0N', 'M_ST'),
    'small alone L0N - STREAM': ('S_L0N', 'S_ST'),
    'joint L0N - STREAM': ('L0N', 'STREAM'),
    'A:A AA - BASE': ('AA', 'BASE'),
}


def mean_ci(values):
    n = len(values); c = st.mean(values); e = T95[min(n - 1, 30)] * st.stdev(values) / math.sqrt(n)
    return dict(us=c, CI95=[c - e, c + e], n=n)


def load(path, name, local):
    text = path.read_text(); receipt = None
    manifest = json.loads((HERE / 'manifest.json').read_text())
    if path.suffix == '.json':
        receipt = json.loads(text)
        queue = json.loads((HERE / 'queue.json').read_text())[name]
        assert receipt['id'] == queue['job'] and hashlib.sha256(path.with_suffix('.py').read_bytes()).hexdigest() == queue['wrapper_sha256'], 'wrapper identity'
        assert receipt['result']['gpu'] == 'H200', 'native H200 receipt required'
        assert receipt['status'] == 'done', receipt['status']
        text = receipt['result']['log_tail']
        records = read_packets(text)
        assert records, 'missing checksummed transport'
        lines = [l for r in records if r['name'].startswith('moe_split') or 'split' in r['name'] for l in r['packet'].get('lines', [])]
        for r in records: assert not r['packet'].get('transport_errors'), r['packet']
    else:
        assert local, 'receipt JSON required'; lines = text.splitlines()
    blobs = [l.split(' ', 1)[1] for l in lines if l.startswith('SPLIT_RESULT_BIN85 ')]
    assert len(blobs) == 1, 'missing or ambiguous split result'
    raw = decode(blobs[0])
    assert raw['complete'] is True and raw['armset'] == name
    assert list(raw['seeds']) == manifest['seeds'] and list(raw['weight_seeds']) == manifest['weight_seeds']
    assert raw['identities_sha256'] == manifest['identities_sha256']
    assert raw['telemetry_error'] is None, raw['telemetry_error']
    assert set(raw['round_units32']) == set(LABELS)
    for rows in raw['round_units32'].values():
        assert len(rows) == 8 and all(len(r) == len(raw['arms']) and all(len(v) == 4 and min(v) * 32 > raw['fixed_ns'] for v in r) for r in rows)
    full = [a for a in ('BASE', 'AA', 'PF') if a in raw['arms']]
    for when in ('before', 'after'):
        assert raw['checks'][when] == {a: 32 for a in full}, raw['checks']
    assert len(raw['work_counts']) == 16
    return raw, receipt


def samples(raw):
    return {label: {arm: [(st.mean(row[i]) * 32 - raw['fixed_ns']) / 1000 for row in rows] for i, arm in enumerate(raw['arms'])}
            for label, rows in raw['round_units32'].items()}


def six_load(gain16, gain32):
    return (WALL[128] / (WALL[128] - gain16) * WALL[256] / (WALL[256] - gain32)) ** (1 / 6)


def summarize(raws):
    report = dict(method='95% t7 over eight seed means of all four retained rounds; paired contrasts; '
                  'six-load R removes the contrast from verify 16 / 32 on S19 walls 321.09 / 425.66 us (transfer scenario, not a G0 verdict)',
                  receipts={}, arms={}, contrasts={}, per_M={}, six_load={}, reuse_tiling={}, histograms={}, resources={}, checks={})
    counts = None
    for name, (raw, receipt) in raws.items():
        report['receipts'][name] = receipt['id'] if receipt else 'LOCAL'
        s = samples(raw); table = CONTRASTS_J if name == 'J' else CONTRASTS_I
        report['arms'][name] = {l: {a: mean_ci(x) for a, x in arms.items()} for l, arms in s.items()}
        report['contrasts'][name] = {l: {k: mean_ci([a - b for a, b in zip(arms[hi], arms[lo])]) for k, (hi, lo) in table.items()} for l, arms in s.items()}
        per_m = {}
        for m in (128, 256):
            ph = [s[f'{m}:w13'], s[f'{m}:w2']]
            per_m[str(m)] = {k: mean_ci([sum(p[hi][i] - p[lo][i] for p in ph) for i in range(8)]) for k, (hi, lo) in table.items()}
        report['per_M'][name] = per_m
        six = {}
        for k, (hi, lo) in table.items():
            g = {m: [sum(s[f'{m}:{p}'][hi][i] - s[f'{m}:{p}'][lo][i] for p in ('w13', 'w2')) for i in range(8)] for m in (128, 256)}
            rr = [six_load(g[128][i], g[256][i]) for i in range(8)]
            six[k] = interval(rr)
        report['six_load'][name] = six
        if name == 'J':
            report['ratios_vs_BASE'] = {l: {a: interval([b / x for b, x in zip(arms['BASE'], arms[a])], [b / x for b, x in zip(arms['BASE'], arms['AA'])]) for a in ('PF', 'AA')} for l, arms in s.items()}
            report['histograms']['HISTN'] = raw['histograms']
        report['resources'][name] = raw['resources']
        report['checks'][name] = raw['checks']
        report.setdefault('telemetry', {})[name] = raw['telemetry_min_sm_mem_max_temp_W']
        report.setdefault('elapsed', {})[name] = dict(compile_s=raw['compile_s'], elapsed_s=raw['elapsed_s'])
        if counts is None: counts = {(si, m): c for si, m, c in raw['work_counts']}
        else: assert counts == {(si, m): c for si, m, c in raw['work_counts']}, 'J/I fixtures differ'
    # Reuse failure from the actual tiling: a heavy (count>16) expert's weights are read once per
    # 64-row chunk by the one CTA owning each (expert, 128-channel) tile; medium (9-16) and small
    # (<=8) experts are read once. Duplicated bytes / 4.8 TB/s bounds the HBM cost of rereads.
    for m in (128, 256):
        per = []
        for si in range(8):
            c = counts[(si, m)]; dup = sum((math.ceil(n / 64) - 1) for n in c if n > 16)
            per.append(dict(heavy=[n for n in c if n > 16], dup_reads=dup, dup_MB=dup * (EB['w13'] + EB['w2']) / 1e6,
                            dup_us_4p8=dup * (EB['w13'] + EB['w2']) / 4.8e6, CTAs_reading_each_byte=1,
                            reads_per_heavy_expert=[math.ceil(n / 64) for n in c if n > 16]))
        report['reuse_tiling'][str(m)] = dict(seeds=per, mean_dup_us=st.mean(p['dup_us_4p8'] for p in per))
    return report


def markdown(report):
    out = ['# S22 large-M load-path split', '', report['method'], '', 'Receipts: ' + ', '.join(f'{k} `{v}`' for k, v in report['receipts'].items()), '']
    for name in report['per_M']:
        out += [f'## {name}: W13+W2 per M (us [95% CI]) and six-load R', '', '| contrast | M128 | M256 | six-load R [CI] |', '|---|---:|---:|---:|']
        for k in report['per_M'][name]['128']:
            a, b, r = report['per_M'][name]['128'][k], report['per_M'][name]['256'][k], report['six_load'][name][k]
            out.append(f"| {k} | {a['us']:.1f} [{a['CI95'][0]:.1f},{a['CI95'][1]:.1f}] | {b['us']:.1f} [{b['CI95'][0]:.1f},{b['CI95'][1]:.1f}] | {r['R']:.4f} [{r['CI95'][0]:.4f},{r['CI95'][1]:.4f}] |")
        out += ['', '| phase | ' + ' | '.join(report['arms'][name]['128:w13']) + ' |', '|---|' + '---:|' * len(report['arms'][name]['128:w13'])]
        for l, arms in report['arms'][name].items():
            out.append(f'| {l} | ' + ' | '.join(f"{v['us']:.1f}" for v in arms.values()) + ' |')
        out.append('')
    out += ['## Reuse from the tiling', '']
    for m, r in report['reuse_tiling'].items():
        out.append(f"M{m}: mean duplicated-byte time {r['mean_dup_us']:.2f} us at 4.8 TB/s (W13+W2); each weight byte is read by one CTA, ceil(count/64) times for heavy experts.")
    if 'ratios_vs_BASE' in report:
        out += ['', '## PF (full math) vs BASE, phase ratio BASE/PF', '']
        for l, v in report['ratios_vs_BASE'].items():
            out.append(f"{l}: PF {v['PF']['R']:.4f} [{v['PF']['CI95'][0]:.4f},{v['PF']['CI95'][1]:.4f}], A:A {v['AA']['R']:.4f}")
    return '\n'.join(out) + '\n'


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('J', type=Path); ap.add_argument('I', type=Path, nargs='?'); ap.add_argument('--local', action='store_true')
    a = ap.parse_args()
    raws = {'J': load(a.J, 'J', a.local)}
    if a.I: raws['I'] = load(a.I, 'I', a.local)
    report = summarize(raws)
    dest = HERE / ('out/local_read' if a.local else 'result'); dest.mkdir(parents=True, exist_ok=True)
    (dest / 'raw.json').write_text(json.dumps({k: v[0] for k, v in raws.items()}, indent=1) + '\n')
    (dest / 'summary.json').write_text(json.dumps(report, indent=1) + '\n')
    (dest / 'split.md').write_text(markdown(report)); print((dest / 'split.md').read_text())
