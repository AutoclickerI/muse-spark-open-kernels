#!/usr/bin/env python3
"""Decode-layer score against stage floors (docs/reference/bound.md "Score vs floor").

Reads current.json (build.py's per-shape stage model of the latest landed S-candidate profile), adds the
paired-lab stage wins that profile does not contain yet, and prints the six-load score:
  - the model and the model plus those pending wins ("pending");
  - each stage's time as a multiple of its one-read floor, per shape;
  - per lever: that lever's stages capped at 1.5x / 1.3x / 1.15x their floor (independent; they don't add);
  - every stage capped at 2.0x / 1.5x / 1.3x / 1.15x its floor (the judge's stream roof is above 3.61 TB/s: our sweep
    fits 2.36 us + bytes / 4.58 TB/s);
  - the projections with 25% fewer dense bytes (lossless BF16 exponent coding) at a fixed 0 / 2 / 4 us per call;
  - each component sped up by a leader-match factor (--match);
  - each decode-1 / verify-1 layer's time above 2.36 us + bytes / 4.58 TB/s per data stage.

    python3 problems/muse_decode_layer/dev/levers/scenarios.py [current.json] [--match attention=1.8,...]

A scenario changes stage times; the score is base * prod_load(L / (L - sum w * (t_old - t_new)))^(1/6), with
L the model's adjusted load time and w the layer count (46 sliding, 16 full). A cap never slows a stage.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

LAYERS = {'sliding': 46, 'full': 16}
ORDER = ('reset', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S7R', 'S8', 'S9', 'S10', 'S11', 'S12')
PROJ = ('qkv_gate', 'o', 'pool_router', 'unpooler')
GLUE = ('boundaries', 'route')
# Leaders' component speed over ours, 2026-10-24 top 3 (attention 2.40-2.52 vs 1.40; MoE discounted for our
# newer in-layer kernels; projections 1.43-1.55 vs 1.18); glue /3 is an assumption, no board measures it.
MATCH = {'attention': 1.8, 'moe': 1.2, 'proj': 1.25, 'glue': 3.0}

LEVERS = {
    'small-M MoE (M<=32)': lambda r, k: k == 'moe' and r['tokens'] <= 32,
    'MoE M=128': lambda r, k: k == 'moe' and r['tokens'] == 128,
    'MoE M=256': lambda r, k: k == 'moe' and r['tokens'] == 256,
    'attention': lambda r, k: k == 'attention',
    '  sliding attention': lambda r, k: k == 'attention' and r['case']['layer'] == 'sliding',
    '  full attention, verify 16/32': lambda r, k: k == 'attention' and r['case']['layer'] == 'full' and r['tokens'] >= 128,
    'projections': lambda r, k: k in PROJ,
    'glue (boundaries, route)': lambda r, k: k in GLUE,
}
GROUPS = {'attention': ('attention',), 'moe': ('moe',), 'proj': PROJ, 'glue': GLUE}


def pending(mode: str) -> list[tuple]:
    """Paired-lab stage wins (tokens, stage, new-time function, source[, layer]) not yet in a `mode` profile."""
    i = ORDER.index(mode)
    out = []
    if i < ORDER.index('S4'):  # one-LOP3 NT1 vs recoded NT1, dc3fea0a
        for t, d in {1: 0.4, 8: 0.7, 16: 2.0, 32: 3.6}.items():
            out.append((t, 'moe', lambda us, d=d: us - d, 'LOP3 NT1 dc3fea0a'))
    if i < ORDER.index('S5'):  # H_H 234.44 us vs NT1-recoded 289.62 / D32g 272.70 (S4), 770809cf
        basis = 289.62 if i < ORDER.index('S4') else 272.70
        out.append((128, 'moe', lambda us, b=basis: us * 234.44 / b, 'H_H 770809cf'))
    if i < ORDER.index('S6'):  # mixed_parallel 344.5 us vs raw CTA1g2112 388.0 / recoded C1g2112 366.2 (S3+), 8483c0c6
        basis = 388.0 if i < ORDER.index('S3') else 366.2
        out.append((256, 'moe', lambda us, b=basis: us * 344.5 / b, 'mixed_parallel 8483c0c6'))
        out.append((1, 'o', lambda us: us - 0.35, 'O config6 c42e9fcc'))
    if i < ORDER.index('S7R'):  # R_full 0.9227x mixed_parallel's M=256 MoE (paired, zero_()-flushed), 110599ba
        out.append((256, 'moe', lambda us: us * 0.9227, 'R_full 110599ba'))
    if i < ORDER.index('S8'):  # P7c: cluster glue 1365a290 (b1+b2, route per layer), attention early release 4d995f13
        # (decode 1 attention -> O chain), verify-dense split-K qkv at M=128 b15a3bde (23.26 -> 20.42 us)
        for t, b, r in ((1, 2.43, 0.26), (8, 1.57, 0.38), (16, 2.85, 0.37), (32, 1.14, 1.41)):
            out.append((t, 'boundaries', lambda us, d=b: us - d, 'cluster glue 1365a290'))
            out.append((t, 'route', lambda us, d=r: us - d, 'cluster glue 1365a290'))
        out.append((1, 'attention', lambda us: us - 0.95, 'early release 4d995f13', 'sliding'))
        out.append((1, 'attention', lambda us: us - 2.06, 'early release 4d995f13', 'full'))
        out.append((128, 'qkv_gate', lambda us: us - 2.84, 'vdsk qkv b15a3bde'))
    if i < ORDER.index('S9'):  # BF16 qkv front 87cd3a84 (attention input: one PDL sum+cast instead of torch's two kernels);
        # small-M flags 95 / 69, router+MoE, mean of cf1de160 and eb9e0382; N_full vs H_H 0.8842 (c01b1916) and vs R_full
        # 0.8701 / 0.9230 (14f942d5)
        out.append((1, 'attention', lambda us: us - 4.21, 'BF16 qkv front 87cd3a84', 'sliding'))
        out.append((1, 'attention', lambda us: us - 4.46, 'BF16 qkv front 87cd3a84', 'full'))
        out.append((16, 'attention', lambda us: us - 1.52, 'BF16 qkv front 87cd3a84', 'sliding'))
        out.append((16, 'attention', lambda us: us - 5.25, 'BF16 qkv front 87cd3a84', 'full'))
        for t, d in {1: 5.63, 8: 3.99, 16: 4.67, 32: 4.57}.items():
            out.append((t, 'moe', lambda us, d=d: us - d, 'small-M flags cf1de160+eb9e0382'))
        out.append((128, 'moe', lambda us: us * 0.8842, 'N_full M128 c01b1916'))
        out.append((256, 'moe', lambda us: us * 0.8701 / 0.9230, 'N_full M256 14f942d5'))
    if i < ORDER.index('S10'):  # N_full stream swap (heavy N32 on the side stream, launched first), 880c3881
        out.append((128, 'moe', lambda us: us * 0.9799, 'N_swap 880c3881'))
        out.append((256, 'moe', lambda us: us * 0.9882, 'N_swap 880c3881'))
    return out


def score(model: dict, rows: list[dict]) -> float:
    total = 0.0
    for load, L in model['loads_us'].items():
        saved = sum(LAYERS[r['case']['layer']] * (sum(r0['stages_us'].values()) - sum(r['stages_us'].values()))
                    for r0, r in zip(model['rows'], rows) if r['case']['load'] == load)
        total += math.log(L / (L - saved))
    return model['modeled_speedup'] * math.exp(total / len(model['loads_us']))


def edit(rows: list[dict], sel, fn) -> list[dict]:
    out = copy.deepcopy(rows)
    for r in out:
        for k, v in r['stages_us'].items():
            if sel(r, k) and r['floors_us'].get(k) is not None:
                r['stages_us'][k] = fn(v, r['floors_us'][k])
    return out


def cap(rows, sel, f):
    return edit(rows, sel, lambda v, floor: min(v, f * floor))


def label(r: dict) -> str:
    return f"{r['case']['load']} {r['case']['layer'][0].upper()}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('model', nargs='?', type=Path, default=Path(__file__).with_name('current.json'))
    ap.add_argument('--match', default=','.join(f'{k}={v}' for k, v in MATCH.items()),
                    help='leader-match speedup per component (attention, moe, proj, glue)')
    a = ap.parse_args()
    model = json.loads(a.model.read_text())
    mode = model['mode']
    rows = copy.deepcopy(model['rows'])
    deltas = pending(mode)
    for r in rows:
        for t, k, fn, _, *layer in deltas:
            if r['tokens'] == t and (not layer or r['case']['layer'] == layer[0]):
                r['stages_us'][k] = fn(r['stages_us'][k])
    base, est = model['modeled_speedup'], score(model, rows)
    print(f"model {mode} `{model['job']}`: measured {model['baseline_source_speedup']:.3f}, model {base:.3f}; "
          f"with pending wins {est:.3f} ({(est / base - 1) * 100:+.1f}%)")
    print('  pending: ' + (', '.join(sorted({d[3] for d in deltas})) or 'none'))

    stages = [k for k in rows[0]['stages_us'] if rows[0]['floors_us'].get(k) is not None]
    ratio = [k for k in stages if k not in GLUE]
    print('\nstage time / one-read floor (x), with pending wins; glue in us (its fused-IO floor is ~0.02 us)')
    print(f"  {'shape':11s} " + ' '.join(f'{k[:8]:>8s}' for k in ratio) + ' ' + ' '.join(f'{k[:8] + " us":>11s}' for k in GLUE))
    for r in rows:
        print(f'  {label(r):11s} ' + ' '.join(f"{r['stages_us'][k] / r['floors_us'][k]:8.1f}" for k in ratio)
              + ' ' + ' '.join(f"{r['stages_us'][k]:11.1f}" for k in GLUE))

    print('\nper lever, capped at 1.5x / 1.3x / 1.15x its floor (independent counterfactuals; they do not add)')
    for name, sel in LEVERS.items():
        values = [score(model, cap(rows, sel, target)) for target in (1.5, 1.3, 1.15)]
        print(f'  {name:32s} ' + '  '.join(f'{v:.3f} ({(v/est-1)*100:+5.1f}%)' for v in values))
    every = lambda r, k: True
    print('\nevery stage capped at ' + ', '.join(f'{f}x floor: {score(model, cap(rows, every, f)):.3f}' for f in (2.0, 1.5, 1.3, 1.15)))

    # Lossless BF16 exponent coding of the dense weights (P10): 25% fewer dense bytes; a projection's time beyond a
    # fixed part (launch, tail, activation IO) scales with its bytes.
    proj = lambda r, k: k in PROJ
    print('\nadditional dense bytes -25% from current encoding (projection time = fixed + (t - fixed) x 0.75): ' + ', '.join(
        f"fixed {f:g} us {(v := score(model, edit(rows, proj, lambda t, _, f=f: min(t, f + (t - f) * 0.75)))):.3f} ({(v / est - 1) * 100:+.1f}%)"
        for f in (0.0, 2.0, 4.0)))

    match = {k: float(v) for k, v in (x.split('=') for x in a.match.split(','))}
    print('\nleader match (stage time / factor): ' + ', '.join(f'{k} /{v:g}' for k, v in match.items()))
    combined = rows
    for k, f in match.items():
        sel = lambda r, s, g=GROUPS[k]: s in g
        print(f'  {k:10s} {score(model, edit(rows, sel, lambda v, _, f=f: v / f)):.3f}')
        combined = edit(combined, sel, lambda v, _, f=f: v / f)
    print(f"  {'all':10s} {score(model, combined):.3f}")

    print('\nlayer time above a stream at 2.36 us + bytes / 4.58 TB/s per data stage: adjusted graph and the sum of '
          'per-stage excess (kernel durations include PDL waits)')
    for r0, r in zip(model['rows'], rows):
        if r['tokens'] in (1, 8):
            stream = sum(r['stream_roof_us'].get(k, r['floors_us'][k]) for k in stages)
            graph = r0['modeled_adjusted_us'] - sum(r0['stages_us'].values()) + sum(r['stages_us'].values())
            excess = sum(max(0.0, r['stages_us'][k] - r['stream_roof_us'].get(k, r['floors_us'][k])) for k in stages)
            print(f'  {label(r):11s} graph {graph:6.1f} us, one-read {stream:5.1f}, above {graph - stream:5.1f}; '
                  f'stage-sum excess {excess:5.1f} us')


if __name__ == '__main__':
    main()
