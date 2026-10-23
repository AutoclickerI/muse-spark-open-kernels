#!/usr/bin/env python3
"""Four fresh servers: two independent self comparisons, then stock versus patch."""
import argparse
import json
import subprocess
import sys

import common
from compare import compare, load


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prefix', required=True)
    ap.add_argument('--tp', type=int, choices=(1, 2), default=1)
    ap.add_argument('--weights', choices=common.WEIGHT_SETS, default='margin',
                    help='margin fixture is the agreement default; random is a tie diagnostic')
    ap.add_argument('--quick', action='store_true', help='10 requests instead of 30; same context coverage')
    ap.add_argument('--load-check', action='store_true')
    ap.add_argument('--trace-paths', action='store_true')
    ap.add_argument('extra', nargs=argparse.REMAINDER)
    args = ap.parse_args()
    labels = [f'{args.prefix}-{suffix}' for suffix in ('stock-a', 'stock-b', 'stock-c', 'patch')]
    report_path = common.RESULTS / f'{args.prefix}-comparison.json'
    if report_path.exists() or any((common.RESULTS / label).exists() for label in labels):
        ap.error('prefix already used; old evidence is never overwritten')
    for tree, label in zip(('stock', 'stock', 'stock', 'patch'), labels):
        cmd = [sys.executable, str(common.HARNESS / 'run.py'), '--tree', tree,
               '--label', label, '--tp', str(args.tp), '--weights', args.weights]
        if args.quick:
            cmd += ['--convs', '1:1,4:4', '--turns', '2']
        if args.load_check:
            cmd += ['--load-check']
        if args.trace_paths:
            cmd += ['--trace-paths']
        cmd += args.extra
        subprocess.run(cmd, check=True)
    runs = [load(label) for label in labels]
    comparisons = {}
    for name, i, j in [('self_ab', 0, 1), ('self_ac', 0, 2),
                        ('patch_a', 0, 3), ('patch_b', 1, 3)]:
        comparisons[name] = compare(*runs[i], *runs[j])
    report_path.write_text(json.dumps(comparisons, indent=2))
    for name, result in comparisons.items():
        print(name, json.dumps({
            'per_level': result['per_level'],
            'agreement_metric': result['agreement_metric'],
        }))
    print(f'Full comparison: {report_path}')
    # A red self baseline must not make a broken patch look acceptable. This
    # local smoke gate is deliberately stricter than the judge's unspecified
    # "nearly as well" criterion; raw metrics are always preserved.
    sys.exit(int(any(r['problems'] or any(not row['exact'] for row in r['rows'])
                     for r in comparisons.values())))


if __name__ == '__main__':
    main()
