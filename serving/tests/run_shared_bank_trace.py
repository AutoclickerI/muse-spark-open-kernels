"""Diagnostic c16/c32 timelines; the uninstrumented brackets own admission."""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

from analyze_ttft import analyze

ROOT = Path(__file__).resolve().parents[2]
DATA = Path('/data2/jihoon/muse_kernel/harness')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tree', type=Path, required=True)
    parser.add_argument('--prefix', required=True)
    args = parser.parse_args()
    evidence = DATA / 'results' / f'{args.prefix}-evidence'
    evidence.mkdir(exist_ok=True)
    labels = []
    # Full c1/c4 warmup precedes the two diagnostic loads, as in the gate.
    for role in ('stock-a', 'patch', 'stock-b'):
        label = f'{args.prefix}-{role}-tp2'
        labels.append(label)
        result = DATA / 'results' / label
        if result.exists():
            meta = json.loads((result / 'meta.json').read_text())
            assert not meta.get('failure') and not meta['validation_errors']
            assert meta['provenance'] == meta['provenance_end']
        else:
            cmd = [sys.executable, 'serving/tests/ttft_harness.py', '--native',
                   '--trace-ttft', '--tree', 'stock' if role.startswith('stock') else str(args.tree),
                   '--weights', 'margin', '--label', label, '--tp', '2', '--gpus', '0,1',
                   '--levels', '1,4,16,32', '--convs', '1:2,4:8,16:16,32:32', '--turns', '3']
            print('TRACE', label, flush=True)
            with (DATA / 'logs' / f'{label}.log').open('w') as log:
                subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        records = [json.loads(s) for s in (result / 'requests.jsonl').read_text().splitlines()]
        assert len(records) == 174 and all(r['n_out'] == len(r['output_ids']) == 220 for r in records)
    for level in (16, 32):
        results = [analyze(label, level) for label in labels]
        (evidence / f'c{level}.json').write_text(json.dumps(results, indent=2) + '\n')
        with (evidence / f'c{level}_per_request.csv').open('w') as stream:
            rows = [dict(label=item['label'], **row) for item in results for row in item['per_request']]
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print('MEAN_MS', level, json.dumps({r['label']: {
            k: 1000 * r['summary_means']['None'][k] for k in
            ('queue_s', 'prefill_execution_s', 'first_emit_s', 'ttft')}
            for r in results}), flush=True)
    for stock in (labels[0], labels[2]):
        subprocess.run([sys.executable, 'serving/harness/compare.py', stock, labels[1],
                        '--require-exact', '--json', str(evidence / f'{stock}_agreement.json')],
                       cwd=ROOT, check=True, stdout=subprocess.DEVNULL)


if __name__ == '__main__':
    main()
