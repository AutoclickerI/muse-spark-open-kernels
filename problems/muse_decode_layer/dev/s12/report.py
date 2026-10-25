"""Summarize a landed S12 leave-one-out receipt without mistaking unchanged loads for gains."""
import argparse
import json
import math
import re
from pathlib import Path

R = Path(__file__).resolve().parents[4]
H = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('job')
    ap.add_argument('--manifest', type=Path, default=H / 's12ab_manifest.json')
    args = ap.parse_args()
    paths = list((R / 'runs/muse_decode_layer').glob('*_' + args.job + '*.json'))
    assert len(paths) == 1, paths
    receipt = json.loads(paths[0].read_text())
    manifest = json.loads(args.manifest.read_text())
    log = receipt['result']['log_tail']
    shapes = re.findall(r'^L (\w+) (\w+) r=([\d.]+) se=([\d.]+) d=([+\-\d.]+) A=([\d.]+) n=(\d+)(.*)$', log, re.M)
    loads = {load: {} for load in ('decode 1', 'decode 16', 'decode 32', 'verify 1', 'verify 16', 'verify 32')}
    for load, arm, ratio in re.findall(r'^LOAD ((?:decode|verify) \d+) (\w+) r=([\d.]+)$', log, re.M):
        loads[load][arm] = float(ratio)
    failures = [line for line in log.splitlines() if any(x in line for x in (' FAIL', 'INCOMPLETE', 'LAB_ERROR', 'ERROR '))]
    short = [f'{shape}/{arm}: n={n}' for shape, arm, _, _, _, _, n, _ in shapes if int(n) != 16]
    complete = receipt['result'].get('passed') and 'END shapes=12/12' in log and not failures and not short
    active = set(manifest['arms']['S11']['active_tokens'])
    unchanged = [load for load in loads if int(load.split()[1]) * (8 if load.startswith('verify') else 1) not in active]
    all_win = complete and all(loads[load].get('S11', 0) > 1 for load in loads) and not unchanged
    piece_losses = {load: {arm: ratio for arm, ratio in values.items() if arm not in ('S11', 'A2') and ratio <= 1}
                    for load, values in loads.items()}
    piece_losses = {load: values for load, values in piece_losses.items() if values}
    report = dict(job=receipt['id'], source_sha256=manifest['host_sha256'], complete=bool(complete),
        loads=loads, unchanged_control_loads=unchanged, failures=failures, short_seeds=short,
        piece_losses=piece_losses, strict_freeze_gate=bool(all_win and not piece_losses))
    if all('S11' in row for row in loads.values()):
        report['six_load_S11_over_S12'] = math.exp(sum(math.log(row['S11']) for row in loads.values()) / 6)
    (H / (receipt['id'][:8] + '_report.json')).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
