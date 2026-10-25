"""Report exact sent S13 arms and enforce the per-piece, per-load A:A selection gate."""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path

H = Path(__file__).resolve().parent
R = H.parents[3]
LOADS = ('decode 1', 'decode 16', 'decode 32', 'verify 1', 'verify 16', 'verify 32')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('job')
    ap.add_argument('--manifest', type=Path, default=H / 's13ab_manifest.json')
    a = ap.parse_args()
    paths = list((R / 'runs/muse_decode_layer').glob('*_' + a.job + '*.json'))
    assert len(paths) == 1, paths
    receipt = json.loads(paths[0].read_text())
    manifest = json.loads(a.manifest.read_text())
    assert hashlib.sha256(paths[0].with_suffix('.py').read_bytes()).hexdigest() == manifest['sha256']
    log = receipt['result']['log_tail']
    loads = {load: {} for load in LOADS}
    for load, arm, ratio in re.findall(r'^LOAD ((?:decode|verify) \d+) (\w+) r=([\d.]+)$', log, re.M):
        loads[load][arm] = float(ratio)
    aa = {load: dict(mean=float(mean), lo=float(lo), hi=float(hi), se=float(se), n=int(n))
          for load, mean, lo, hi, se, n in re.findall(
              r'^AA ((?:decode|verify) \d+) mean=([\d.]+) lo=([\d.]+) hi=([\d.]+) se=([\d.]+) n=(\d+)$', log, re.M)}
    pairs = {load: {} for load in LOADS}
    for load, arm, mean, se, norm, normse, n in re.findall(
            r'^PAIR ((?:decode|verify) \d+) (\w+) mean=([\d.]+) se=([\d.]+) norm=([\d.]+) normse=([\d.]+) n=(\d+)$', log, re.M):
        pairs[load][arm] = dict(mean=float(mean), se=float(se), norm=float(norm), normse=float(normse), n=int(n))
    failures = [line for line in log.splitlines() if any(x in line for x in (' FAIL', 'INCOMPLETE', 'LAB_ERROR', 'ERROR '))]
    shapes = re.findall(r'^L (\w+) (\w+) r=[\d.]+ se=[\d.]+ d=[+\-\d.]+ A=[\d.]+ n=(\d+)', log, re.M)
    complete = bool(receipt['result'].get('passed') and 'END shapes=12/12' in log and not failures
                    and len(aa) == 6 and all(row['n'] == 16 for row in aa.values())
                    and shapes and all(int(n) == 16 for _, _, n in shapes))
    decisions = {}
    controls = {}
    active = set(manifest['arms']['S12']['active_tokens'])
    for load in LOADS:
        t = int(load.split()[1]) * (8 if load.startswith('verify') else 1)
        noise = aa.get(load)
        radius = max(abs(noise['lo'] - 1), abs(noise['hi'] - 1)) if noise else math.inf
        for arm, spec in manifest['arms'].items():
            if t not in spec['active_tokens']:continue
            p = pairs[load].get(arm)
            ratio = p['mean'] if p else loads[load].get(arm)
            decisions.setdefault(arm, {})[load] = dict(ratio=ratio, noise_radius=radius,
                wins=bool(ratio is not None and ratio > 1 + radius), paired=p)
        if t not in active:
            p = pairs[load].get('S12')
            ratio = p['mean'] if p else loads[load].get('S12')
            controls[load] = dict(ratio=ratio, noise_radius=radius,
                within_AA=bool(ratio is not None and abs(ratio-1) <= radius))
    aa12 = {load: dict(mean=float(mean), lo=float(lo), hi=float(hi), se=float(se), n=int(n))
            for load, mean, lo, hi, se, n in re.findall(
                r'^AA12 ((?:decode|verify) \d+) mean=([\d.]+) lo=([\d.]+) hi=([\d.]+) se=([\d.]+) n=(\d+)$', log, re.M)}
    subpairs = {load: {} for load in LOADS}
    for load, arm, mean, se, norm, n in re.findall(
            r'^SUBPAIR ((?:decode|verify) \d+) (\w+) mean=([\d.]+) se=([\d.]+) norm=([\d.]+) n=(\d+)$', log, re.M):
        radius=max(abs(aa12[load]['lo']-1),abs(aa12[load]['hi']-1))
        subpairs[load][arm]=dict(mean=float(mean),se=float(se),norm=float(norm),n=int(n),noise_radius=radius,
            wins=float(mean)>1+radius,within_AA=abs(float(mean)-1)<=radius)
    s12_gate = None
    if aa12:
        s12_gate = (len(aa12)==6 and all(r['n']==16 for r in aa12.values())
                    and subpairs['decode 1'].get('S12_nocode',{}).get('wins',False)
                    and all(subpairs[load].get('S12_noF4',{}).get('wins',False) for load in ('verify 16','verify 32'))
                    and all(subpairs[load].get('S11',{}).get('within_AA',False) for load in ('decode 16','decode 32','verify 1')))
    retained = [p['name'] for p in manifest['patches']]
    all_pieces_win = all(arm in decisions and all(r['wins'] for r in decisions[arm].values()) for arm in retained)
    all_changed_win = 'S12' in decisions and all(r['wins'] for r in decisions['S12'].values())
    report = dict(job=receipt['id'], source_sha256=manifest['host_sha256'], complete=complete,
        loads=loads, A_A=aa, decisions=decisions, controls=controls, failures=failures,
        S12_A_A=aa12, S12_subpairs=subpairs, S12_recheck_gate=s12_gate,
        freeze_gate=bool(complete and all_pieces_win and all_changed_win and all(c['within_AA'] for c in controls.values()) and s12_gate is not False))
    if all('S12' in row for row in loads.values()):
        report['six_load_S12_over_S13'] = math.exp(sum(math.log(row['S12']) for row in loads.values()) / 6)
    out = H / (receipt['id'][:8] + '_report.json')
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
