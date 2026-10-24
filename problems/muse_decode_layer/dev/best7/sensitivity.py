"""Derive marginal score sensitivities from the latest H200 median load times."""
from pathlib import Path
import argparse,json,math,statistics
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--receipt',type=Path,default=ROOT/'problems/muse_decode_layer/profiles/cand_r3_best6.1b80ffc037cc454c.json')
parser.add_argument('--update-focus',action='store_true')
args=parser.parse_args();receipt=args.receipt.resolve()
data=json.loads(receipt.read_text());loads={}
for row in data['benchmarks']:
    case=row['case'];load=case['load'];entry=loads.setdefault(load,dict(us=0.,baseline_us=0.,layers={}))
    entry['us']+=case['weight']*row['median_ns']/1e3
    entry['baseline_us']+=case['weight']*row['baseline_ns']/1e3
    entry['layers'][case['layer']]=row['median_ns']/1e3
score=math.prod(row['baseline_us']/row['us'] for row in loads.values())**(1/6)
assert abs(score-data['summary']['speedup'])<1e-8
v9_jobs=('0e94268cf5654012','7307be7878604e6d','1365af6696884c7e')
seed_loads={}
for job in v9_jobs:
    path=next((ROOT/'runs/muse_decode_layer').glob('*'+job+'.json'))
    receipt_result=json.loads(path.read_text())['result'];seed={}
    for row in receipt_result['records']:
        if row.get('event')!='benchmark':continue
        case=row['case'];entry=seed.setdefault(case['load'],dict(us=0.,baseline_us=0.))
        entry['us']+=case['weight']*row['median_ns']/1e3
        entry['baseline_us']+=case['weight']*row['baseline_ns']/1e3
    for row in seed.values():row['speedup']=row['baseline_us']/row['us']
    assert abs(math.prod(row['speedup'] for row in seed.values())**(1/6)-receipt_result['speedup'])<1e-8
    seed_loads[job]=seed
lines=['| Load | v9 mean x | best6 x | best6 load us | +1 us: sliding % | full % | both % | +10 us both % |',
       '|---|---:|---:|---:|---:|---:|---:|---:|']
for load,row in loads.items():
    duration=row['us']
    # Exact finite differences, changing only this load's contribution.
    gain=lambda saved:100.*((duration/(duration-saved))**(1/6)-1.)
    best6=row['baseline_us']/duration
    v9=statistics.mean(seed[load]['speedup'] for seed in seed_loads.values())
    row.update(sliding_1us_pct=gain(46),full_1us_pct=gain(16),both_1us_pct=gain(62),both_10us_pct=gain(620),
               v9_mean_speedup=v9,best6_speedup=best6,optimistic_combined_speedup=max(v9,best6))
    lines.append(f'| {load} | {v9:.3f} | {best6:.3f} | {duration:.1f} | {gain(46):.4f} | {gain(16):.4f} | {gain(62):.4f} | {gain(620):.3f} |')
result=dict(receipt=str(receipt.relative_to(ROOT)),score=score,loads=loads,
            formula='relative score factor = product_l[T_l/(T_l-46*dS_l-16*dF_l)]**(1/6)',
            limit='critical-path savings, not sums of overlapping GPU profile durations; one receipt router is not a seed mean',
            v9_seed_loads=seed_loads,
            optimistic_combined_score=math.prod(row['optimistic_combined_speedup'] for row in loads.values())**(1/6),
            combination_limit='per-load better of three-seed v9 mean and one best6 check; stage composition and scale conversion prevent treating it as a candidate prediction')
(HERE/'sensitivity.json').write_text(json.dumps(result,indent=2)+'\n')
(HERE/'SENSITIVITY.md').write_text('\n'.join(lines)+'\n')
if args.update_focus:
    focus=HERE.parents[1]/'FOCUS.md';source=focus.read_text()
    start=source.index('| Load | v9 mean x |');end=source.index('\n\n',start)
    source=source[:start]+'\n'.join(lines)+source[end:]
    assert len(source.splitlines())<=30
    focus.write_text(source)
print('\n'.join(lines))
