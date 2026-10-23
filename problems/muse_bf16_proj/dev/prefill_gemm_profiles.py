"""Separate observed profile aggregates from inferred per-op attribution."""
from collections import defaultdict
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'problems/muse_prefill_layer/profiles/promoted_6a9cd943fe404d74.965a9a816d4744ea.json'
v = json.loads(SOURCE.read_text())
shapes = dict(qkv_gate=(2304,8192), o=(8192,1024), pooler=(4096,8192),
              unpooler=(8192,4096), router=(512,8192))
groups = defaultdict(lambda: defaultdict(float))
weight = defaultdict(float)
for profile in v['profiles']:
    case = profile['case']; m = case['batch'] * case['q_len']; w = case['weight']
    weight[m] += w
    for line in profile['report'].splitlines():
        match = re.match(r'(nvjet\S+)\s+([\d.]+) µs\s+[\d.]+%\s+(\d+)', line)
        if match:
            name, us, count = match.groups()
            groups[m][name] += float(us) * w

out = dict(receipt='965a9a816d4744ea', peak_tflops=989,
           attribution='shape-to-tactic inference; reports aggregate names, individual calls unavailable', shapes=shapes, rows=[])
for m in (753,3012,16384):
    times = {k:t/weight[m] for k,t in groups[m].items()}
    def find(part): return next(t for k,t in times.items() if part in k)
    flop = {name:2*m*n*k/1e12 for name,(n,k) in shapes.items()}
    if m == 753:
        attribution = [('qkv_gate', ('qkv_gate',), find('112x128')),
                       ('o+pooler+unpooler', ('o','pooler','unpooler'), find('128x192')),
                       ('router', ('router',), find('64x96'))]
    elif m == 3012:
        attribution = [('qkv_gate', ('qkv_gate',), find('192x144')),
                       ('pooler', ('pooler',), find('128x256')),
                       ('o+unpooler', ('o','unpooler'), find('256x152')),
                       ('router', ('router',), find('128x104'))]
    else:
        attribution = [('qkv_gate', ('qkv_gate',), find('320x128')),
                       ('o+pooler+unpooler', ('o','pooler','unpooler'), find('192x192')),
                       ('router', ('router',), find('256x128'))]
    for name,roles,us in attribution:
        f = sum(flop[r] for r in roles)
        # If every peer cannot exceed 989, allocate all group's excess time
        # to this op to derive its slowest possible individual throughput.
        bounds = {}
        for role in roles:
            max_us = us - (f-flop[role])/989*1e6
            bounds[role] = [flop[role]/max_us*1e6, 989] if len(roles)>1 else [f/us*1e6]*2
        out['rows'].append(dict(m=m, roles=roles, group_us=us, group_tflops=f/us*1e6,
                                percent_peak=f/us*1e6/9.89, bounds_tflops=bounds))
out['first_turn'] = dict(weight=weight[16384], dense_us=sum(groups[16384].values()),
                        floor_us=sum(2*16384*n*k/989e12*1e6 for n,k in shapes.values())*weight[16384])
dest = ROOT / 'runs/screen/prefill_gemm/profile_arithmetic.json'
dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(out,indent=2)+'\n')
print('| GEMM | W[N,K] | M753 TF/s | M3012 TF/s | M16384 TF/s |')
print('|---|---|---:|---:|---:|')
for role,shape in shapes.items():
    values=[]
    for m in (753,3012,16384):
        row=next(r for r in out['rows'] if r['m']==m and role in r['roles'])
        lo,hi=row['bounds_tflops'][role]
        values.append(f'{lo:.0f}' if lo==hi else f'{lo:.0f}–{hi:.0f}*')
    print('| '+role+' | '+str(shape)+' | '+' | '.join(values)+' |')
print('first turn',out['first_turn'])
for r in out['rows']:
    print(r['m'],','.join(r['roles']),f"{r['group_us']:.2f}us {r['group_tflops']:.1f}TF/s {r['percent_peak']:.1f}% peak")
