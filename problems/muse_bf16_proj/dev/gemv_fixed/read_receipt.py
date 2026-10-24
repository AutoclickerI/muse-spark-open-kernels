"""Read a landed local receipt once; report decomposition and integration cells."""
import argparse
import hashlib
import json
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
parser=argparse.ArgumentParser();parser.add_argument('receipt');args=parser.parse_args()
p=Path(args.receipt)
if not p.exists():
    hits=list(ROOT.glob('runs/*/*_'+args.receipt+'.json'))
    if not hits:raise SystemExit('No local receipt for '+args.receipt)
    p=hits[0]
receipt=json.loads(p.read_text());tail=receipt['result'].get('log_tail','')
block=re.search(r'^LAB_BEGIN dense_gemv_fixed (\w+)\n(.*?)^LAB_END.*?$',tail,re.M|re.S)
if not block:raise SystemExit('No dense_gemv_fixed section in '+str(p))
manifest=json.loads((HERE/'manifest.json').read_text())
if block[1]!=manifest['payload_sha256'][:16]:raise SystemExit('Receipt does not match current payload manifest')
if 'LAB_ERROR' in block[2] or not re.search(r'^END roles=16 segments=4 ',block[2],re.M):
    raise SystemExit('Incomplete lab:\n'+block[2])
pattern=(r'^R (\d+)([QOPU]) B=([\d.]+)/([\d.]+) N(\d+)=([\d.]+)/([\d.]+) r=([\d.]+) '
         r'C=([\d,-]+) L/W/T=([\d.-]+)/([\d.-]+)/([\d.-]+) B:(\S+) R=([\d.-]+)/([\d.-]+):([\d.-]+) s=([\d.,]+) ')
roles=[];cells=[]
role_names=dict(Q='qkv_gate',O='o',P='pool_router',U='unpooler')
for match in re.finditer(pattern,block[2],re.M):
    v=match.groups();t=int(v[0]);role=role_names[v[1]]
    if role=='unpooler' and t<=8:role='unpooler_slice'
    c=list(map(int,v[8].split(',')));seeds=list(map(float,v[16].split(',')))
    row=dict(tokens=t,role=role,config=int(v[4]),baseline_producer_us=float(v[2]),baseline_with_consumer_us=float(v[3]),
        producer_us=float(v[5]),with_consumer_us=float(v[6]),paired_ratio=float(v[7]),seed_ratios=seeds,
        observer_cycles=dict(zip(('entry_first_issue','pdl_wait','mainloop','local_k_reduce'),c)),
        dispatch_first_us=float(v[9]),last_wave_us=float(v[10]),tail_p95_us=float(v[11]),
        baseline_diagnostic=v[12],consumer_partial_load_cycles=float(v[13]),
        consumer_split_us=float(v[14]),consumer_one_us=float(v[15]))
    roles.append(row)
    # Candidate for a live-layer check, not a dispatch promotion. A gain must
    # exceed the judge's stated minimum1.5% interval and persist on every seed.
    if row['paired_ratio']<.985 and max(seeds)<1.:
        choice=manifest['choices'][str(t)+':'+role]
        cells.append(dict(tokens=t,role=role,config=row['config'],
            fp32=role=='pool_router' or choice=='custom',paired_ratio=row['paired_ratio'],
            seed_ratios=seeds,requires_live_layer_agreement=True))
if len(roles)!=16:raise SystemExit(f'Only {len(roles)} of16 role rows survived the receipt')
projection=HERE/'projection.py';sha=hashlib.sha256(projection.read_bytes()).hexdigest()
if sha!=manifest['projection_sha256']:raise SystemExit('Projection source changed since the measurement')
report=dict(receipt=str(p),payload_sha256=manifest['payload_sha256'],
    projection=str(projection.relative_to(ROOT)),projection_sha256=sha,
    roles=roles,segments=re.findall(r'^SEG .*$',block[2],re.M),integration_cells=cells,
    integration="Construct Best7ProjectionCell outside capture; insert into state['best7_stream_ops'][(tokens, role, config, 1)] and set that plan's projection[role]=('stream', config, 1). Validate live whole-layer routing before dispatch.")
(HERE/'h200_result.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
