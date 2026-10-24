"""Read a landed local receipt and propose best7 cells from paired layer gains."""
import argparse
import hashlib
import json
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('receipt');args=parser.parse_args()
p=Path(args.receipt)
if not p.exists():
    hits=list((ROOT/'runs').glob(f'*/*_{args.receipt}.json'))
    if len(hits)!=1:raise SystemExit(f'{len(hits)} local receipts; no judge request made')
    p=hits[0]
data=json.loads(p.read_text());tail=data['result'].get('log_tail','')
section=re.search(r'^LAB_BEGIN l2_overlap_lab ([0-9a-f]+)\n(.*?)(?=^LAB_END)',tail,re.M|re.S)
if section is None:raise SystemExit('l2_overlap_lab section missing')
payload=HERE.parent/'l2_overlap_lab.py';sha=hashlib.sha256(payload.read_bytes()).hexdigest()
if not sha.startswith(section[1]):raise SystemExit('payload SHA differs; use the archived source')
if 'P7_END cases=6/6' not in section[2] or 'LAB_ERROR' in section[2]:raise SystemExit('incomplete lab; no settings emitted')
rows=[];issues=[]
for line in section[2].splitlines():
    if line.startswith('P7_PICK '):
        v=dict(re.findall(r'(\w+)=([^ ]+)',line))
        for k in ('c','m','w','r','n','rej'):v[k]=int(v[k])
        for k in ('ratio','noise','fp32'):v[k]=float(v[k])
        a,b=map(float,v.pop('us').split('/'));v.update(a_us=a,b_us=b,saved_us=a-b)
        v['seeds']=list(map(float,v['seeds'].split(',')))
        v['clears_aa_noise']=v['ratio']<1. and a-b>abs(v['noise'])
        v['best7_key']=[v['m'],1,v['w']]
        v['projection_cell']={'l2_prefetch':v['r']} if v['clears_aa_noise'] else {}
        rows.append(v)
    elif any(k in line for k in ('REJECT','ERROR','INCOMPLETE','NO_PICK')):issues.append(line)
if len(rows)!=6:raise SystemExit('missing case picks')
out=dict(payload_sha256=sha,receipt=str(p.resolve().relative_to(ROOT)),rows=rows,issues=issues,
    note='Only paired whole-layer savings beyond the measured A/A mean delta propose a switch. Profile sums include PDL waits and are diagnostic. Re-screen the selected best7 composition before candidate handoff.')
(HERE/'handoff.json').write_text(json.dumps(out,indent=2)+'\n')
for r in rows:print(r)
for issue in issues:print('ISSUE',issue)
