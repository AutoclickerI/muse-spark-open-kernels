"""Make a SHA-pinned projection/glue handoff from a landed local receipt."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[3]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('receipt')
args=parser.parse_args()
receipt=Path(args.receipt)
if not receipt.exists():
    hits=list((REPO/'runs').glob(f'*/*_{args.receipt}.json'))
    if len(hits)!=1:raise SystemExit(f'expected one landed receipt, got {len(hits)}; no judge request made')
    receipt=hits[0]
data=json.loads(receipt.read_text());tail=data['result'].get('log_tail','')
section=re.search(r'^LAB_BEGIN streaming_glue_lab ([0-9a-f]+)\n(.*?)(?=^LAB_END)',tail,re.M|re.S)
if section is None:raise SystemExit('no streaming_glue_lab section; inspect the receipt error')
payload=HERE.parent/'streaming_glue_lab.py'
sha=hashlib.sha256(payload.read_bytes()).hexdigest()
if not sha.startswith(section[1]):raise SystemExit('payload SHA differs; read against the archived source')
if 'SG_DONE' not in section[2] or 'LAB_ERROR' in section[2]:raise SystemExit('payload did not finish cleanly')
stages=[];chains=[];issues=[]
for line in section[2].splitlines():
    if line.startswith('SG_PICK '):
        values=dict(re.findall(r'(\w+)=([^ ]+)',line))
        for k in ('m','cfg','split','tried','rejected'):values[k]=int(values[k])
        for k in ('cublas_us','us','ratio','separate_us','fuse_ratio','raw_TBs'):values[k]=float(values[k])
        values['role']=line.split()[1]
        values['paired_gain_vs_cublas']=values['ratio']<1.
        values['paired_gain_from_fusion']=values['fuse_ratio']<1. if values['role']!='qkv' else None
        stages.append(values)
    elif line.startswith('SG_CHAIN ') and 'configs=' in line:
        v=dict(re.findall(r'(\w+)=([^ ]+)',line.split(' configs=')[0]))
        for k in ('m','prologue'):v[k]=int(v[k])
        for k in ('cublas_us','us','ratio'):v[k]=float(v[k])
        v['configs']=ast.literal_eval(line.split('configs=',1)[1]);chains.append(v)
    elif any(s in line for s in ('ERROR','REJECT','BUDGET','NO_PICK','missing_winner')):issues.append(line)
out=dict(receipt=str(receipt.resolve().relative_to(REPO)),payload_sha256=sha,stages=stages,chains=chains,issues=issues,
         interpretation='Cold paired standalone medians. TB/s includes fused glue; differences near noise need replication. No layer gain inferred.',
         abi='FusedStreamingProjection in prepare; call returns (hidden,stream) or (pool,ids,weights,local_ids). Config+200 uses distributed moments. Pool_norm reads the FP32 stream.')
dest=HERE/'handoff.json';dest.write_text(json.dumps(out,indent=2)+'\n')
print(dest)
for r in stages:print(f"{r['role']:10} M={r['m']:3} config={r['cfg']:3}/{r['split']} cuBLAS={r['cublas_us']:.3f} fused={r['us']:.3f} separate={r['separate_us']:.3f} ratio={r['ratio']:.4f} fusion={r['fuse_ratio']:.4f}")
for r in chains:print('CHAIN',r)
for r in issues:print('ISSUE',r)
