"""Read a landed native receipt once and emit its explicit dispatch handoff."""
import argparse
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]


def handoff(receipt, payload):
    sha = hashlib.sha256(payload.read_bytes()).hexdigest()
    tail = receipt['result'].get('log_tail','')
    match = re.search(r'^LAB_BEGIN verify_streaming_lab '+sha[:16]+r'\n(.*?)^LAB_END',tail,re.M|re.S)
    if not match:
        raise ValueError('receipt has no matching payload section')
    section = match[1]
    if 'native=True' not in section or 'WEIGHT_WIDE_SUMMARY' not in section:
        raise ValueError('native H200 execution or complete sweep not established')
    if re.search(r'^(INCOMPLETE|REJECT_ALL|LAB_ERROR)',section,re.M):
        raise ValueError('lab did not complete')
    rows = []
    for line in section.splitlines():
        if not line.startswith('M='):
            continue
        fields = dict(re.findall(r'(\w+)=([^ ]+)',line))
        rows.append(dict(tokens=int(fields['M']),role=fields['role'],
                         code=int(fields['code']),splits=int(fields['split']),
                         ratio=float(fields['ratio']),old_us=float(fields['old_us']),
                         new_us=float(fields['new_us']),rel=float(fields['rel']),
                         router=float(fields['router']),row=float(fields['row'])))
    expected = {(m,n) for m in (8,128,256) for n in
                ('qkv_gate','o','pool_router','unpooler_slice' if m==8 else 'unpooler')}
    if {(r['tokens'],r['role']) for r in rows} != expected or len(rows)!=12:
        raise ValueError('receipt does not contain all twelve projection keys')
    for r in rows:
        limit = 1e-4 if r['role']=='pool_router' else .01
        if not (0<r['ratio']<100 and r['old_us']>0 and r['new_us']>0 and
                0<=r['rel']<=limit and 0<=r['row']<=5*limit and 0<=r['router']<=1e-4):
            raise ValueError('invalid numeric admission row '+str(r))
    recipes = {(r['tokens'],r['role']):(r['code'],r['splits']) for r in rows if r['ratio']<1.0}
    return dict(payload_sha256=sha,rows=rows,recipes=repr(recipes),native=True,
                note='Ratios below one are paired standalone gains; integrated whole-layer performance remains unverified.')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipt',help='job ID or local receipt JSON')
    args = parser.parse_args()
    path = Path(args.receipt)
    if not path.exists():
        hits = list(REPO.glob(f'runs/*/*_{args.receipt}.json'))
        if len(hits)!=1:
            raise SystemExit('No unique local receipt; no judge request was made.')
        path = hits[0]
    print(json.dumps(handoff(json.loads(path.read_text()),HERE.parent/'verify_streaming_lab.py'),indent=2))
