"""Read a landed local receipt; emit a prefill-only H200-positive shortlist.

No API requests, polling or waiting. Keep frozen lab files intact until read.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pprint

ROOT=Path(__file__).resolve().parents[4]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('receipt',type=Path)
parser.add_argument('--recipes',type=Path)
args=parser.parse_args()
tail=json.loads(args.receipt.read_text())['result'].get('log_tail','')
payload=ROOT/'prof/lab/prefill_dense_cutlass.py'
sha=hashlib.sha256(payload.read_bytes()).hexdigest()[:16]
if f'LAB_BEGIN prefill_dense_cutlass {sha}' not in tail:
    raise SystemExit('receipt does not identify this frozen payload; no performance attribution')
rows=[json.loads(line.split(' ',1)[1]) for line in tail.splitlines() if line.startswith('PFC_ROW ')]
recipes={}
print('| T | Role | cuBLAS us | Incumbent us | CUTLASS us | ratio | TFLOPS | recipe |')
print('|---:|---|---:|---:|---:|---:|---:|---|')
for row in rows:
    best=row['best'];pick=row['pick']
    tested=(next(item for item in row['finals'] if item[0]==pick) if pick is not None
            else min(row['finals'],key=lambda item:item[3]) if row['finals'] else None)
    if best is None:
        print(f'| {row["m"]} | {row["key"]} | {row["cu"]} | {row["inc"]} | rejected/budget | | | |')
        continue
    recipe=tuple(tested[0])
    print(f'| {row["m"]} | {row["key"]} | {row["cu"]} | {row["inc"]} | {best[1]} | {best[2]} | {best[3]} | {recipe} |')
    if pick is not None and best[2]<1 and tuple(pick)==recipe:
        recipes[row['m'],row['key']]=[recipe]
expected={(m,key) for m in (753,3012,16384) for key in ('pooler','unpooler','qkv','o','router_cat')}
seen={(row['m'],row['key']) for row in rows}
print('H200-positive keys',len(recipes),'missing',sorted(expected-seen))
for line in tail.splitlines():
    if line.startswith(('PFC_META','PFC_REJECT','PFC_BUDGET','PFC_END','LAB_END')):print(line)
if args.recipes:
    args.recipes.write_text('PFC_RECIPES = '+pprint.pformat(recipes,width=100,sort_dicts=False)+'\n')
