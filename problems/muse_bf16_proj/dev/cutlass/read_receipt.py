"""Extract only measured positive CUTLASS finalists from an already-landed receipt.

No judge requests, waits or polling. Run once on the receipt supplied by the owner.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import pprint
import re

ROOT=Path(__file__).resolve().parents[4]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('receipt',type=Path)
parser.add_argument('--recipes',type=Path)
args=parser.parse_args()
doc=json.loads(args.receipt.read_text());tail=doc['result'].get('log_tail','')
expected=hashlib.sha256((ROOT/'prof/lab/verify_proj_cutlass.py').read_bytes()).hexdigest()[:16]
if f'LAB_BEGIN verify_proj_cutlass {expected}' not in tail:
    raise SystemExit('receipt does not identify this exact frozen lab payload; do not attribute it')
recipes={};seen=set()
for line in tail.splitlines():
    if line.startswith(('VGC_CONTROLS','VGC_REJECT','VGC_END','LAB_END')):print(line)
    m=re.fullmatch(r'VGC_BEST (\d+) (\S+) pick=(.*?) us=(.*?) ratio=([0-9.]+) tried=(.*)',line)
    if not m:continue
    tokens,name,choice,times,ratio,rest=m.groups();key=(int(tokens),name);seen.add(key)
    recipe=ast.literal_eval(choice);times=ast.literal_eval(times);ratio=float(ratio)
    print(f'{key}: recipe={recipe} control/new_us={times} paired_ratio={ratio}; {rest}')
    if recipe is not None and times is not None and ratio<1:
        recipes[key]=[recipe]
print('H200 keys',len(seen),'positive',len(recipes),'missing',
      [(m,n) for m in (128,256) for n in ('qkv_gate','pool_router','unpooler','o') if (m,n) not in seen])
if args.recipes:
    args.recipes.write_text('VGC_RECIPES = '+pprint.pformat(recipes,width=100,sort_dicts=False)+'\n')
