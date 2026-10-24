"""Read projection lab receipts; emit a positive-H200-only layer shortlist."""
import argparse
import ast
import hashlib
import json
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]


def read(path):
    receipt=json.loads(path.read_text())
    if receipt.get('id')=='6b72087f64584258':
        raise ValueError('native v1 reinitialized live mbarriers; invalid for dispatch selection')
    result=receipt['result']
    if not result.get('passed'):raise ValueError(f'{path}: host correctness failed')
    text=result.get('log_tail','')
    if 'LAB_ERROR' in text or 'LAB_BEGIN' not in text:raise ValueError(f'{path}: lab did not complete')
    rows=[]
    for line in text.splitlines():
        match=re.match(r'VP_LT (128|256) (\w+) best=(.*?) tried=',line)
        if match:
            m,name,best=match.groups();ratio,choice,times=ast.literal_eval(best)
            if choice=='torch':continue
            recipe=('lt',*choice)
            rows.append((int(m),name,recipe,times,ratio))
        match=re.match(r'(VP|VG)_TMA_BEST (128|256) (\w+) pick=(.*?) raw_us=(.*?) ratio=([^ ]+)',line)
        if match:
            prefix,m,name,choice,times,ratio=match.groups()
            choice=ast.literal_eval(choice);times=ast.literal_eval(times);ratio=float(ratio)
            if choice is not None:rows.append((int(m),name,('grouped' if prefix=='VG' else 'tma',choice),times,ratio))
    return rows


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('receipts',nargs='+',type=Path)
    ap.add_argument('--recipes',type=Path,help='write a literal shortlist for decode prepare')
    args=ap.parse_args();recipes={}
    print('|M|role|family|control us|candidate us|paired ratio|')
    print('|---:|---|---|---:|---:|---:|')
    for path in args.receipts:
        print(f'\n{path.name}; receipt_sha256={hashlib.sha256(path.read_bytes()).hexdigest()}')
        try:rows=read(path)
        except Exception as exc:
            print(f'REJECT {type(exc).__name__}: {exc}')
            continue
        for m,name,recipe,times,ratio in rows:
            print(f'|{m}|{name}|`{recipe}`|{times[0]:.3f}|{times[1]:.3f}|{ratio:.4f}|')
            if ratio<1.:recipes.setdefault((m,name),[]).append(recipe)
    if args.recipes:
        args.recipes.write_text('VERIFY_PROJ_RECIPES = '+repr(recipes)+'\n')
        print(f'\n{args.recipes}: {len(recipes)} H200-positive keys; caller must retime against its own incumbent.')


if __name__=='__main__':main()
