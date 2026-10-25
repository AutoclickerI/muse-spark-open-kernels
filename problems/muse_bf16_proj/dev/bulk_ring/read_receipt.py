"""Parse a completed local receipt; never contact or wait on the judge."""
import argparse
import hashlib
import json
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]

def parse(text):
    pattern=(r'^R (\d+)([QOPU]) c(\d+) B=([\d.]+)/([\d.]+) N=([\d.]+)/([\d.]+) r=([\d.]+) '
        r's=([\d.,]+) aa=([\d.]+)/([\d.]+) G/S/B=(\d+)/(\d+)/(\d+) sm=(\d+) u=(\d+):(\d+) tail=([\d.]+) v=([\d./,]+)$')
    roles=[];names=dict(Q='qkv_gate',O='o',P='pool_router',U='unpooler')
    for match in re.finditer(pattern,text,re.M):
        a=match.groups();t=int(a[0]);role=names[a[1]]
        if role=='unpooler' and t<=8:role='unpooler_slice'
        coarse=[list(map(float,v.split('/'))) for v in a[18].split(',')]
        if len(coarse)!=21:raise ValueError('coarse coverage is not21')
        roles.append(dict(tokens=t,role=role,config=int(a[2]),baseline_producer_us=float(a[3]),
            baseline_combined_us=float(a[4]),producer_us=float(a[5]),combined_us=float(a[6]),ratio=float(a[7]),
            seed_ratios=list(map(float,a[8].split(','))),aa_producer=float(a[9]),aa_combined=float(a[10]),
            grid=int(a[11]),partial_splits=int(a[12]),ring_bytes=int(a[13]),sms=int(a[14]),
            units_min=int(a[15]),units_max=int(a[16]),tail_us=float(a[17]),coarse_producer_combined=coarse))
    segments=[]
    for match in re.finditer(r'^S (\d+) aa=([\d.]+) (.*)$',text,re.M):
        trials=[]
        for v in re.finditer(r'(O6P|O6|OP|O|P)=([\d.]+):([\d.]+)/([\d.]+)/flip(\d+)/s([\d.:]+)',match[3]):
            trials.append(dict(arm=v[1],baseline_us=float(v[2]),candidate_us=float(v[3]),ratio=float(v[4]),
                route_token_flips=int(v[5]),seed_ratios=list(map(float,v[6].split(':')))))
        segments.append(dict(tokens=int(match[1]),aa=float(match[2]),trials=trials))
    return dict(roles=roles,segments=segments,fixed_o6=re.findall(r'^O6 .*$',text,re.M),
        complete=len(roles)==16 and len(segments)==4 and 'END roles16 segments4 configs21 ' in text and 'LAB_ERROR' not in text)

def main():
    p=argparse.ArgumentParser();p.add_argument('receipt');args=p.parse_args();path=Path(args.receipt)
    if not path.exists():
        hits=list(ROOT.glob('runs/*/*_'+args.receipt+'.json'))
        if len(hits)!=1:raise SystemExit('No unique completed local receipt for '+args.receipt)
        path=hits[0]
    receipt=json.loads(path.read_text());tail=receipt['result'].get('log_tail','')
    block=re.search(r'^LAB_BEGIN dense_bulk_ring (\w+)\n(.*?)^LAB_END.*?$',tail,re.M|re.S)
    if not block:raise SystemExit('No dense_bulk_ring section')
    manifest=json.loads((HERE/'manifest.json').read_text())
    if block[1]!=manifest['payload_sha256'][:16]:raise SystemExit('Receipt/payload mismatch')
    projection=HERE/'projection.py';sha=hashlib.sha256(projection.read_bytes()).hexdigest()
    if sha!=manifest['projection_sha256']:raise SystemExit('Projection source changed since measurement')
    result=parse(block[2]);cells=[]
    if result['complete']:
        for row in result['roles']:
            if row['ratio']<.985 and max(row['seed_ratios'])<1.:
                cells.append(dict(tokens=row['tokens'],role=row['role'],config=row['config'],ratio=row['ratio'],
                    seed_ratios=row['seed_ratios'],requires_live_layer_agreement=True,
                    aa_combined=row['aa_combined']))
    result.update(receipt=str(path),payload_sha256=manifest['payload_sha256'],projection=str(projection.relative_to(ROOT)),
        projection_sha256=sha,candidate_cells=cells,dispatch_changed=False,
        integration="Prepare BulkRingProjection; a ('bulk',config) branch must return its (partials,S) tuple directly. Live whole-layer gates remain required.")
    (HERE/'h200_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
