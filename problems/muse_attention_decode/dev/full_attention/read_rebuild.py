"""TASK-gated shape nominations and paired six-load G0 diagnostics."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path
from audit_report import parse

H=Path(__file__).resolve().parent


def main():
    ap=argparse.ArgumentParser();ap.add_argument('receipt',type=Path);ap.add_argument('--out',type=Path);ap.add_argument('--reuse',action='store_true');a=ap.parse_args()
    directory=H/('reuse' if a.reuse else 'rebuild')
    d=parse(a.receipt.resolve());r=d['result'];manifest=json.loads((directory/'lab_manifest.json').read_text())
    assert r['base_sha256']==manifest['base_sha256']
    v=r.get('seed_us',{});points=[]
    # Ignore unchanged dispatch cells: their noise is not a candidate piece.
    targets={name:('16x8f','32x8f') for name in ('ready','ready_elide','reg','parity')}
    # d1 full dispatches partial_port before attn_main, so it is unchanged.
    targets['kmajor']=('16x1f','32x1f')
    if a.reuse:targets={name:('16x1f','32x1f') for name in ('kmajor','kmajor_reuse')}
    if d['correctness_passed']:
        for name,shapes in targets.items():
            for shape in shapes:
                ratios=[x[name]/x['A'] for x in v[shape]]
                mean=statistics.mean(ratios);se=statistics.stdev(ratios)/len(ratios)**.5
                points.append(dict(arm=name,shape=shape,ratio=mean,SE=se,seeds=len(ratios),
                    nominated=mean<1,patch=str((H/'rebuild'/(name+'.patch')).relative_to(H.parents[3])),
                    patch_sha256=hashlib.sha256((H/'rebuild'/(name+'.patch')).read_bytes()).hexdigest(),
                    source_sha256=manifest['arms'][name]))
    d['shape_points']=points
    d['decision']='Nominate every affected shape point<1; P4 composes and confirms held-out six-load R lower CI>1. Failed TASK/probe leaves all arms unmeasured.'
    if not a.reuse and '32x8f' in v:
        d['parity_over_register_control']={shape:[row['parity']/row['reg'] for row in v[shape]] for shape in ('16x8f','32x8f')}
    (a.out or directory/'result.json').write_text(json.dumps(d,indent=2)+'\n')
    print('correctness',d['correctness_passed'],'A:A',d['controls'])
    print('G0',d['paired_six_load_score'])
    for p in points:print(p)


if __name__=='__main__':main()
