"""Exact-receipt S15 promotion: retained changes win, every untouched shape/load stays within A:A."""
import argparse
import hashlib
import json
from pathlib import Path

H=Path(__file__).resolve().parent
R=H.parents[3]


def radius(aa):
    return max(abs(x-1) for x in aa)


def active(name,b,q,w):
    t=b*q
    if name=='A2':return False
    if name=='last':return b==1 and q==1 and bool(w)
    if name=='regmax':return q==8 and b in (16,32) and not w
    if name in ('proj','o'):return t in (1,8,16,32)
    if name=='qkv':return t in (1,32)
    if name=='unpooler':return t in (16,32)
    if name=='S14':return any(active(n,b,q,w) for n in ('proj','last','regmax'))
    raise AssertionError(name)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('job');a=ap.parse_args()
    files=list((R/'runs/muse_decode_layer').glob('*_'+a.job+'*.json'));assert len(files)==1,files
    receipt=json.loads(files[0].read_text())
    queued=json.loads((H/'queue.json').read_text())
    assert hashlib.sha256(files[0].with_suffix('.py').read_bytes()).hexdigest()==queued['wrapper_sha256']
    log=receipt['result'].get('log_tail','')
    rows=[line.removeprefix('RESULT ') for line in log.splitlines() if line.startswith('RESULT ')]
    assert len(rows)==1,'missing or truncated RESULT'
    raw=json.loads(rows[0]);manifest=json.loads((H/'ab_manifest.json').read_text())
    assert raw['host_sha256']==manifest['sha256']
    assert raw['seeds']==16
    controls={};wins={}
    for label,shape in raw['shapes'].items():
        b=int(label[1:-1]);q=1 if label[0]=='d' else 8;w=2048 if label[-1]=='S' else 0
        for name,r in shape['r'].items():
            if name!='A2' and not active(name,b,q,w):
                controls[name+':'+label]=dict(ratio=r,AA_radius=radius(shape['aa']),within_AA=abs(r-1)<=radius(shape['aa']))
    for load,row in raw['loads'].items():
        kind,b=load.split();b=int(b);q=1 if kind=='decode' else 8
        for name,values in row['r'].items():
            if name=='A2':continue
            r=values[0];key=name+':'+load;aa=radius(row['aa'])
            if any(active(name,b,q,w) for w in (2048,0)):
                wins[key]=dict(ratio=r,se=values[1],normalized=values[2],normse=values[3],AA_radius=aa,wins=r>1+aa)
            else:controls[key]=dict(ratio=r,AA_radius=aa,within_AA=abs(r-1)<=aa)
    pieces_win=all(v['wins'] for k,v in wins.items() if k.split(':')[0] in ('proj','last','regmax'))
    base_win=all(v['wins'] for k,v in wins.items() if k.startswith('S14:'))
    controls_pass=all(v['within_AA'] for v in controls.values())
    report=dict(job=receipt['id'],source_sha256=raw['host_sha256'],official_pass=receipt['result'].get('passed',False),
                complete=raw['complete'],wins=wins,controls=controls,raw=raw,
                freeze_gate=bool(receipt['result'].get('passed') and raw['complete'] and pieces_win and base_win and controls_pass))
    (H/(receipt['id'][:8]+'_report.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('raw','controls')},indent=2))


if __name__=='__main__':main()
