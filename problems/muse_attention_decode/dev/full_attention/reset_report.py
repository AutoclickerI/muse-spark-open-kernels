"""Parse frozen reset receipts without mistaking a failed host/control for a loss."""
import argparse
import hashlib
import json
import re
from pathlib import Path

H=Path(__file__).resolve().parent;R=H.parents[3]
PAT=re.compile(r'(\w+)=([\d.]+)/r([\d.]+)\[([\d.]+),([\d.]+)\](?:/se([\d.]+))?')


def parse(path,kind):
    receipt=json.loads(path.read_text());run=receipt.get('result',{});tail=run.get('log_tail','')
    blocks=re.findall(r'^LAB_BEGIN (\S+).*?$(.*?)(?=^LAB_END|\Z)',tail,re.M|re.S)
    blocks=[(n,b) for n,b in blocks if ('decode_softmax' in n if kind=='decode' else 'softmax_mix' in n)]
    out=dict(job=receipt.get('id',path.stem.split('_')[-1]),receipt=str(path.relative_to(R)),
             receipt_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),kind=kind,
             host_status=run.get('status'),host_passed=run.get('passed'),rows=[],clocks={},errors=[])
    if len(blocks)!=1:out['errors'].append('missing/ambiguous lab block');return out
    name,text=blocks[0];out['payload']=name
    out['errors'] += re.findall(r'^LAB_ERROR.*$',text,re.M)
    out['complete']='END elapsed=' in text and len(re.findall(r'^R B',text,re.M))==2 and not out['errors']
    for line in text.splitlines():
        if line.startswith('CLOCK B'):
            key,raw=line.split(' ',2)[1:];out['clocks'][key]=json.loads(raw)
        if not line.startswith('R B'):continue
        b=int(re.search(r'R B(\d+)',line)[1]);a=float(re.search(r' A=([\d.]+)',line)[1])
        arms={n:dict(us=float(us),ratio=float(r),lo=float(lo),hi=float(hi),se=float(se) if se else None)
              for n,us,r,lo,hi,se in PAT.findall(line)}
        assert 'A2' in arms,line
        aa=arms['A2'];out['rows'].append(dict(batch=b,control_us=a,aa=aa,arms=arms,
            # Conservative application of the written "beat A:A spread" rule.
            clear_winners=[n for n,v in arms.items() if n!='A2' and v['hi']<min(1.,aa['lo'])]))
    out['native_checks_completed']=bool(out.get('complete'))
    return out


def main():
    ap=argparse.ArgumentParser();ap.add_argument('kind',choices=('verify','decode'));ap.add_argument('receipt',type=Path)
    ap.add_argument('--out',type=Path);a=ap.parse_args();r=parse(a.receipt.resolve(),a.kind)
    if a.out:a.out.write_text(json.dumps(r,indent=2)+'\n')
    print('job',r['job'],'complete',r.get('complete',False),'errors',r['errors'])
    for row in r['rows']:
        print('B',row['batch'],'A us',row['control_us'],'A:A',row['aa'],'clear winners',row['clear_winners'])
        for name,v in row['arms'].items():print(name,v)


if __name__=='__main__':main()
