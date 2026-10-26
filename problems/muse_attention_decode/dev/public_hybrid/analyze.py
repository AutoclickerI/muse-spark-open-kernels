import argparse,base64,hashlib,json,lzma,statistics,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];sys.path.insert(0,str(R/'tools'));from lab_packet import paired_evidence
ap=argparse.ArgumentParser();ap.add_argument('receipt',type=Path);a=ap.parse_args();receipt=json.loads(a.receipt.read_text());d=paired_evidence(receipt['result'].get('log_tail',''),'RESULT');assert d and d['complete'] and not d['failures'],d
exec((H/'codec.py.inc').read_text());d=decode_rounds(d)
meta=json.loads((H/'out/lab_manifest.json').read_text());assert d['seeds']==meta['seeds'];assert d['identities']==meta['identities'];assert d['checks']=={k:meta[k+'_TASK'] for k in ('official','ABI','scored')}
fixed=d['fixed_ns'];points=[];controls={};selection={};times={};contrasts=[]
def ci(v):
 mean=statistics.mean(v);se=statistics.stdev(v)/len(v)**.5;return dict(r=mean,se=se,lo=mean-2.364624251*se,hi=mean+2.364624251*se,seed_ratios=v)
for cell,samples in d['round_ns'].items():
 names=meta['case_arms'][cell];assert len(samples)==8 and all(set(s)==set(names) and all(len(v)==2*len(names) and min(v)>fixed for v in s.values()) for s in samples)
 means=[{n:statistics.mean(v)-fixed for n,v in s.items()} for s in samples];times[cell]=means;controls[cell]=ci([s['A2']/s['A'] for s in means]);best=None
 for n in names:
  if n in ('A','A2'):continue
  p=dict(cell=cell,arm=n,**ci([s[n]/s['A'] for s in means]));p['nominated']=p['r']<1;points.append(p)
  if p['nominated'] and (best is None or p['r']<best[0]):best=p['r'],n
  if 'F' in names and n in ('P','Q','E','H'):contrasts.append(dict(cell=cell,arm=n,parent='F',**ci([s[n]/s['F'] for s in means])))
 if best:selection[cell]=best[1]
 ps=[p for p in points if p['cell']==cell];p=min(ps,key=lambda p:p['r']);print(cell,p['arm'],f"{p['r']:.6f}[{p['lo']:.6f},{p['hi']:.6f}]",'AA',f"{controls[cell]['r']:.6f}[{controls[cell]['lo']:.6f},{controls[cell]['hi']:.6f}]")
r=dict(measured=True,job=receipt['id'],receipt=str(a.receipt.resolve().relative_to(R)),payload_sha256=meta['payload_sha256'],seeds=d['seeds'],checks=d['checks'],points=points,controls=controls,selection=selection,seed_ns=times,contrasts=contrasts,graph_nodes=d['graph_nodes'],elapsed=d['elapsed'],gate='All positive correct per-cell points nominated; P4 held-out whole-layer G0 is promotion. No per-load veto. All rounds retained.')
(H/'result.json').write_text(json.dumps(r,indent=2)+'\n');(H/'receipt_data.json').write_text(json.dumps(d,indent=2)+'\n');(H/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
