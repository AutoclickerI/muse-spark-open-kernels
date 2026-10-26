"""One held-out six-load G0 for fixed structural and native-layout portfolios."""
import base64,hashlib,json,lzma,math,statistics,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];sys.path.insert(0,str(R/'tools'))
from lab_packet import result_lines,read_packets

def ci(v):
 z=[math.log(x) for x in v];mu=statistics.mean(z);h=2.131449546*statistics.stdev(z)/math.sqrt(len(z));return [math.exp(x) for x in (mu,mu-h,mu+h)]
p=Path(sys.argv[1])
if not p.exists():p=next((R/'runs').glob('muse_*/*'+sys.argv[1]+'.json'))
d=json.loads(p.read_text());tail=d['result']['log_tail'];lines=result_lines(tail);bs=[s.split(' ',1)[1] for s in lines if s.startswith('WHOLE_XZ85 ')];assert len(bs)==1,[s[:2000] for s in lines];raw=json.loads(lzma.decompress(base64.b85decode(bs[0])));transport=[e for packet in read_packets(tail) for e in packet['packet'].get('transport_errors',[])];rows={tuple(r['case']):r['values'] for r in raw['rows']};arms=raw['arms'][1:];shapes={};loads={};whole={}
valid=raw['complete'] and raw['checks']==16*12*len(raw['arms']) and raw.get('native_projection_TASK')==48 and not transport
for k,values in rows.items():shapes[str(k)]={a:ci([b/x for b,x in zip(values[a],values['A'])]) for a in arms}
for b,q,label in ((1,1,'decode1'),(16,1,'decode16'),(32,1,'decode32'),(1,8,'verify1'),(16,8,'verify16'),(32,8,'verify32')):
 s,f=rows[b,q,2048],rows[b,q,0];control=[46*a+16*b for a,b in zip(s['A'],f['A'])];loads[label]={}
 for a in arms:
  v=[c/(46*x+16*y) for c,x,y in zip(control,s[a],f[a])];loads[label][a]=dict(R=ci(v),samples=v)
for a in arms:
 v=[math.exp(statistics.mean(math.log(loads[l][a]['samples'][i]) for l in loads)) for i in range(16)];r=ci(v);whole[a]=dict(R=r,samples=v,G0=valid and r[1]>1);print(a,r,'G0',whole[a]['G0'])
out=dict(job=d['id'],receipt=str(p.relative_to(R)),receipt_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),valid=valid,transport_errors=transport,raw=raw,arms=whole,loads=loads,shapes=shapes,CI='Student-t15 across16paired log ratios. Each seed retains six-load covariance, Tload=46sliding+16full; no per-row veto. Stage selection seeds15/16/18 disjoint from held-out19; fresh weight/router seeds19000311/13.')
(H/'final_whole_results.json').write_text(json.dumps(out,indent=2)+'\n');print('PREPARE',raw['prepare_s'],raw['added_prepare_s']);print('NATIVE RESOURCES',raw['resources']['np'])
for label in ('verify16','verify32'):print(label,{a:loads[label][a]['R'] for a in arms})
