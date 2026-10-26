"""Fresh native one-boundary oracle; all symmetric rounds enter arithmetic means."""
import base64,json,lzma,math,statistics,sys,struct
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];sys.path.insert(0,str(R/'tools'))
from lab_packet import result_lines,read_packets
from promotion import interval
exec((H/'codec.py.inc').read_text())
def ci(v,log=False):
 z=[math.log(x) for x in v] if log else v;mean=statistics.mean(z);h=2.364624251*statistics.stdev(z)/math.sqrt(len(z));v=[mean,mean-h,mean+h];return [math.exp(x) for x in v] if log else v
p=Path(sys.argv[1])
if not p.exists():p=next((R/'runs').glob('muse_*/*'+sys.argv[1]+'.json'))
d=json.loads(p.read_text());transport=read_packets(d['result']['log_tail']);assert transport and all(not x['packet'].get('transport_errors') and x['packet'].get('complete') is not False for x in transport),'Incomplete or failed wrapper';lines=result_lines(d['result']['log_tail']);blob=next(x.split(' ',1)[1] for x in lines if x.startswith('NATIVE_ORACLE_XZ85 '));raw=nb_decode(blob);assert raw['complete'],raw['failure'];arms=['AA','P_READY','SINGLE_READY','C_READY','BOTH_READY'];values={};shapes=[]
for row in raw['rows']:
 k=tuple(row['case']);z={a:[] for a in ['A']+arms}
 for sample in row['times']:
  assert all(len(v)==16 for v in sample.values())
  for a in z:z[a].append(statistics.mean(sample.get(a,sample['AA']))*.008)
 values[k]=z;r=dict(case=k,active=row['active'],arms={a:dict(time_ratio=ci([b/a0 for b,a0 in zip(z[a],z['A'])],True),saved_us=ci([a0-b for b,a0 in zip(z[a],z['A'])])) for a in arms})
 r['split_only_us']=ci([a-b for a,b in zip(z['P_READY'],z['SINGLE_READY'])]);r['consumer_interaction_us']=ci([a-p-c+b for a,p,c,b in zip(z['A'],z['P_READY'],z['C_READY'],z['BOTH_READY'])]);shapes.append(r)
loads={};whole={}
for b,q,label in ((1,1,'d1'),(16,1,'d16'),(32,1,'d32'),(1,8,'v1'),(16,8,'v16'),(32,8,'v32')):
 s,f=values[b,q,2048],values[b,q,0];z={a:[46*x+16*y for x,y in zip(s[a],f[a])] for a in ['A']+arms};loads[label]={a:dict(R=ci([a0/b0 for a0,b0 in zip(z['A'],z[a])],True),saved_us_per_layer=ci([(a0-b0)/62 for a0,b0 in zip(z['A'],z[a])]),samples=[a0/b0 for a0,b0 in zip(z['A'],z[a])]) for a in arms}
 loads[label]['split_only']=dict(R=ci([a/b for a,b in zip(z['P_READY'],z['SINGLE_READY'])],True),saved_us_per_layer=ci([(a-b)/62 for a,b in zip(z['P_READY'],z['SINGLE_READY'])]),samples=[a/b for a,b in zip(z['P_READY'],z['SINGLE_READY'])])
for a in arms+['split_only']:
 samples=[math.exp(statistics.mean(math.log(loads[k][a]['samples'][i]) for k in loads)) for i in range(8)];noise=whole['AA']['samples'] if a!='AA' else None;iv=interval(samples,noise);whole[a]=dict(R=[iv['R'],*iv['CI95']],samples=samples,interval=iv)
out=dict(job=d['id'],receipt=str(p.relative_to(R)),raw=raw,shapes=shapes,loads=loads,whole=whole,limits='All arms are synthetic native-graph deletion bounds, never production/G0 candidates. Missing inactive arm equals unchanged AA. Split-only compares two equally deleted-producer graphs. Every round contributes; no trimming or selective stopping.')
(H/'oracle_results.json').write_text(json.dumps(out,indent=2)+'\n');print('whole',json.dumps(whole));print('split',[(k,x['split_only']['saved_us_per_layer']) for k,x in loads.items()])
