"""Factorial decoder-vs-bytes and overlap-vs-serialization analysis."""
import base64,hashlib,json,lzma,math,statistics,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];sys.path.insert(0,str(R/'tools'))
from lab_packet import read_packets,result_lines
from whole_read import ci
p=Path(sys.argv[1])
if not p.exists():p=next((R/'runs').glob('muse_*/*'+sys.argv[1]+'.json'))
d=json.loads(p.read_text());target='grouped_results.json' if d['id']=='ff32499a47704733' else 'pipeline_results.json';tail=d['result']['log_tail'];lines=result_lines(tail);bs=[s.split(' ',1)[1] for s in lines if s.startswith('PIPE_XZ85 ')]
if not bs:
 out=dict(job=d['id'],valid=False,reason='No retained complete pipeline matrix; unmeasured',errors=[s[:3000] for s in lines if 'ERROR' in s]);(H/target).write_text(json.dumps(out,indent=2)+'\n');print(out);raise SystemExit(0)
assert len(bs)==1
raw=json.loads(lzma.decompress(base64.b85decode(bs[0])));errors=[e for packet in read_packets(tail) for e in packet['packet'].get('transport_errors',[])];rows=[];summary=[]
for row in raw['rows']:
 arms=[tuple(a) for a in row['arms']];unit=raw.get('unit_us',1);ps=[[v*unit for v in a] for a in row['producer']]
 for kind,values in [('producer',row['producer']),('neighbour',row['neighbour'])]:
  if values is None:continue
  vals=[[v*unit for v in a] for a in values]
  for j,arm in enumerate(arms[2:],2):
   ratio=ci([b/a for a,b in zip(vals[0],vals[j])]);entry=dict(t=row['t'],role=row['role'],kind=kind,arm=arm,ratio=ratio,control_us=statistics.mean(vals[0]),candidate_us=statistics.mean(vals[j]),samples=vals[j],control_samples=vals[0],AA_ratio=ci([b/a for a,b in zip(vals[0],vals[1])]))
   piped,coded,bn,s=arm
   if piped:
    pair=arms.index((False,coded,bn,s));entry['grouped_and_tile_order_vs_serial' if target.startswith('grouped') else 'overlap_vs_serial']=ci([a/b for a,b in zip(vals[j],vals[pair])])
   if coded:
    pair=arms.index((piped,False,bn,s));entry['coded_vs_raw_same_math']=ci([a/b for a,b in zip(vals[j],vals[pair])])
   rows.append(entry)
 chosen='producer' if row['role']=='qkv_gate' else 'neighbour';rs=[r for r in rows if r['t']==row['t'] and r['role']==row['role'] and r['kind']==chosen and r['arm'][1]];summary.append(min(rs,key=lambda r:r['ratio'][0]))
out=dict(job=d['id'],receipt=str(p.relative_to(R)),receipt_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),valid=raw['complete'] and not errors,transport_errors=errors,raw=raw,rows=rows,best_coded=summary,CI='Student-t7 over eight paired log ratios; selection data; candidate/control time')
if target.startswith('grouped'):
 out['best_grouped_coded']=[min([r for r in rows if r['t']==t and r['role']==role and r['kind']==('producer' if role=='qkv_gate' else 'neighbour') and r['arm'][0] and r['arm'][1]],key=lambda r:r['ratio'][0]) for t in (128,256) for role in ('qkv_gate','o','pool_router','unpooler')]
(H/target).write_text(json.dumps(out,indent=2)+'\n')
for r in summary:print(r['t'],r['role'],r['arm'],*[round(v,6) for v in r['ratio']], 'code/raw',r.get('coded_vs_raw_same_math'),'pipe/serial',r.get('overlap_vs_serial'))
print('VALID',out['valid'],'FAILURE',raw['failure'])
