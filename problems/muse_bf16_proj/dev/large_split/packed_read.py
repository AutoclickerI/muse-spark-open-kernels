"""Read K-major bank evidence; isolate same-geometry layout effects and all arms."""
import base64,hashlib,json,lzma,sys
from pathlib import Path
import packed_source
from combined_read import ci
H=Path(__file__).resolve().parent;R=H.parents[3];sys.path.insert(0,str(R/'tools'))
from lab_packet import result_lines,read_packets
codec=dict(json=json,lzma=lzma,base64=base64);exec((H/'codec.py.inc').read_text(),codec)
p=Path(sys.argv[1])
if not p.exists():p=next((R/'runs').glob('muse_*/*'+sys.argv[1]+'.json'))
d=json.loads(p.read_text());tail=d['result']['log_tail'];lines=result_lines(tail);bs=[l.split(' ',1)[1] for l in lines if l.startswith('PACKED_XZ85 ')];assert len(bs)==1,[l[:1000] for l in lines];raw=codec['combined_decode'](bs[0]);transport=[e for p in read_packets(tail) for e in p['packet'].get('transport_errors',[])];rows=[];best=[];points=[]
for cell in raw['candidate']['rows']:
 arms=[tuple(a) for a in cell['arms']];unit=raw['candidate']['unit_us']
 for kind,key in (('producer','producer'),('neighbour','neighbour')):
  if cell.get(key) is None:continue
  v=[[x*unit for x in a] for a in cell[key]];eligible=[]
  for j,a in enumerate(arms[2:],2):
   r=dict(t=cell['t'],role=cell['role'],kind=kind,arm=a,ratio=ci([b/z for b,z in zip(v[j],v[0])]),AA=ci([b/z for b,z in zip(v[1],v[0])]),A_us=sum(v[0])/8,B_us=sum(v[j])/8,samples=v[j],control_samples=v[0]);rows.append(r)
   if a[0]=='P':
    cfg=packed_source.CONFIGS[a[1]];r['config']=cfg;eligible.append(r)
    if cfg==(cell['t'],128,1,2,0) and a[2] in (2,3,4):
     u=next(i for i,z in enumerate(arms) if z[0]=='U' and z[2]==a[2]);r['packed_vs_rowmajor_same_geometry']=ci([b/z for b,z in zip(v[j],v[u])])
  if kind==('producer' if cell['role']=='qkv_gate' else 'neighbour'):
   b=min(eligible,key=lambda r:r['ratio'][0]);best.append(b);points.append([b['t'],b['role'],b['config'],b['arm'][2]]);print('BEST',b['t'],b['role'],b['config'],b['arm'][2],b['ratio'])
out=dict(job=d['id'],receipt=str(p.relative_to(R)),receipt_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),valid=raw['complete'] and not transport,transport_errors=transport,raw=raw,rows=rows,best=best,CI='Student-t7 across8paired log time ratios; stage selection only; held-out layer G0 governs promotion')
(H/'packed_results.json').write_text(json.dumps(out,indent=2)+'\n');(H/'packed_points.json').write_text(json.dumps(points,indent=2)+'\n');print('VALID',out['valid'],'prepare',raw['candidate']['prepare_s'],'FAILURE',raw['candidate']['failure'])
for r in rows:
 if r.get('packed_vs_rowmajor_same_geometry') and r['kind']=='producer':print('LAYOUT',r['t'],r['role'],r['arm'][2],r['packed_vs_rowmajor_same_geometry'])
