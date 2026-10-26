"""Select native dispatch points by actual whole-layer impact; held-out G0 follows."""
import base64,json,lzma,math,statistics,sys,struct
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];sys.path.insert(0,str(R/'tools'))
from lab_packet import result_lines,read_packets
from promotion import interval
exec((H/'codec.py.inc').read_text())
def ci(v,control=None):
 x=interval(v,control);return [x['R'],*x['CI95']]

p=Path(sys.argv[1])
if not p.exists():p=next((R/'runs').glob('muse_*/*'+sys.argv[1]+'.json'))
d=json.loads(p.read_text());transport=read_packets(d['result']['log_tail']);assert transport and all(not x['packet'].get('transport_errors') and x['packet'].get('complete') is not False for x in transport),'Incomplete or failed wrapper';blob=next(l.split(' ',1)[1] for l in result_lines(d['result']['log_tail']) if l.startswith('NATIVE_TUNE_XZ85 '));raw=nb_time_decode(blob);assert raw['complete'],raw['failure'];cells={tuple(x['case']):{a:[statistics.mean(z[a])*.008 for z in x['times']] for a in x['arms']} for x in raw['rows']};manifest=json.loads((H/'tune_manifest.json').read_text());arms=[]
for name,point in manifest['arms'].items():
 t,role,cfg,s=point[0];b=t//8;sl,fu=cells[b,8,2048],cells[b,8,0];a=[46*x+16*y for x,y in zip(sl['A'],fu['A'])];aa=[46*x+16*y for x,y in zip(sl['AA'],fu['AA'])];z=[46*x+16*y for x,y in zip(sl[name],fu[name])];ratio=[b/a0 for b,a0 in zip(z,a)];control=[b/a0 for b,a0 in zip(aa,a)]
 arms.append(dict(arm=name,t=t,role=role,config=cfg,split=s,time_ratio=ci(ratio,control),AA=ci(control),saved_us=statistics.mean((a0-b)/62 for b,a0 in zip(z,a)),shape_ratios={str(w):ci([b/a0 for b,a0 in zip(cells[b,8,w][name],cells[b,8,w]['A'])],[b/a0 for b,a0 in zip(cells[b,8,w]['AA'],cells[b,8,w]['A'])]) for w in (2048,0)}))
points=[];best=[]
for t in (128,256):
 for role in ('qkv_gate','o','pool_router','unpooler'):
  row=min((x for x in arms if x['t']==t and x['role']==role),key=lambda x:x['time_ratio'][0]);best.append(row)
  if row['time_ratio'][0]<1:points.append([t,role,row['config'],row['split']])
out=dict(job=d['id'],raw=raw,arms=arms,best=best,points=points,limits='Stage selection uses complete whole-layer native graphs at the two changed loads, every16round arithmetic mean,8freshseeds. CI uses larger primary/A:A log SE. New independent six-load G0 required; no per-shape veto.')
(H/'tune_results.json').write_text(json.dumps(out,indent=2)+'\n');(H/'native_points.json').write_text(json.dumps(points,indent=2)+'\n')
for row in best:print(row['t'],row['role'],row['config'],row['split'],row['time_ratio'],row['saved_us'])
