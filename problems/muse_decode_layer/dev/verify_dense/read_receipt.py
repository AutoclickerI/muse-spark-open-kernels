"""Promote only complete H200 receipt lines beating both pinned incumbent references."""
from pathlib import Path
import argparse,json,re,hashlib
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
p=argparse.ArgumentParser();p.add_argument('receipt',type=Path);a=p.parse_args()
receipt=json.loads(a.receipt.read_text());text=json.dumps(receipt,ensure_ascii=False)
# Receipts often store multiline log_tail as a string; recursively collect strings.
def strings(v):
 if isinstance(v,str):yield v
 elif isinstance(v,dict):
  for x in v.values():yield from strings(x)
 elif isinstance(v,list):
  for x in v:yield from strings(x)
log='\n'.join(strings(receipt));manifest=json.loads((HERE/'manifest.json').read_text())
assert 'LAB_BEGIN verify_dense_library '+manifest['payload_sha256'][:16] in log,'source mismatch'
assert 'END rows=' in log and 'INCOMPLETE' not in log and 'LAB_ERROR' not in log,'incomplete lab'
rows=[];selections={}
rolemap={'q':'qkv_gate','o':'o','pr':'pool_router','u':'unpooler'}
cellmap={'lt':'lt_search','dg':'deep_gemm','c0':'cutlass0','c1':'cutlass1','c2':'cutlass2','reg':'best7_registry'}
for line in log.splitlines():
 if re.match(r'S\d+ ',line):
  tag,role,selection=line.split(' ',2);selections[int(tag[1:]),rolemap[role]]=json.loads(selection)
 if not re.match(r'D\d+ ',line):continue
 match=re.match(r'D(\d+) (\w+) (\w+) (\d+) ([\d.]+)/([\d.]+) ([\d.]+) ([-\d.]+) ([\d.e+-]+)',line)
 assert match,line
 m,role,cell,n,old,new,ratio,aa,error=match.groups()
 rows.append(dict(m=int(m),role=rolemap[role],cell=cellmap[cell],count=int(n),old=float(old),new=float(new),ratio=float(ratio),aa=float(aa),error=float(error)))
for row in rows:
 if row['cell']=='lt_search':row['selection']={key:value for key,value in selections[row['m'],row['role']].items() if key in ('family','nn','blob','config')}
 elif row['cell']=='deep_gemm':row['selection']=dict(family='deep_gemm')
 elif row['cell'].startswith('cutlass'):row['selection']=dict(family='cutlass',code=int(row['cell'][-1]),mode=1)
 else:row['selection']=dict(family='registry')

winners={}
for m in (64,128,256):
 for role in ('qkv_gate','o','pool_router','unpooler'):
  subset=[r for r in rows if r['m']==m and r['role']==role and r['count']==3]
  assert subset,(m,role)
  control=min([r['old'] for r in subset]+[r['new'] for r in subset if r['cell']=='best7_registry'])
  for row in sorted(subset,key=lambda r:r['new']):
   if row['cell']=='best7_registry':continue
   if f'"m":{m},"role":"{role}","cell":"{row["cell"]}"' in log and 'DISABLED' in log:
    # A partial row may precede a later flip. Scan only disabled records.
    bad=False
    for ln in log.splitlines():
     if ln.startswith('DISABLED '):
      d=json.loads(ln[len('DISABLED '):]);bad|=d.get('m')==m and d.get('role')==role and d.get('cell')==row['cell']
    if bad:continue
   if row['new']<control*.985 and control-row['new']>2*abs(row['aa']):
    winners[m,role]=row['selection'];break
lines=['# H200-receipt-selected verify dense projection cells.','VD_SELECTIONS={']
for key,selection in winners.items():lines.append('    '+repr(key)+': '+repr(selection)+',')
lines+=['}','']
adapter=(HERE/'adapter.py.inc').read_text().replace('VD_SELECTIONS={}', '\n'.join(lines))
cell=(HERE/'projection.py').read_text()+'\n'+adapter
out=HERE/'h200_cell.py';out.write_text(cell)
result=dict(receipt=str(a.receipt),projection_sha256=manifest['projection_sha256'],winners=[dict(m=k[0],role=k[1],selection=v) for k,v in winners.items()],cell=str(out.relative_to(ROOT)),cell_sha256=hashlib.sha256(cell.encode()).hexdigest(),whole_layer_verified=False)
(HERE/'handoff.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
