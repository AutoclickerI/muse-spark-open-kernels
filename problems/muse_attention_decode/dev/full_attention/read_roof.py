"""Read TASK-gated stage measurements; whole-layer promotion belongs to P4."""
import argparse,base64,hashlib,json,lzma
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3]
ap=argparse.ArgumentParser();ap.add_argument('receipt',type=Path);ap.add_argument('--page-ring',action='store_true');a=ap.parse_args();o=H/('page_ring' if a.page_ring else 'roof')
receipt=json.loads(a.receipt.read_text());tail=receipt.get('result',{}).get('log_tail','')
assert receipt.get('status')=='done' and receipt.get('result',{}).get('passed'),(receipt.get('status'),receipt.get('result',{}).get('error'))
blob=next(l[12:] for l in tail.splitlines() if l.startswith('RESULT_XZ85 '));d=json.loads(lzma.decompress(base64.b85decode(blob)))
assert d['complete'] and not d['failures'] and d['counts']['official']==(192 if a.page_ring else 224) and d['counts']['scored']==(288 if a.page_ring else 336),d
assert d['counts']['profile']==11,d['counts']
meta=json.loads((o/'manifest.json').read_text());points=[]
targets={'F':('1x1f',),'G':('16x1f','32x1f'),'P':('1x8f',),'H1':('1x8f','16x8f','32x8f'),'H2':('1x8f','16x8f','32x8f')}
if a.page_ring:
    targets={name:('16x1f','32x1f') for name in ('G','R4','R8')}
    meta['G']=json.loads((H/'roof/manifest.json').read_text())['G']
    targets['D']=('16x8f','32x8f')
    meta['D']=json.loads((H/'pvsum/manifest.json').read_text())
for name,shapes in targets.items():
    for shape in shapes:
        z=d['summary'][shape]['ratios'][name]
        patch=H/'pvsum/candidate.patch' if name=='D' else (H/'roof' if a.page_ring and name=='G' else o)/(name+'.patch')
        row=dict(arm=name,shape=shape,**z,seeds=d['seeds'],nominated=z['r']<1,patch=str(patch.relative_to(R)),**meta[name]);points.append(row)
out=dict(receipt=str(a.receipt),receipt_sha256=hashlib.sha256(a.receipt.read_bytes()).hexdigest(),result=d,points=points,
    policy='Stage points nominate; no standalone G0 claim from six full shapes. P4 composes and checks paired six-load R on held-out seeds.')
(o/'result.json').write_text(json.dumps(out,indent=2)+'\n')
(o/'P4_HANDOFF.json').write_text(json.dumps(dict(base='60254883',receipt=str(a.receipt),points=[p for p in points if p['nominated']],policy=out['policy']),indent=2)+'\n')
for shape,row in d['summary'].items():print(shape,'A',row['control_us'],'logicalTB/s',row['bytes']/row['control_us']/1e6,'ratios',row['ratios'])
print('resources',d['resources']);print('profiles',d['profiles']);print('nominations',[p for p in points if p['nominated']])
