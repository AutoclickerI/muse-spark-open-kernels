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
        if a.page_ring and name in ('R4','R8','D'):
            row['source_sha256']=row.get('source_sha256',row.get('candidate_sha256'))
            cp=o/(name+'.cuda.patch');row.update(cuda_patch=str(cp.relative_to(R)),cuda_patch_sha256=hashlib.sha256(cp.read_bytes()).hexdigest())
out=dict(receipt=str(a.receipt),receipt_sha256=hashlib.sha256(a.receipt.read_bytes()).hexdigest(),result=d,points=points,
    policy='Stage points nominate; no standalone G0 claim from six full shapes. P4 composes and checks paired six-load R on held-out seeds.')
(o/'result.json').write_text(json.dumps(out,indent=2)+'\n')
handoff=dict(base='60254883',receipt=str(a.receipt),points=[p for p in points if p['nominated']],policy=out['policy'])
if a.page_ring:
    handoff.update(preferred={shape:min((p for p in points if p['shape']==shape and p['nominated']),key=lambda p:p['r'])['arm']
                              for shape in targets['R8']+targets['D'] if any(p['shape']==shape and p['nominated'] for p in points)},
        TASK_checks=sum(d['counts'].values()),A_A={k:v['ratios']['A2'] for k,v in d['summary'].items()},
        controls_note='Verify16 A:A .996655 CI excludes1; retain drift. Confirm held-out whole-layer G0.',
        integration_note='Root patches apply onto exact S15 60254883; source-only CUDA hunks permit preserving S18 PDL and V-wait removal. Select R8 only decode16/32full, D only verify16/32full. No changed dispatch for d1/v1.')
    if (o/'S18_portability.json').exists():handoff['S18_ports']=json.loads((o/'S18_portability.json').read_text())
(o/'P4_HANDOFF.json').write_text(json.dumps(handoff,indent=2)+'\n')
for shape,row in d['summary'].items():print(shape,'A',row['control_us'],'logicalTB/s',row['bytes']/row['control_us']/1e6,'ratios',row['ratios'])
print('resources',d['resources']);print('profiles',d['profiles']);print('nominations',[p for p in points if p['nominated']])
