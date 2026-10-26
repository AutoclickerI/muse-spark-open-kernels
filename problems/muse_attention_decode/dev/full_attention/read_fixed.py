"""Read TASK-gated structural points; oracle savings are never promotions."""
import argparse,base64,hashlib,json,lzma,statistics,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];o=H/'fixed'
ap=argparse.ArgumentParser();ap.add_argument('receipt',type=Path);a=ap.parse_args()
sys.path.insert(0,str(R/'tools'));from lab_packet import result_lines,read_packets
receipt=json.loads(a.receipt.read_text());tail=receipt.get('result',{}).get('log_tail','')
meta=json.loads((o/'lab_manifest.json').read_text())
for packet in read_packets(tail):
    if packet['name']=='fixed_attention':assert packet['source_sha256']==meta['payload_sha256'][:16]
rows=[l[12:] for l in result_lines(tail) if l.startswith('RESULT_XZ85 ')]
if not rows:
    out=dict(receipt=str(a.receipt),measured=False,host_status=receipt.get('status'),reason='No complete TASK-gated result; LAB_ERROR/timeout is unmeasured',log_tail=tail)
    (o/'result.json').write_text(json.dumps(out,indent=2)+'\n');print(out);raise SystemExit(1)
d=json.loads(lzma.decompress(base64.b85decode(rows[0])))
assert d['complete'] and not d['failures'] and d['counts']==dict(official=84,scored=320,stress=160),d['counts']
assert d.get('graph_nodes') and all(n>0 for n in d['graph_nodes'].values()),'No native capture proof; unmeasured'
assert all(x>0 for rows in d['prepass_us'].values() for x in rows),'Invalid timer'
meta=json.loads((o/'lab_manifest.json').read_text());points=[];oracles=[];preferred={}
for shape,record in d['summary'].items():
    candidates=[]
    for arm,z in record['ratios'].items():
        if arm=='A2':continue
        row=dict(arm=arm,shape=shape,**z,seeds=d['seeds'])
        if arm=='Ffree':row['ceiling_only']=True;oracles.append(row);continue
        role='last' if arm.startswith('L') else arm if arm.startswith('C') else 'F_vq'
        row.update(nominated=z['r']<1,kernel_sha256=meta['native'][role],kernel_patch=str((o/(role+'.cuda.patch')).relative_to(R)),driver='problems/muse_attention_decode/dev/full_attention/fixed_driver.py.inc')
        row['kernel_patch_sha256']=hashlib.sha256((o/(role+'.cuda.patch')).read_bytes()).hexdigest()
        if arm.startswith('L'):row['nk'],row['splits']=map(int,arm[1:].split('s'))
        points.append(row)
        if row['nominated']:candidates.append(row)
    if candidates:preferred[shape]=min(candidates,key=lambda p:p['r'])['arm']
controls={shape:record['ratios']['A2'] for shape,record in d['summary'].items()}
out=dict(receipt=str(a.receipt),receipt_sha256=hashlib.sha256(a.receipt.read_bytes()).hexdigest(),host_passed=receipt.get('result',{}).get('passed'),result=d,points=points,oracles=oracles,
    controls=controls,control_drift_shapes=[s for s,z in controls.items() if z['lo']>1 or z['hi']<1],measured=True)
(o/'result.json').write_text(json.dumps(out,indent=2)+'\n')
handoff=dict(base='S18 e04635ec + measured R8/D',recipe='Native all-key fixed reference (paid), full lastCTA, compact q1 history-ring reuse',receipt=str(a.receipt),TASK_checks=sum(d['counts'].values()),
 points=[p for p in points if p['nominated']],preferred=preferred,controls=controls,seeds=d['seeds'],
 candidate_builder='problems/muse_attention_decode/dev/full_attention/build_fixed_candidate.py',
 policy='Every changed point<1 is nominated; Ffree is an oracle, never shipped. P4 confirms held-out whole-layer paired six-load G0, with no per-load veto.')
if (o/'candidate_manifest.json').exists():
    handoff['concrete_C3_patch']=dict(**json.loads((o/'candidate_manifest.json').read_text()),patch=str((o/'candidate.patch').relative_to(R)),
        applies_onto='fixed/control.py = S18 e04635ec plus measured R8/D CUDA globals',runtime_note='Layer caller preserves the S18 sumcast; P4 confirms whole-layer G0. No standalone small-gain send.')
(o/'P4_HANDOFF.json').write_text(json.dumps(handoff,indent=2)+'\n')
for shape,row in d['summary'].items():print(shape,row)
print('prepass_us',d['prepass_us']);print('resources',d['q1_resources'],d['vq_resources']);print('preferred',preferred);print('AA drift',out['control_drift_shapes'])
