"""Read paired S19 attention points; a nonpositive/absent gate is unmeasured."""
import argparse,base64,hashlib,json,lzma,statistics,sys
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];o=H/'phases';sys.path.insert(0,str(R/'tools'))
from lab_packet import read_packets,result_lines
ap=argparse.ArgumentParser();ap.add_argument('receipt',type=Path);a=ap.parse_args();receipt=json.loads(a.receipt.read_text());tail=receipt.get('result',{}).get('log_tail','')
meta=json.loads((o/'lab_manifest.json').read_text())
for p in read_packets(tail):
 if p['name']=='phases_attention':assert p['source_sha256']==meta['payload_sha256'][:16]
lines=result_lines(tail);rows=[x[12:] for x in lines if x.startswith('RESULT_XZ85 ')]
if not rows:
 out=dict(measured=False,job=receipt['id'],reason='No complete native TASK-gated archive',errors=[x for x in lines if x.startswith('LAB_ERROR')]);(o/'result.json').write_text(json.dumps(out,indent=2)+'\n');print(out);raise SystemExit(1)
queue=json.loads((o/'queue.json').read_text());assert hashlib.sha256(a.receipt.with_suffix('.py').read_bytes()).hexdigest()==queue['wrapper_sha256']
d=json.loads(lzma.decompress(base64.b85decode(rows[0])));assert d['complete'] and not d['failures'] and d['counts']==dict(official=78,scored=280,stress=140)
assert d.get('profiles') and d.get('profile_names'),'No native CUPTI proof'
assert d['base_sha256']==meta['base_sha256'] and tuple(d['seeds'])==tuple(meta['seeds'])
assert all(n>0 for n in d['graph_nodes'].values()) and all(t>0 for samples in d['seed_us'].values() for row in samples for t in row.values())
controls={k:v['ratios']['A2'] for k,v in d['summary'].items()};points=[];preferred={}
for shape,v in d['summary'].items():
 for arm,z in v['ratios'].items():
  if arm=='A2':continue
  row=dict(arm=arm,shape=shape,**z,seeds=d['seeds'],nominated=z['r']<1)
  if arm in meta['native']:row['cuda_sha256']=meta['native'][arm]['sha256']
  if arm.startswith(('M','U')):row['num_splits']=int(arm[1:]);row['static_metadata']=arm.startswith('M')
  if arm.startswith('P'):row['splits'],row['cluster']=map(int,arm[1:].split('c'))
  points.append(row)
 wins=[p for p in points if p['shape']==shape and p['nominated']]
 if wins:preferred[shape]=min(wins,key=lambda p:p['r'])['arm']
# Compare constant and dynamic schedules under the same requested cap; prep identical. Actual dynamic split counts may differ.
pairs={}
for arm in ('M1','M8','M16','M32','M64','M128'):
 rr=[s[arm]/s['U64'] for s in d['seed_us']['1x8f']];r=statistics.mean(rr);se=statistics.stdev(rr)/len(rr)**.5
 pairs[arm+'_vs_U64']=dict(r=r,se=se,lo=r-2.364624251*se,hi=r+2.364624251*se)
profiles={}
for shape,rows in d.get('profiles',{}).items():
 profiles[shape]=[dict(name=d['profile_names'][v[0]],start_us=v[1]/1000,end_us=v[2]/1000,grid=v[3:6],block=v[6],smem=v[7],regs=v[8]) for v in rows]
assert d['graph_nodes']['1x8f:M64']==4 and d['graph_nodes']['1x8f:A']==5
assert not any('prepare_varlen_num_blocks' in x['name'] for x in profiles['1x8f:M64'])
assert any('prepare_varlen_num_blocks' in x['name'] for x in profiles['1x8f:A'])
out=dict(measured=True,job=receipt['id'],receipt=str(a.receipt),receipt_sha256=hashlib.sha256(a.receipt.read_bytes()).hexdigest(),result=d,controls=controls,points=points,preferred=preferred,metadata_pairs=pairs,profiles=profiles)
(o/'result.json').write_text(json.dumps(out,indent=2)+'\n')
hand=dict(base='S19 final expanded62ea5bf4, shipped'+meta['base_sha256'],job=receipt['id'],seeds=d['seeds'],TASK=sum(d['counts'].values()),controls=controls,points=[p for p in points if p['nominated']],preferred=preferred,patch=None,builder='build_phases_candidate.py',policy='Per-shape points nominate; P4 confirms composed held-out six-load G0. No independent per-load veto or stage-only promotion.')
for stem,key in [('m64','new_piece'),('candidate','preferred_combination')]:
 path=o/(stem+'_manifest.json')
 if path.exists():
  artifact=json.loads(path.read_text());assert artifact['base_sha256']==meta['base_sha256']
  hand[key]=artifact;hand[key+'_patch']=str((o/(stem+'.patch')).relative_to(R))
if (o/'candidate_admission.json').exists():hand['admission']=str((o/'candidate_admission.json').relative_to(R))
hand['additional_point_patches']={}
for stem in ('t16','u64'):
 path=o/(stem+'_manifest.json')
 if path.exists():hand['additional_point_patches'][stem]=dict(**json.loads(path.read_text()),patch=str((o/(stem+'.patch')).relative_to(R)),local_proof='All native+glue trial calls pass498TASK in both local modes and on H200; derived full-layer composition remains P4 confirmation.')
hand['control_note']='Largest A:A mean drift-.365% v16; CI excludes1. G0 uses A:A noise, no per-load veto. E losses7.6/22.2% remain much larger.'
hand['mechanism_note']='M64/U64 share requested cap64; actual default dynamic split count may differ. The gain combines metadata removal and fixed legal scheduling.'
(o/'P4_HANDOFF.json').write_text(json.dumps(hand,indent=2)+'\n')
for k,v in d['summary'].items():print(k,v)
print('metadata_vs_U64',pairs);print('resources',d['q1_resources']);print('preferred',preferred)
for k,v in profiles.items():
 if k.startswith('1x8'):print(k,[(x['name'][:120],round(x['end_us']-x['start_us'],3)) for x in v])
