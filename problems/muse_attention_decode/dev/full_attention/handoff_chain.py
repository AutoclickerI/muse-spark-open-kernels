"""Freeze all admitted stage points and the concrete selected S19 patch identities."""
import hashlib,json
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parents[3];o=H/'chain';d=json.loads((o/'result.json').read_text());assert d['measured'];summ=d['summary']
lines=['# All paired arm points','',f"Native job `{d['job']}`;928TASK;8seeds5671243+7919*i,8cold rounds.",'New/S19 time,95%t7 CI,SE across eight paired seed ratios. A/A2 identical;','S is an identity control except the two actual FA3 calls. D controls validate','the native exported launcher against M at the same split count. No G0 claim.','', '| Shape | Arm | Ratio [95% CI] | SE | Native kernels |','|---|---|---|---|---|']
for shape,v in summ.items():
 for arm,z in v['ratios'].items():lines.append(f"| {shape} | {arm} | {z['r']:.6f} [{z['lo']:.6f},{z['hi']:.6f}] | {z['se']:.6f} | {d['launches'][shape][arm]} |")
lines+=['','Matching recipe comparisons:','','| Shape: new/control | Ratio [95% CI] | SE |','|---|---|---|']
for k,z in d['pairs'].items():lines.append(f"| {k} | {z['r']:.6f} [{z['lo']:.6f},{z['hi']:.6f}] | {z['se']:.6f} |")
(o/'POINTS.md').write_text('\n'.join(lines)+'\n')
hand=dict(base='S19 shipped8b415553, expanded62ea5bf4',base_sha256=d['result']['base_sha256'],job=d['job'],receipt=d['receipt'],TASK=d['result']['counts'],seeds=d['result']['seeds'],controls={k:v['ratios']['A2'] for k,v in summ.items()},points=[p for p in d['points'] if p['nominated']],preferred=d['preferred'],pairs=d['pairs'],launches=d['launches'],all_points=str((o/'POINTS.md').relative_to(R)),policy='Per-shape points nominate only; P4 confirms held-out six-load G0. D and S graph aliases are controls, not new shader wins. Identity-S points are not changed pieces.',correction='S19 v1 sliding is native sliding_main<8,4,128>, one launch, not FA3. Explicit FA3 replacements all lose; preserve P5 native path.',limits='A:A1x8s .982109 CI[.970057,.994161] drift disclosed; actual FA3 call controls cover1. No per-load veto. End-to-end prepare cost remains P4 check.',composition='candidate.patch is combined nomination. fa.patch, vq.patch and schedule_selected.patch each apply independently onto S19; do not concatenate prepare wrappers. v1F R64 replaces prior M64 selection; remove an earlier direct _p5_phase_call M64 interceptor when composing.',scope='No P5 query setup, native sliding or RAW2/merge edits. Only FA3 launch structure and selected full VQ8 separate merge.')
for stem in ('candidate','fa','vq','schedule_selected'):
 artifact=json.loads((o/(stem+'_manifest.json')).read_text());assert artifact['base_sha256']==hand['base_sha256']
 for ext,k in [('py','source_sha256'),('patch','patch_sha256')]:assert hashlib.sha256((o/(stem+'.'+ext)).read_bytes()).hexdigest()==artifact[k]
 hand[stem]=dict(**artifact,patch=str((o/(stem+'.patch')).relative_to(R)))
admission={}
for mode in ('normal','forced'):
 p=o/('candidate_'+mode+'_screen.json')
 if p.exists():
  r=json.loads(p.read_text());assert all(x['ok'] for x in r['tests']+r['static']);assert r['sha256']==hashlib.sha256((o/'candidate_attention.py').read_bytes()).hexdigest()
  admission[mode]=dict(candidate_sha256=hand['candidate']['source_sha256'],adapter_sha256=r['sha256'],fixtures=len(r['tests']),eager_TASK=sum(x['eager_ok'] for x in r['tests']),poisoned_replay_TASK=sum(x['ok'] for x in r['tests']),static=len(r['static']))
hand['candidate_admission']=admission
(o/'P4_HANDOFF.json').write_text(json.dumps(hand,indent=2)+'\n');(o/'candidate_admission.json').write_text(json.dumps(admission,indent=2)+'\n')
phases={}
for k,rows in d['profiles'].items():
 latest=0;out=[]
 for r in rows:
  inc=max(0,r['end_us']-latest);latest=max(latest,r['end_us']);out.append(dict(name=r['name'],duration_us=r['end_us']-r['start_us'],critical_increment_us=inc,grid=r['grid'],block=r['block'],regs=r['regs']))
 phases[k]=out
(o/'critical_phases.json').write_text(json.dumps(phases,indent=2)+'\n')
print('P4',hand['candidate']['patch'],hand['candidate']['patch_sha256'][:8],list(admission))
