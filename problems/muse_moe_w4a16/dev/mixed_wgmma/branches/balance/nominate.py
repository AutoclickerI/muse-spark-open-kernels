"""Hand P4 all valid per-shape point gains, retaining both control contrasts."""
import hashlib
import json
from pathlib import Path
H=Path(__file__).resolve().parent;ROOT=H.parents[5]
r=json.loads((H/'results.json').read_text());mf=json.loads((H/'manifest.json').read_text());assert r['native_H200']
nominations=[];choices={}
for m in (128,256):
 rows=[x for x in r['arms'] if x['M']==m and x['arm']!='CH4'];best=min(rows,key=lambda x:x['vs_CH4']['mean'])
 choices[m]=best['arm'] if best['vs_CH4']['mean']<1 else 'CH4'
 for row in rows:
  if not row['P4_point_nomination']:continue
  arm=row['arm'];patch=H/(arm.lower()+'.patch');assert hashlib.sha256(patch.read_bytes()).hexdigest()==mf[arm]['patch_sha256']
  nominations.append(dict(**row,patch=str(patch.relative_to(ROOT)),patch_sha256=mf[arm]['patch_sha256'],candidate=mf[arm]['candidate'],candidate_sha256=mf[arm]['candidate_sha256'],base_sha256=mf['base_sha256'],fixtures_per_seed=3))
out=dict(receipt=r['receipt'],base='problems/muse_decode_layer/cand_r3_best7_s15.py',base_sha256=mf['base_sha256'],current_component_control='CHUNK4, already in S17/S18',nominations=nominations,default_point_choice=choices,AA=r['AA'],integration='Root-applicable S15 patches append a complete NF cell and compile hook. P4 should preserve its dependent W2/finalizer shell while composing route/producer ownership. All branches read actual C except NT1, which reads the masked LOW vector; N16 uses the matching task prefix. No activation bank/layout changes.',screen='All five exact candidates normal+forced local_eval, 966 current-input replay TASK checks, native sm90a compile and exact packed16-seed rehearsal; H200 receipt validates real WGMMA and all arms/seeds.',promotion='Every valid per-shape point gain against either control is nominated with both descriptive seed CIs. The default choice minimizes the direct paired CH4 ratio. P4 confirms held-out six-load geomean R lower bound>1. A:A is diagnostic, not a veto.')
(H/'P4_HANDOFF.json').write_text(json.dumps(out,indent=2)+'\n')
print('P4 choices',choices,'nominations',[(x['M'],x['arm']) for x in nominations])
