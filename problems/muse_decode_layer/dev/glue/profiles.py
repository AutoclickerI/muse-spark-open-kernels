"""Inventory every non-compute profile row without inventing unreported call times."""
from pathlib import Path
import json,re
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).parent
rows=[]
def inventory(p):
 x=json.loads(p.read_text());out=[]
 for profile in x.get('profiles',[]):
  case=profile['case'];row={'load':case['load'],'layer':case['layer']}
  for line in profile['report'].splitlines():
   m=re.match(r'(.+?)\s+([\d.]+) µs\s+[\d.]+%\s+(\d+)$',line)
   if not m:continue
   name,us,n=m.groups()
   if name.startswith(('_boundary','_route','_qk_prep','_norm_gate','void at::native::vectorized_elementwise','void flash::prepare_varlen')):
    row[name]={'us':float(us),'calls':int(n)}
  out.append(row)
 return out
base=ROOT/'problems/muse_decode_layer/profiles/cand_r3_v9.0e94268cf5654012.json'
rows=inventory(base)
lines=['# H200 glue inventory','',
 'Source: board receipt `0e94268cf5654012` (1.427904x). Times are **aggregate GPU durations per shape**, in µs; boundary times combine its two calls because the organizer profile does not separate them. PDL durations include dependency waits and can overlap the preceding kernel, so sums are not exposed critical-path latency.',
 '', '| Load | Layer | Residual/norm ×2 | Route + norm | QK prep | Norm/gate | Clear | FA3 scheduler | Glue sum* |',
 '|---|---|---:|---:|---:|---:|---:|---:|---:|']
for r in rows:
 def val(prefix):return sum(v['us'] for k,v in r.items() if k.startswith(prefix) and isinstance(v,dict))
 vals=[val(p) for p in ('_boundary','_route','_qk_prep','_norm_gate','void at::native::vectorized_elementwise','void flash::prepare_varlen')]
 lines.append('| '+r['load']+' | '+r['layer']+' | '+' | '.join(f'{v:.1f}' if v else '—' for v in vals)+f' | {sum(vals[:5]):.1f} |')
lines+=['','*Glue sum excludes FA3 scheduling (listed separately because it is attention-owned). cuBLAS split-K reduction and attention combine belong to their compute operations and are unchanged. A dash means that no separate launch appears in that profile; q1 native attention already contains its QK/cache preparation and norm/gate.', '',
 'Best5 SHA `a651af25cb44550932f497b57eb82fb32c0fbe07c1f21a8592bd0982f937c150` is still in `runs/waiting/1047930.json` at this turn’s one receipt check. No per-stage values exist yet; do not substitute best (`7b2165c8`) for best5.']
(HERE/'PROFILE.md').write_text('\n'.join(lines)+'\n');(HERE/'profile_board.json').write_text(json.dumps(rows,indent=2)+'\n')
print('\n'.join(lines))
