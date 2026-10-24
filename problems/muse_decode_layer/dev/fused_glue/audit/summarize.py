"""Bind the H200 rejection to launch geometry and local causal evidence."""
from pathlib import Path
import json,re,hashlib
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[4]
receipt=ROOT/'problems/muse_moe_w4a16/lab/profiles/lab_best7_fused_glue.80b70f7f20fa41c6.log'
rows=[]
for line in receipt.read_text().splitlines():
 m=re.match(r'G (\d+)x(\d+)(S|F) (\w+) A/B_us=([\d.]+)/([\d.]+) B:A=([\d.]+).*flips=(\d+)',line)
 if m:
  b,q=int(m[1]),int(m[2]);rows.append(dict(batch=b,q_len=q,tokens=b*q,layer=m[3],cell=m[4],
   control_us=float(m[5]),candidate_us=float(m[6]),paired_ratio=float(m[7]),added_us=float(m[6])-float(m[5]),route_flips=int(m[8])))
local=json.loads((HERE/'mechanism_local.json').read_text());assert len(rows)==32
lines=['# Why the first native glue mapping loses','',
  'H200 job80b70f7f20fa41c6 rejects every arm (1.51–17.63x whole-layer).',
  'The local SM120 reproducer isolates the same failure without MoE/attention.',
  'These are work-deletion diagnostics, never candidates. H200 phase attribution',
  'is inferred from this local causal test and the identical source mapping.','',
  '| M | Pool full -> no prologue us | Unpool full -> no boundary us | O full -> no boundary us |',
  '|---|---:|---:|---:|']
geometry=[]
for M in (1,8,16,32):
 pick=lambda name:next(x for x in local['rows'] if x['tokens']==M and x['mode']==name)
 p,u,o=pick('pool_prologue'),pick('unpool_boundary2'),pick('o_boundary1')
 tm=lambda r,k:r['timings_us'][k]
 lines.append(f'| {M} | {tm(p,"full"):.1f} -> {tm(p,"no_prologue"):.1f} | {tm(u,"full"):.1f} -> {tm(u,"no_epilogue"):.1f} | {tm(o,"full"):.1f} -> {tm(o,"no_epilogue"):.1f} |')
 geometry.append(dict(tokens=M,pool_ctas=576*((M+7)//8),pool_full_row_recomputations=576*M,
  unpool_ctas=(64 if M<=8 else 512)*((M+7)//8),last_epilogue_ctas=(M+7)//8,
  active_norm_warps_per_epilogue=4 if M>1 else 4,warp_serial_row_elements=256 if M>1 else 64,
  token_rows_per_final_cta=8 if M>=8 else 1))
lines+=['',
  'The initial grid is not short: pool launches576/576/1152/2304 CTAs at M1/8/16/32.',
  'It redundantly recomputes the entire8192-column boundary576 times per token:',
  '72 output-row tiles x8 K splits. At M32 that is18,432 full-row scans.',
  'One warp handles each token with256 serial column iterations; only one warp',
  'does useful norm work at M1, and the prologue repeats before every matrix tile.',
  'Deleting that prologue removes81–95% of its local total cost.','',
  'The opposite failure occurs after GEMM: only1/1/2/4 last CTAs survive for',
  'M1/8/16/32, regardless of the original grid size. Each owns up to8 full rows.',
  'At M>=8 four warps each process two token rows through repeated256-iteration',
  'scalar scans; at M1 the special whole-CTA path still uses serial64-element',
  'per-thread scans. Deleting the boundary removes74–95% of O/unpool time.',
  'The pool routing epilogue itself costs roughly49–82us locally, rather than',
  'the few-us parallel Triton route consumer it replaced.','',
  'PDL on/off changes these local stage times by at most about3%; no source spin',
  'exists. Each losing CTA returns after its acq/rel arrival. The SM90a build has',
  'zero register spills. Source review therefore excludes an unresident-grid',
  'barrier and PDL launch ordering as the cause of the large penalty. The matrix',
  'mainloop remains slower than the incumbent too (skinny8-token tiles, scalar',
  'shared-fragment loads); deleting glue does not turn it into a measured win.','',
  'This is a performance mapping defect, not a numerical/launch correctness bug.',
  'It closes this replicated-row prologue / single-CTA scalar epilogue mapping.',
  'It does not establish a bound against fusion. A replacement must keep each',
  'norm cooperative and avoid repeating a whole-row reduction for every output/K',
  'tile; routing metadata should be produced once and consumed by the expert grid.',
  'No replacement projection mapping or remote run was built from this diagnosis.',
  '', 'Reproduce locally with',
  '`CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_decode_layer/dev/fused_glue/audit/diagnose.py`,',
  'then run `python problems/muse_decode_layer/dev/fused_glue/audit/summarize.py`.']
result=dict(h200_job='80b70f7f20fa41c6',h200_rows=rows,geometry=geometry,local=local,
 source_sha256=local['source_sha256'],receipt_sha256=hashlib.sha256(receipt.read_bytes()).hexdigest(),
 conclusion='Redundant full-row prologue and under-parallel scalar last-CTA epilogues dominate; not a spin or PDL issue.',
 limitation='Local ablations infer the H200 mechanism; no H200 per-phase counters were collected.')
(HERE/'mechanism.json').write_text(json.dumps(result,indent=2)+'\n');(HERE/'README.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines[:13]))
