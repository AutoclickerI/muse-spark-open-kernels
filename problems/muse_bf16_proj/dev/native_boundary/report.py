"""Human-readable native-boundary receipts; physical claims stay separate from bounds."""
import json,statistics
from pathlib import Path
H=Path(__file__).resolve().parent
def ci(x):
 if isinstance(x,dict):x=[x['R'],*x['CI95']]
 return f'{x[0]:.6f} [{x[1]:.6f},{x[2]:.6f}]'
sized=json.loads((H/'sizing_results.json').read_text())
out=['# S19 native sizing before building','','H200 `'+sized['job']+'`: eight fresh seeds,16 symmetric A:A rounds,192published layer TASK. CUPTI observes separate replays without inserted device work. The fit is actual selected bank bytes/4.61419582TB/s+2.876us. Sizes below use observed completion contributions; raw PDL residence is not independently removable. These are scenarios, not physical bounds or promotion CIs.','', '| Term | six-load R scenario [sampling95%CI] | d1 / d16 / d32 / v1 / v16 / v32 removed us/layer |','|---|---|---|']
for label,z in sized['terms'].items():out.append('| '+label+' | '+ci(z['R'])+' | '+' / '.join(f'{l["removed_us_per_layer"][0]:.3f}' for l in z['loads'])+' |')
out+=['','The large upper scenario includes complete b1/route/b2 removal. Producer-only modeled surplus is12.187/32.504us at v16/v32; complete consumers add14.330/20.131us. Split-only reduction is measured later. O16F103.818us residence has6.076us exposed contribution; gate32F148.398us residence has2.657us exposed. Pure residence-minus-stream sums would invent savings.','', '| Shape | Site | Encoding | bank bytes | S | Residence us [95%CI] | Exposed us [95%CI] | stream-fit us |','|---|---|---|---|---|---|---|---|']
for x in sized['rows']:out.append('| '+str(x['case'])+' | '+x['role']+' | '+x['encoding']+' | '+str(x['weight_bytes'])+' | '+str(x['splits'])+' | '+ci(x['P_us_CI'])+' | '+ci(x['critical_us_CI'])+f' | {x["roof_us"]:.6f} |')
out+=['','Logical allocation bytes include selected metadata/padding; physical HBM counters are unmeasured. Input/output traffic, tensor peaks, scalar throughput and extra prefetch overlap are not priced. The stream fit is a target, not a universal bank lower bound. Completion-path attribution and model sampling CIs cannot validate a counterfactual schedule. The9ab81a3f physical split remains unmeasured.','', 'Observer control: separate profiled whole-event / unprofiled A means. No scaling is applied to timestamps.','', '| Shape | Profiled/A |','|---|---|']
for row in sized['raw']['rows']:
 ratio=statistics.mean(p['event_ticks'] for pair in row['profiles'] for p in pair)/statistics.mean(statistics.mean(x['A']) for x in row['times']);out.append('| '+str(row['case'])+f' | {ratio:.6f} |')
(H/'SIZING.md').write_text('\n'.join(out)+'\n')
o=json.loads((H/'oracle_results.json').read_text());out=['# One-boundary native S19 oracle','','H200 `'+o['job']+'`: eight fresh seeds,16 palindromic rounds retained and averaged,320published layer TASK checks. Native kernels and PDL are untouched except the declared deletions; no added timed GPU nodes. These are lab-only bounds, never candidates. Whole CIs use tools/promotion.interval with larger primary/A:A log SE.','', '| Arm | six-load R [95% CI] |','|---|---|']
for a,v in o['whole'].items():out.append('| '+a+' | '+ci(v['R'])+' |')
out+=['', '| Load | Producer deletion saved us [95%CI] | Complete-consumer deletion saved us [95%CI] | Split-only saved us [95%CI] |','|---|---|---|---|']
for load,v in o['loads'].items():out.append('| '+load+' | '+' | '.join(ci(v[a]['saved_us_per_layer']) for a in ('P_READY','C_READY','split_only'))+' |')
out+=['', 'P_READY and SINGLE_READY delete the same producers using same-seed real-S19 buffers; SINGLE_READY pre-sums the original partials outside timing. Their difference measures native split-consumer sensitivity after equal producer deletion, not free production summation. T256 already uses S1. C_READY deletes entire consumers; BOTH_READY tests combined deletion. Deletions alter dependency frontiers and remaining contention, so they are not exclusive hardware wait cycles. Inactive arms use measured unchanged AA; their timings remain explicit.','', '| Shape | Arm | candidate/S19 time [95%CI] | saved us [95%CI] |','|---|---|---|---|']
for r in o['shapes']:
 for a,v in r['arms'].items():out.append('| '+str(r['case'])+' | '+a+' | '+ci(v['time_ratio'])+' | '+ci(v['saved_us'])+' |')
(H/'ORACLE.md').write_text('\n'.join(out)+'\n')
if (H/'tune_results.json').exists():
 t=json.loads((H/'tune_results.json').read_text());out=['# Native dispatch selection in the whole layer','','H200 `'+t['job']+'`: all arms use S19’s unchanged native WGMMA body, eight fresh seeds and16 symmetric rounds; every round retained. Ratios are changed-load layer time/S19, weighted46sliding+16full, with larger primary/A:A SE. These select points; the disjoint six-load G0 decides promotion.','', '| T | Site | Config BT/BW/BK/stages/warps/TMA | S | layer time/S19 [95%CI] | saved us/layer |','|---|---|---|---|---|---|']
 for r in t['arms']:out.append(f'| {r["t"]} | {r["role"]} | {r["config"]} | {r["split"]} | '+ci(r['time_ratio'])+f' | {r["saved_us"]:.3f} |')
 (H/'TUNING.md').write_text('\n'.join(out)+'\n')
