"""Summarize both H200 receipts without guessing individual aggregated call times."""
from pathlib import Path
import json,re
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).parent
PROFILES=ROOT/'problems/muse_decode_layer/profiles'
a=json.loads((PROFILES/'cand_r3_v9.0e94268cf5654012.json').read_text())
b=json.loads((PROFILES/'cand_r3_best5.691bff0bcff44e99.json').read_text())
def table(x):
 out=[]
 for profile in x['profiles']:
  d=dict(case=profile['case'],projection=0.,attention=0.,moe=0.,glue=0.,scheduler=0.,misc=0.,glue_kernels=[])
  for line in profile['report'].splitlines():
   m=re.match(r'(.+?)\s+([\d.]+) µs\s+[\d.]+%\s+(\d+)$',line)
   if not m:continue
   name,us,n=m.groups();us=float(us)
   if 'moe_fused' in name:kind='moe'
   elif name.startswith(('_gemv_kernel','nvjet','void cublasLt::splitKreduce')):kind='projection'
   elif 'prepare_varlen_num_blocks' in name:kind='scheduler'
   elif any(v in name for v in ('attn_main','attn_combine','flash::enable_sm90','FlashAttnFwdCombine')):kind='attention'
   elif name.startswith(('_boundary','_route_norm','_qk_prep','_norm_gate')):kind='glue'
   else:kind='misc'
   d[kind]+=us
   if kind in ('glue','scheduler','misc'):d['glue_kernels'].append(dict(name=name.strip(),us=us,calls=int(n),category=kind))
  out.append(d)
 return out
old,new=table(a),table(b)
lines=['# Best5 H200 stage receipt','',
 'Board control: `0e94268cf5654012`, **1.427904x**. Best5: `691bff0bcff44e99`, **1.353449x**, 13/13 tests, prepare **158.992 s**. Times below are aggregate profile GPU durations per layer, in µs; dependency waits may overlap preceding compute. The organizer aggregates repeated kernels, so individual boundary-call times are unavailable.', '',
 '| Load | Layer | Projection | Attention | MoE | Glue | Scheduler | Misc reduce/cast | Attention control → best5 |',
 '|---|---|---:|---:|---:|---:|---:|---:|---:|']
for x,y in zip(old,new):
 assert x['case']['load']==y['case']['load'] and x['case']['layer']==y['case']['layer']
 lines.append('| '+y['case']['load']+' | '+y['case']['layer']+' | '+' | '.join(f'{y[k]:.1f}' for k in ('projection','attention','moe','glue','scheduler','misc'))+f' | {x["attention"]:.1f} → {y["attention"]:.1f} |')
lines+=['',
 'Attention includes its main and combine kernels. Scheduler is the separate FA3 schedule launch. Glue includes boundary, routing/norm, QK preparation and norm/gate. Misc is the remaining Torch reduction/cast/clear work; the truncated profile symbols do not identify every individual operation.', '',
 'The verify-32 full FA3 main kernel regressed **377.9 → 501.5 us** (+123.6 us, +32.7%), while that shape’s SGLang baseline changed **1113.06 → 1095.01 us** (-1.6%). Best5 selected explicit `(32, True)` from one old synthetic draw at 386.432 us. The old sampler already used ±7.5% lengths at B>1; its one fixed draw, cache/input distribution and explicit scheduler baseline were the mismatch, rather than the spread envelope alone.', '',
 'The board q8 FA3 call forwards `data["scheduler_metadata"]`. Public generate_input returns **None at q8** (FA3 schedules inside its call), and builds shared scheduler metadata only for full q1. The previous best5/best6 `layer` alias instead reached explicit `(0, None)` metadata construction; the revised control restores the board call.', '',
 'MoE routing variance explains much of the overall score difference: sliding verify-16 **258.9 → 340.6 us**, sliding verify-32 **477.0 → 527.2 us**, full verify-32 **449.5 → 530.3 us**. Best5 chose NT1 at M32/128 and NT2 at M256; all native gates passed (~0.17% L2). These are different per-run routers, so none is a matched-route MoE regression measurement.', '',
 'Holding every other stage fixed, reclaiming 123.6 us x 16 on verify-32 would imply about **+0.7%** on the six-load geometric score. This is an accounting estimate, not a measured gain from the revised candidate.']
(HERE/'BEST5_STAGES.md').write_text('\n'.join(lines)+'\n')
(HERE/'stages.json').write_text(json.dumps(dict(control=old,best5=new),indent=2)+'\n')
inventory=['# Best5 non-compute kernel inventory','',
 'From H200 receipt `691bff0bcff44e99`. Every profile row outside projections, attention and MoE is listed below. Times are aggregate GPU durations in µs per layer; PDL waits can overlap producer compute. Two boundary calls share one aggregate row. Names truncated by the organizer are kept verbatim.', '']
for row in new:
 inventory += ['## '+row['case']['load']+' '+row['case']['layer'],'',
               '| Kernel | Calls | Aggregate us | Category |','|---|---:|---:|---|']
 for kernel in row['glue_kernels']:
  inventory.append(f'| `{kernel["name"]}` | {kernel["calls"]} | {kernel["us"]:.1f} | {kernel["category"]} |')
 inventory.append('')
(HERE/'BEST5_GLUE.md').write_text('\n'.join(inventory))
print('\n'.join(lines[:7]+lines[7:19]))
