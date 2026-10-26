"""Readable per-shape oracle allocations and every measured arm/CI."""
import csv,json,math,statistics
from pathlib import Path
H=Path(__file__).resolve().parent
r=json.loads((H/'joint_results.json').read_text());p=json.loads((H/'packed_results.json').read_text());w=json.loads((H/'final_whole_results.json').read_text())
def fmt(x):return f'{x[0]:.4f} [{x[1]:.4f}, {x[2]:.4f}]'
out=['# Large-shape oracle split (microseconds,95% CI)','',f'H200 `{r["job"]}`;8pairedseeds/8cold rounds. The three interacting interventions are allocated by the three-axis Shapley average. They sum to RAW minus the fully stripped twin for each seed. This is an operational oracle allocation, not a unique physical decomposition.','', '| T | Projection | activation TMA/barrier | tensor issue/sync | epilogue | residual |','|---|---|---|---|---|---|']
for row in r['oracle']:out.append('| '+str(row['t'])+' | '+row['role']+' | '+' | '.join(fmt(row['Shapley'][k]['us']) for k in ('activation_TMA_barrier','tensor_issue_and_sync','epilogue','residual'))+' |')
out+=['', 'Residual includes weight issue/transport, prologue, initial activation tiles, checksum work, synchronization and the timing boundary. The inner-event S19 measurements run3.16–3.79us above the outer-event/empty-graph-subtracted S19 controls in the second phase. These phases use different input seeds; this descriptive offset is not a measured calibration to subtract. Thus residual is not a physical HBM time or a direct explanation of S19’s gap. Intervention differences stay inside one timing protocol.','', 'Math-free twins retain every vector weight load and activation TMA (SASS proof in native_validation.json), consume weights through XOR/checksum stores, and retain CTA synchronization so the producer warp cannot lap mbarrier phases. Dynamic shared padding matches active blocks to the full twin. Their checksums and replacement synchronization limit attribution to tensor issue. Shared-activation twins reuse two real tiles and remove repeated TMA/barrier phases; they are synthetic bounds, never candidates.','', 'The raw schedule requests2–4activation bytes from L2/shared transport per weight byte. Its T256 QKV/pool128-token tiling requests the complete weight bank twice; physical HBM traffic depends on cache and is unmeasured.','', '# Direct current-S19 activation L2 oracle','', '| T | Projection | hot-X/S19 time [95% CI] |','|---|---|---|']
for row in r['oracle']:out.append(f'| {row["t"]} | {row["role"]} | {fmt(row["ratios"]["S19_hotX"]["vs_S19"])} |')
out+=['', 'The prefetch kernel is excluded from this oracle interval. No production speedup is inferred from that exclusion. Current S19 activation-cache warming is neutral or small; the largest synthetic savings do not establish a large activation-cache bottleneck in the optimized kernels. Opaque Lt kernels have no exact source math-free twin.','', '# Every oracle arm','']
for row in r['oracle']:
 out += [f'## T{row["t"]} {row["role"]}','', '| Arm | time/S19 [95% CI] | time/RAW [95% CI] |','|---|---|---|']
 for a,v in row['ratios'].items():out.append(f'| {a} | {fmt(v["vs_S19"])} | {fmt(v["vs_RAW"])} |')
 out.append('')
 if row['neighbour_ratios']:
  out += ['Actual S19 neighbour included:', '', '| Arm | time/S19 [95% CI] |', '|---|---|']
  for a,v in row['neighbour_ratios'].items():out.append(f'| {a} | {fmt(v["vs_S19"])} |')
  out.append('')
(H/'oracle_split.md').write_text('\n'.join(out).rstrip()+'\n')
rows=[]
for family,data in (('shared_weight',r['candidate']),('K_major',p['rows'])):
 seen=set()
 for row in data:
  key=row['t'],row['role'],row['kind']
  if key not in seen:
   a=row['AA'];rows.append(dict(family=family,t=row['t'],role=row['role'],kind=row['kind'],arm='AA',ratio=a[0],CI_low=a[1],CI_high=a[2],A_us=row['A_us'],B_us=''));seen.add(key)
  a=row['ratio'];rows.append(dict(family=family,t=row['t'],role=row['role'],kind=row['kind'],arm=json.dumps(row['arm']),ratio=a[0],CI_low=a[1],CI_high=a[2],A_us=row['A_us'],B_us=row['B_us']))
with (H/'all_stage_ratios.csv').open('w',newline='') as f:
 fields=['family','t','role','kind','arm','ratio','CI_low','CI_high','A_us','B_us'];writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(rows)
out=['# Structural changes at every large shape','', 'Ratios are candidate/S19 time with95%paired t7 log intervals. O/pool/unpool include their actual S19 neighbour; QKV is producer-only. Best point per cell is selection evidence. All arms are in all_stage_ratios.csv and the raw joint/packed result JSON.','', '| T | Projection | Shared weights/WGMMA | K-major weight bank |','|---|---|---|---|']
for a,b in zip(r['primary'],p['best']):assert (a['t'],a['role'])==(b['t'],b['role']);out.append(f'| {a["t"]} | {a["role"]} | {fmt(a["ratio"])} | {fmt(b["ratio"])} |')
out += ['', '# Same geometry: K-major/row-major producer time','', 'BT/BW/BK/stages/warps/K-split and accumulation are held fixed; only the exact physical weight layout/TMA address changes. Columns are K-split counts. All24interval upper bounds are below1.','', '| T | Projection | S2 | S3 | S4 |','|---|---|---|---|---|']
for t in (128,256):
 for role in ('qkv_gate','o','pool_router','unpooler'):
  cell=[x for x in p['rows'] if x['t']==t and x['role']==role and x['kind']=='producer' and 'packed_vs_rowmajor_same_geometry' in x];by={x['arm'][2]:x['packed_vs_rowmajor_same_geometry'] for x in cell};out.append(f'| {t} | {role} | '+' | '.join(fmt(by[s]) for s in (2,3,4))+' |')
out += ['', 'The improvement transfers poorly to the native S19 configuration: the dedicated whole-layer NATIVE_PACK arm is inconclusive. A bank-layout gain inside a slower schedule is not a promotion.']
(H/'stage_ratios.md').write_text('\n'.join(out)+'\n')
out=['# Held-out layer judgment against S19','',f'H200 `{w["job"]}`,16disjoint held-out seeds,fresh weights/router,960layer TASK +48strict native-pack projection/router checks. Tload=46sliding+16full; each seed retains six-load covariance. G0 is lower95%>1 across the six-load geomean.','', '| Fixed arm | R [95% CI] | G0 |','|---|---|---|']
for a,v in w['arms'].items():out.append(f'| {a} | '+f'{v["R"][0]:.6f} [{v["R"][1]:.6f}, {v["R"][2]:.6f}] | {v["G0"]} |')
out += ['', '| Shape (batch,q_len,window) | SS/S19 time | PS/S19 time | native-pack/S19 time |','|---|---|---|---|']
for key,v in w['shapes'].items():out.append('| '+key+' | '+' | '.join(fmt(v[a]) for a in ('SS','PS','NATIVE_PACK'))+' |')
out += ['', '| Load | native-pack R [95% CI] |','|---|---|']
for key,v in w['loads'].items():out.append('| '+key+' | '+fmt(v['NATIVE_PACK']['R'])+' |')
out += ['', f'Combined experimental prepare{w["raw"]["prepare_s"]:.6f}s; measured additions{w["raw"]["added_prepare_s"]}. The optional minimal NP patch drops SS/PS banks; its isolated prepare and its S20 composition remain unmeasured. The NATIVE_PACK arm has no changed T256 code; movement there receives no mechanism attribution.','', 'No new candidate passes G0. The T128 native-layout patch is an unpromoted option for an already-planned P4 batch, with no separate confirmation requested.']
(H/'whole_ratios.md').write_text('\n'.join(out)+'\n')
