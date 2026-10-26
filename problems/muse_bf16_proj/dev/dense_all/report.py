"""Render all per-shape intervals and untimed bank costs from retained H200 data."""
import json
from pathlib import Path
H=Path(__file__).resolve().parent
r=json.loads((H/'results.json').read_text());rows=r['rows']
def fmt(v):return f'{v[0]:.4f} [{v[1]:.4f}, {v[2]:.4f}]'
out=['# Every dense projection: candidate/control time, paired 95% CI','',f'H200 `{r["job"]}`; eight paired seeds, eight cold alternating rounds. All arms pass the published projection checker. These are selection intervals. The selected layer candidate has an independent held-out G0 receipt.','', 'Small arms: 0 = S4/WK4/ST1; 1 = S2/WK4/ST1; 2 = S1/WK8/ST1; 3 = arm0 with first weight vectors before the dependency wait.','', 'Wide arms: 0 = full-token/S4/PF1; 1 = full-token/S2/PF1; 2 = BN64/S4/PF1; 3 = BN128/S4/PF2; 4 = arm0 with early weights; 5 = full-token/S8/PF1; 6 = full-token/S1/PF1. Register-A weight reuse, exact code, FP32 partials in every wide arm.','']
for role in ('qkv_gate','qkv3','gate','gate_apply','o','pool_router','unpooler'):
 kind='neighbour' if role in ('gate_apply','o','pool_router','unpooler') else 'producer';out += [f'## {role} ({kind})','', '| T | Active in layer | '+ ' | '.join('arm'+str(a) for a in range(7))+' |','|---|---|'+ '|'.join(['---']*7)+'|']
 for t in (1,8,16,32,128,256):
  cell=[row for row in rows if row['t']==t and row['role']==role and row['kind']==kind]
  if not cell:continue
  active='/'.join(cell[0]['active_at']) or 'inactive diagnostic';vals={x['arm']:fmt(x['ratio']) for x in cell};out.append('| '+str(t)+' | '+active+' | '+' | '.join(vals.get(a,'—') for a in range(7))+' |')
 out.append('')
out += ['Raw gate and inactive QKV3 producer controls use a diagnostic FP32-output Torch matmul. O/pool/unpool use their actual S18 consumers. Full QKV at T1 is inactive. Gate_apply at T8 is inactive. The complete raw-gate M128/256 measurements are included without implying those stages execute in the layer.','', '# Preparation seconds','', '| New bank | small GPU pack + geometries | wide GPU pack + split metadata |','|---|---:|---:|']
tm=r['added_prepare']
for role in ('qkv_gate','qkv3','gate','o','pool_router','unpooler','unpooler_slice'):
 small=sum(v for k,v in tm.items() if k=='small_pack_'+role or k.startswith("small_stream_('"+role+"',"));wide=sum(v for k,v in tm.items() if k=='wide_pack_'+role or k.startswith("wide_metadata_('"+role+"',"));out.append(f'| {role} | {small:.6f} | {wide:.6f} |')
out+=['',f'All-bank added total {tm["total"]:.6f} s; native compilation {sum(v for k,v in tm.items() if k.startswith("compile_")):.6f} s. Base S18 prepare {r["base_prepare_s"]:.6f} s. Each wide body is packed once and shared by split1/2/4/8; metadata is repartitioned on GPU.','', 'The held-out production gate bank adds 9.192027 s total; GPU pack + metadata 0.003525 s; S19 plus bank 170.454230 s. Compilation dominates the added cost.']
(H/'shape_ratios.md').write_text('\n'.join(out)+'\n')
first=json.loads((H/'first_whole/whole_results.json').read_text());confirm=json.loads((H/'whole_results.json').read_text());out=['# Layer changes: every shape and held-out G0','', 'Shape intervals are candidate/control time. Aggregate intervals are old/new time, using each paired seed’s six-load geometric mean and46sliding+16full layers per load. First batch has8selection seeds; the fixed gate1/sliding candidate has16disjoint confirmation seeds.','', '| Shape | Coding | Early weights | QKV+gate merge | Code+merge | Confirmed gate1 sliding |','|---|---|---|---|---|---|']
for key,cell in first['per_shape_candidate_control_time'].items():
 final=confirm['per_shape_candidate_control_time'][key]['CODE'];out.append('| '+key+' | '+' | '.join(fmt(cell[a]) for a in ('CODE','EARLY','MERGE','BOTH'))+' | '+fmt(final)+' |')
out+=['', '| Six-load arm | R [95% CI] | Held-out gate |','|---|---|---|']
for arm in ('CODE','EARLY','MERGE','BOTH'):out.append('| Selection '+arm+' | '+fmt(first['arms'][arm]['R'])+' | no |')
out.append('| Fixed gate1 sliding | '+fmt(confirm['arms']['CODE']['R'])+' | pass |');out+=['', 'The candidate changes only T1 sliding gate. Recorded movements in unchanged shapes are timing variation and receive no mechanism attribution. A:A confidence intervals and raw paired samples remain in the result JSON. The final P4 composition requires its own held-out gate.']
(H/'layer_ratios.md').write_text('\n'.join(out)+'\n')
pipe=json.loads((H/'pipeline_results.json').read_text());group=json.loads((H/'grouped_results.json').read_text());out=['# Wide changes, paired95% CI at every active site','', 'Candidate/control time includes the actual residual/norm or route/norm neighbour for O/pool/unpool. QKV is producer-only. Eight paired seeds per batch; all raw samples and every arm remain in the JSON. The overlap and grouped columns select their best coded arm in that batch; cross-batch differences are not paired estimates.','', '| T | Role | Decoder/MMA overlap vs layer | Code/raw same math | K128+adjacent grid vs layer | Grouped/serial same tile |','|---|---|---|---|---|---|']
for r in pipe['best_coded']:
 t,role=r['t'],r['role'];kind='producer' if role=='qkv_gate' else 'neighbour';eligible=[g for g in group['rows'] if g['t']==t and g['role']==role and g['kind']==kind and g['arm'][0] and g['arm'][1]];g=min(eligible,key=lambda x:x['ratio'][0]);out.append('| '+str(t)+' | '+role+' | '+fmt(r['ratio'])+' | '+fmt(r['coded_vs_raw_same_math'])+' | '+fmt(g['ratio'])+' | '+fmt(g['grouped_and_tile_order_vs_serial'])+' |')
out+=['', 'The grouped/serial ratio changes both K grouping and token tile order. Their separate contributions are unmeasured. All replacements lose to the existing layer; this experiment establishes no large-shape winner. The original FP32-router check is in the all36-cell batch; wide follow-ups validate the full BF16 hi/lo bank with the projection checker and are not a router-specific precision promotion.']
(H/'wide_ratios.md').write_text('\n'.join(out)+'\n')
