"""Current evidence, including explicit pending times until a verified H200 receipt exists."""
import json
from pathlib import Path
D=Path(__file__).resolve().parent;read=lambda n:json.loads((D/n).read_text());q=read('queue.json');size=read('sizing.json');counts=read('counts.json');local=read('local_validation.json');leaf=read('leaf_validation.json');payload=read('payload.json');prod=read('production_plan.json');result=read('results.json') if (D/'results.json').exists() else None
def cell(v):return f"{v['mean']:.3f} [{v['CI95'][0]:.3f}, {v['CI95'][1]:.3f}]"
lines=['# S19 operand/issue oracle batch','',
       '**P3a active.** '+(f"Verified receipt {result['job']}." if result else f"H200 measurements pending: sender **{q['sender_pid']}**, wrapper **{q['wrapper_sha256'][:16]}**, MoE check at **2026-10-27 09:00 KST**. All five hosts had zero remaining scored/check quota; no judge job is assigned yet."),'',
       'Read first: moe_public_ports/P4_HANDOFF.json and MECHANISMS.md, including matched-ABI port losses; '
       'FINDINGS mixed_wgmma DIRECT/TRANS/WS/CTA2/CHUNK history. No donor code or decoded-weight/WGMMA '
       'producer-consumer design is copied. The operator register-only raw loads/bit-permuted decode are source '
       'mechanisms, not demonstrated small-M gains. This batch retains exact S19 math/copy/route/readiness code.', '',
       '## Arms and interpretation','',
       '| Arm | One change / comparison |', '|---|---|',
       '| BASE / AA | Exact S19 62ea5bf4, independent workspaces |',
       '| BASE_PAD | Exact S19 body with +8320B dynamic reservation; controls shared-bank reservation |',
       '| FRAG_SMEM | Synthetic BF16 fragments prelaid in two vector planes in an8KiB shared bank;keeps segment-cached Bf;compare BASE_PAD |',
       '| FRAG_REG | Pack-free register constants;removes activation gathers and their addressing/lifetime;loose oracle,not an exact load-latency split |',
       '| FRAG_JIT | Same synthetic shared bank,load only one K-group at a time;shorter Bf lifetime,more shared reads per tile;compare FRAG_SMEM |',
       '| ACC_CTRL / ACC_FREE | Same live accumulator-result sinks after every HMMA,all32 sites retained;FREE zeroes C each instruction,removing recurrence;wrong math |',
       '| PAD_CTRL / PAD_USEFUL | Same decode sinks retain all weight/decode work;USEFUL issues HMMA on only r of8 K-warps to realize the useful-N instruction budget;wrong math,not a legal narrow-N implementation |',
       '| L1 / L2 / L3 | Same prior sink-controlled decode / MMA+activation / epilogue-NOWAIT cuts,compiled in the same TU/cohort |',
       '| HIST | Minimal exact-BASE observer;per-CTA work,SM and finish records,observer tax timed |',
       '| LAZY | Correct original activation loads moved from segment entry to each K-group/tile;shorter Bf lifetime,more global reads;native leaf prototype only |','',
       'FRAG_SMEM/JIT use a repeated synthetic pattern,not a64KiB bank of the real activations. '
       'Their timed8KiB initialization and vector LDS remain. This is a pack-free/on-chip bound; real BF16 '
       'packing/publication and its shared capacity must be paid by a correct build. Oracle results are ignored. '
       'BASE/AA/BASE_PAD/ACC_CTRL/PAD_CTRL/HIST/LAZY are independently TASK-checked before/after every case. '
       'S19 loads BF16 bits directly into uint4/Bf; no activation conversion exists to remove. '
       'ACC_FREE originally compiled to one HMMA; paired live-result sinks fixed that before sending. '
       'Decode sinks use the current ping-pong reduction buffer, preserving its retirement boundary.', '',
       '## H200 us and confidence intervals','',
       'Eight matched seeds ×four symmetric all-round arithmetic means; Student-t95% across seed means. '
       'Same route/weight seeds as the preceding cohort,all264 CTAs,256threads,depth2 and tile order. '
       'Extra shared reservation is an explicit paired control. These are selection data,not held-out G0.', '',
       '| Arm | M1 | M8 | M16 | M32 |','|---|---:|---:|---:|---:|']
names=['BASE','AA','BASE_PAD','FRAG_SMEM','FRAG_REG','FRAG_JIT','ACC_CTRL','ACC_FREE','PAD_CTRL','PAD_USEFUL','L1','L2','L3','HIST','LAZY']
for name in names:
 vals=[cell(next(v for v in result['summary'] if v['M']==m and v['arm']==name)) for m in (1,8,16,32)] if result else ['pending']*4
 lines.append('| '+name+' | '+' | '.join(vals)+' |')
if result:
 lines+=['','| Conditional saving (us) | M1 | M8 | M16 | M32 |','|---|---:|---:|---:|---:|']
 for term in sorted({v['term'] for v in result['contrasts']}):lines.append('| '+term+' | '+' | '.join(cell(next(v for v in result['contrasts'] if v['M']==m and v['term']==term)) for m in (1,8,16,32))+' |')
 lines+=['','| Term | M8–32 modeled R [CI] |','|---|---:|']
 for v in result['ceilings']:lines.append(f"| {v['term']} | {v['M8_32_R']['R']:.5f} [{v['M8_32_R']['CI95'][0]:.5f}, {v['M8_32_R']['CI95'][1]:.5f}] |")
lines+=['','## Work counts and pre-receipt sizing','',
        'These counts reconstruct executed warp-HMMA from the retained per-CTA tile histogram ×8warps ×32 '
        'SASS sites,checked against768tiles per expert/token block. They are not NVPerf hardware counters. '
        'Useful equivalents scale by actual per-expert tokens;N<8 is not supported by this BF16 HMMA form.', '',
        '| M | Issued warp-HMMA mean | Useful equivalent mean | Useful fraction [95%CI] |', '|---|---:|---:|---:|']
for v in size['padding_summary']:lines.append(f"| {v['M']} | {v['issued_warp_HMMA']['mean']:.0f} | {v['useful_equivalent_warp_HMMA']['mean']:.0f} | {cell(v['useful_fraction'])} |")
c=size['ceilings'];lines+=['',
 f"The nominal padding-service scenario (wasted HMMA×4096FLOP/989TFLOP/s,all wall credit) gives six-load R **{c['padding_nominal_service_M8_32_R']:.5f}**. "
 'That is neither measured exposed time nor a removable hardware floor. A correct narrow-token method needs '
 'another algorithm and its decode/epilogue cost; PAD_USEFUL itself skips required K work and is never eligible.', '',
 f"Prior matched ACT−LOOP reference gives R **{c['activation_reference_M8_32_R']:.5f}** including operand sinks; "
 f"epilogue reference R **{c['epilogue_reference_M8_32_R']:.5f}**. The still-unsplit MMA+activation ceiling "
 f"R **{c['register_lifetime_unallocated_upper_R']:.5f}** bounds register/scheduling work only loosely. "
 'These overlap and cannot be added. The new contrasts must rank the largest buildable term before selecting a production build.', '',
       'Occupancy-only upside is **R1.00000**: BASE126regs,L2 127,L3 128 all retain two CTAs. '
       'The shared footprint also prevents a third CTA even if registers drop. Accumulator recurrence is '
       'separated with matching128-register controls; fragment-cache scheduling is tested independently. '
       'Old MATH drops BASE126→47 registers as well as changing operands/copies,so max(MATH,STREAM) '
       'is not a proved production critical-path bound. This work ranks compute/operand mechanisms first.', '',
       '## Exact resource budgets','', '| Arm | M1 regs | M8/16/32 regs | M32 shared B | CTAs/SM |', '|---|---:|---:|---:|---:|']
for arm in sorted(counts['qualified']):
 a=next(v for v in counts['rows'] if v['M']==1 and v['arm']==arm);b=next(v for v in counts['rows'] if v['M']==32 and v['arm']==arm)
 lines.append(f"| {arm} | {a['registers']} | {b['registers']} | {b['shared']} | 2 |")
lines+=['', 'All56 exact sm90a resource rows have zero stack/spills;all HMMA-bearing paths retain32static sites, '
        'L1 retains none. Real full Bf storage is65536B. At M32,two raw slots plus that bank require150672B '
        'and lose two-CTA residency. One raw slot plus the bank fits113808B,but must pay repeated shared reads, '
        'publication and the old one-slot penalty. It is not admitted before the FRAG_JIT bound is known.', '',
        '## Build decision and production graph','',
        'LAZY is a built and native-correct leaf experiment,not a production winner. The largest new oracle '
        'term and a production build cannot be judged before the H200 receipt. No new P4 nominee is exported. '
        'The next build must include real timed packing/producer work and be tested on whole production graphs '
        'with all12 scored shapes;per-call gains are not used as a transferred score.', '',
        f"Pinned production base S21 expanded **{prod['expanded_sha256']}**, shipment **{prod['shipped_sha256']}**. "
        'Use fresh disjoint seeds,all-round symmetric cohorts and promotion.evaluate on '
        '`T=46*t_sliding+16*t_full`;only held-out six-load R lower bound>1 and TASK establish promotion. '
        'S21 itself and its scheduled scored sender are untouched. Existing83 P3a nominations remain unchanged.', '',
        '## Screens, send and recovery','',
        f"Exact payload **{payload['sha256']}**, native local packet **{local['wire_bytes']}B** within11700B declaration. "
        'All448 native control TASK checks pass;all1920 raw round times,96NVML snapshots,60runtime resource '
        'rows and CTA/SM work/finish records survive. Local timing is sm120 correctness/transport evidence only. '
        f"LAZY separately passes **{leaf['checks']}** native gates acrossM1/8/9/16/24/32,real/all/empty routes,local_start0/96/224, "
        'and three changed inputs under each reused captured graph. Static/import/native-codegen/lab-only transport checks pass.', '',
        f"Actual waiting sender **{q['sender_pid']}**, launcher {q.get('launcher_pid')}, wrapper **{q['wrapper_sha256']}**, "
        f"waiting record `{q.get('waiting')}`. `--lane P3a --decision` and check quota are pinned. "
        'All hosts had zero quota;the queued sender waits on the real reset. No H200 timing or CI is reported yet.', '',
        'One earlier sender stopped at AST admission because the compressed outer payload lacked a literal '
        'lab definition;no judge job/quota. The corrected wrapper is screened. A local startup error from '
        'a script shadowing stdlib queue was fixed by renaming it send_batch.py;both exact native screens reran. '
        'No failed correctness control is classified as a loss.', '',
        'After assignment,`python3 tools/h200_lab.py read <job>` waits for the receipt. Then '
        '`python3 .../operand/read.py runs/.../<receipt>.json` audits immutable source/packets/resources/controls, '
        'and `python3 .../operand/report.py` fills the pending table. `sizing.py` reproduces the pre-receipt models. '
        'Find this job by wrapper e1710fb3 or the operand decision in runs/ledger.jsonl / runs/pending.','']
(D/'REPORT.md').write_text('\n'.join(lines))
hand=dict(state=q['state'],job=result['job'] if result else None,sender_pid=q['sender_pid'],wrapper_sha256=q['wrapper_sha256'],payload_sha256=payload['sha256'],rows=[],oracle_outputs_ignored=True,production_build_measured=False,production_plan=prod,report=str(D.relative_to(D.parents[5])/'REPORT.md'),previous_point_nominees_unchanged=83)
(D/'P4_HANDOFF.json').write_text(json.dumps(hand,indent=2)+'\n')
print('OPERAND_REPORT',hand['state'])
