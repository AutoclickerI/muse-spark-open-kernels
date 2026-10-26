# S19 operand/issue oracle batch

**P3a active.** H200 measurements pending: sender **3637718**, wrapper **e1710fb3ff91fcb6**, MoE check at **2026-10-27 09:00 KST**. All five hosts had zero remaining scored/check quota; no judge job is assigned yet.

Read first: moe_public_ports/P4_HANDOFF.json and MECHANISMS.md, including matched-ABI port losses; FINDINGS mixed_wgmma DIRECT/TRANS/WS/CTA2/CHUNK history. No donor code or decoded-weight/WGMMA producer-consumer design is copied. The operator register-only raw loads/bit-permuted decode are source mechanisms, not demonstrated small-M gains. This batch retains exact S19 math/copy/route/readiness code.

## Arms and interpretation

| Arm | One change / comparison |
|---|---|
| BASE / AA | Exact S19 62ea5bf4, independent workspaces |
| BASE_PAD | Exact S19 body with +8320B dynamic reservation; controls shared-bank reservation |
| FRAG_SMEM | Synthetic BF16 fragments prelaid in two vector planes in an8KiB shared bank;keeps segment-cached Bf;compare BASE_PAD |
| FRAG_REG | Pack-free register constants;removes activation gathers and their addressing/lifetime;loose oracle,not an exact load-latency split |
| FRAG_JIT | Same synthetic shared bank,load only one K-group at a time;shorter Bf lifetime,more shared reads per tile;compare FRAG_SMEM |
| ACC_CTRL / ACC_FREE | Same live accumulator-result sinks after every HMMA,all32 sites retained;FREE zeroes C each instruction,removing recurrence;wrong math |
| PAD_CTRL / PAD_USEFUL | Same decode sinks retain all weight/decode work;USEFUL issues HMMA on only r of8 K-warps to realize the useful-N instruction budget;wrong math,not a legal narrow-N implementation |
| L1 / L2 / L3 | Same prior sink-controlled decode / MMA+activation / epilogue-NOWAIT cuts,compiled in the same TU/cohort |
| HIST | Minimal exact-BASE observer;per-CTA work,SM and finish records,observer tax timed |
| LAZY | Correct original activation loads moved from segment entry to each K-group/tile;shorter Bf lifetime,more global reads;native leaf prototype only |

FRAG_SMEM/JIT use a repeated synthetic pattern,not a64KiB bank of the real activations. Their timed8KiB initialization and vector LDS remain. This is a pack-free/on-chip bound; real BF16 packing/publication and its shared capacity must be paid by a correct build. Oracle results are ignored. BASE/AA/BASE_PAD/ACC_CTRL/PAD_CTRL/HIST/LAZY are independently TASK-checked before/after every case. S19 loads BF16 bits directly into uint4/Bf; no activation conversion exists to remove. ACC_FREE originally compiled to one HMMA; paired live-result sinks fixed that before sending. Decode sinks use the current ping-pong reduction buffer, preserving its retirement boundary.

## H200 us and confidence intervals

Eight matched seeds ×four symmetric all-round arithmetic means; Student-t95% across seed means. Same route/weight seeds as the preceding cohort,all264 CTAs,256threads,depth2 and tile order. Extra shared reservation is an explicit paired control. These are selection data,not held-out G0.

| Arm | M1 | M8 | M16 | M32 |
|---|---:|---:|---:|---:|
| BASE | pending | pending | pending | pending |
| AA | pending | pending | pending | pending |
| BASE_PAD | pending | pending | pending | pending |
| FRAG_SMEM | pending | pending | pending | pending |
| FRAG_REG | pending | pending | pending | pending |
| FRAG_JIT | pending | pending | pending | pending |
| ACC_CTRL | pending | pending | pending | pending |
| ACC_FREE | pending | pending | pending | pending |
| PAD_CTRL | pending | pending | pending | pending |
| PAD_USEFUL | pending | pending | pending | pending |
| L1 | pending | pending | pending | pending |
| L2 | pending | pending | pending | pending |
| L3 | pending | pending | pending | pending |
| HIST | pending | pending | pending | pending |
| LAZY | pending | pending | pending | pending |

## Work counts and pre-receipt sizing

These counts reconstruct executed warp-HMMA from the retained per-CTA tile histogram ×8warps ×32 SASS sites,checked against768tiles per expert/token block. They are not NVPerf hardware counters. Useful equivalents scale by actual per-expert tokens;N<8 is not supported by this BF16 HMMA form.

| M | Issued warp-HMMA mean | Useful equivalent mean | Useful fraction [95%CI] |
|---|---:|---:|---:|
| 1 | 516096 | 64512 | 0.125 [0.125, 0.125] |
| 8 | 1400832 | 365568 | 0.271 [0.214, 0.329] |
| 16 | 1474560 | 703488 | 0.488 [0.397, 0.578] |
| 32 | 2187264 | 1508352 | 0.691 [0.629, 0.753] |

The nominal padding-service scenario (wasted HMMA×4096FLOP/989TFLOP/s,all wall credit) gives six-load R **1.01097**. That is neither measured exposed time nor a removable hardware floor. A correct narrow-token method needs another algorithm and its decode/epilogue cost; PAD_USEFUL itself skips required K work and is never eligible.

Prior matched ACT−LOOP reference gives R **1.02126** including operand sinks; epilogue reference R **1.01580**. The still-unsplit MMA+activation ceiling R **1.04014** bounds register/scheduling work only loosely. These overlap and cannot be added. The new contrasts must rank the largest buildable term before selecting a production build.

Occupancy-only upside is **R1.00000**: BASE126regs,L2 127,L3 128 all retain two CTAs. The shared footprint also prevents a third CTA even if registers drop. Accumulator recurrence is separated with matching128-register controls; fragment-cache scheduling is tested independently. Old MATH drops BASE126→47 registers as well as changing operands/copies,so max(MATH,STREAM) is not a proved production critical-path bound. This work ranks compute/operand mechanisms first.

## Exact resource budgets

| Arm | M1 regs | M8/16/32 regs | M32 shared B | CTAs/SM |
|---|---:|---:|---:|---:|
| ACC_CTRL | 127 | 128 | 85136 | 2 |
| ACC_FREE | 127 | 128 | 85136 | 2 |
| BASE | 117 | 126 | 85136 | 2 |
| BASE_PAD | 117 | 126 | 93456 | 2 |
| FRAG_JIT | 96 | 63 | 93456 | 2 |
| FRAG_REG | 96 | 64 | 85136 | 2 |
| FRAG_SMEM | 123 | 117 | 93456 | 2 |
| HIST | 119 | 125 | 85136 | 2 |
| L1 | 48 | 64 | 85136 | 2 |
| L2 | 119 | 127 | 85136 | 2 |
| L3 | 127 | 128 | 85136 | 2 |
| LAZY | 94 | 96 | 85136 | 2 |
| PAD_CTRL | 128 | 128 | 85136 | 2 |
| PAD_USEFUL | 124 | 127 | 85136 | 2 |

All56 exact sm90a resource rows have zero stack/spills;all HMMA-bearing paths retain32static sites, L1 retains none. Real full Bf storage is65536B. At M32,two raw slots plus that bank require150672B and lose two-CTA residency. One raw slot plus the bank fits113808B,but must pay repeated shared reads, publication and the old one-slot penalty. It is not admitted before the FRAG_JIT bound is known.

## Build decision and production graph

LAZY is a built and native-correct leaf experiment,not a production winner. The largest new oracle term and a production build cannot be judged before the H200 receipt. No new P4 nominee is exported. The next build must include real timed packing/producer work and be tested on whole production graphs with all12 scored shapes;per-call gains are not used as a transferred score.

Pinned production base S21 expanded **109c0135b9a445b88b2efd1ebb73f15002a332ce553c8e50afb5557d6816398d**, shipment **eacc087bd59b3cd58ee6b7690913868d86a8deb2ad1ff50040329ece62f7c8a4**. Use fresh disjoint seeds,all-round symmetric cohorts and promotion.evaluate on `T=46*t_sliding+16*t_full`;only held-out six-load R lower bound>1 and TASK establish promotion. S21 itself and its scheduled scored sender are untouched. Existing83 P3a nominations remain unchanged.

## Screens, send and recovery

Exact payload **f4a1f8835eed7835e9119b11c54bb40231f03b9c76ce8206ffda80d4ed583bb9**, native local packet **8985B** within11700B declaration. All448 native control TASK checks pass;all1920 raw round times,96NVML snapshots,60runtime resource rows and CTA/SM work/finish records survive. Local timing is sm120 correctness/transport evidence only. LAZY separately passes **162** native gates acrossM1/8/9/16/24/32,real/all/empty routes,local_start0/96/224, and three changed inputs under each reused captured graph. Static/import/native-codegen/lab-only transport checks pass.

Actual waiting sender **3637718**, launcher 3637457, wrapper **e1710fb3ff91fcb6a430acf5bede584ae4172ae08188823f94d673034f636301**, waiting record `runs/waiting/3637718.json`. `--lane P3a --decision` and check quota are pinned. All hosts had zero quota;the queued sender waits on the real reset. No H200 timing or CI is reported yet.

One earlier sender stopped at AST admission because the compressed outer payload lacked a literal lab definition;no judge job/quota. The corrected wrapper is screened. A local startup error from a script shadowing stdlib queue was fixed by renaming it send_batch.py;both exact native screens reran. No failed correctness control is classified as a loss.

After assignment,`python3 tools/h200_lab.py read <job>` waits for the receipt. Then `python3 .../operand/read.py runs/.../<receipt>.json` audits immutable source/packets/resources/controls, and `python3 .../operand/report.py` fills the pending table. `sizing.py` reproduces the pre-receipt models. Find this job by wrapper e1710fb3 or the operand decision in runs/ledger.jsonl / runs/pending.
