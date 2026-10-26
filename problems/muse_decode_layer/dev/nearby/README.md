# S15 nearby-shape audit

Candidate: `cand_r3_best7_s15.py`, SHA256 `602548834f7481747701469b2a85154dde214bea28076c10fefce2c46b5240cb`.
S15 is unchanged. This audit owns the P4 nearby-shape criterion; no promotion or kernel fix is made here.

The final review rule is explicit in `upstream/llms.txt:112` and this problem's `AGENTS.md:34`.
`TASK.md` fixes hidden8192, Q16/KV2 heads of64, experts32×4096 and q_len1/8. Nearby request counts,
contexts, page sizes, ragged lengths and the position within the 2048-token window are valid audit axes.
It does not define a quantitative “much slower” cutoff. Report the measured ratios without inventing one.

## Branch inventory

“Retained” below means the same optimized route remains selected; it is a dispatch finding, not an H200
speed claim. “Generic” means the specialized route is lost; its actual penalty awaits the queued lab.
The exhaustive Python table/branch/assertion index is [branch_inventory.json](branch_inventory.json).
The line numbers below refer to the frozen S15.

| Branch / location | Off-shape behavior | Minimal action if H200 finds a cliff |
|---|---|---|
| `kernel_fn` / `BEST7_SETTINGS` (106,5321): twelve `(B,L,window)` keys, plus `other` | Different B selects `other`: best6 glue, auto attention, auto MoE, auto projection. Context/page/spread changes retain the selected row. No scored-context assertion. | Choose kernels by supported geometry/cost, rather than twelve exact B/L rows. |
| `GLUE_CL_ROWS` (123), `_best7_boundary` / `_best7_route` (7026,7043) | Eight small-token scored rows retain cluster8 norm/router at every context. Other B loses this path; uses tuned per-T geometry if T exists, otherwise row-CTA defaults `(8,1)` / `(4,1)`. T>128 late-weight-load glue only applies when a plan supplies geometry. | Extend safe cluster geometry or late-load dispatch by T range after measuring. |
| `SMALLM_FLAGS={1:95,8:69,16:69,32:69}` (7408); explicit `nt1_lop3` rows | Only explicit best7 rows use this dictionary. Nearby T does **not** index a missing key: auto MoE reaches V9. Even verify2/T16 and decode8/T8 lose the explicit small-M path. | Generalize flag choice and prepare replay-safe scratch for the supported T range; do not simply add an unsafe lookup default. |
| `hh_m128`, `mixed_parallel_m256` (6455,7055) | Fixed best7 rows select these at T128/256. Other B/T uses `moe6_plan` at128/256 if available, otherwise V9. No fixed-M expert is accidentally called at T192/264. | Measured range dispatch and corresponding prepared geometry; preserve routing-dependent work. |
| V9 / `experts_v9` (1585,1611) | Generic integer M, exact dimension assertions only. NT choice defaults1 outside the prepared32/128/256 rows; dynamically reduces stages2→1 if routing scratch needs it. Explicit ValueError only beyond device shared-memory capacity. Tested through T264. | If extending much further, move routing metadata out of shared memory; nearby T264 needs no change. |
| N128 register-fold full verify / `_VQ8_ROWS` (7733) | Only B16/32,L8,full select it; arbitrary context/page/scale uses the same kernel and dynamic page/seq addressing. Verify2/24 goes to FA3 on Hopper, Triton locally. **Likely performance discontinuity**, correctness independent. | Admit a broader B range after native correctness/timing; honor persistent split/partial capacity. |
| `_CLUSTER_ROWS={(1,1,True)}` last-CTA sliding (7937) | Decode1 sliding retains last-CTA on short/long context and page64. `cs_plan` includes page count,size,window and clamps splits to actual span; no 80K assumption. B2+ takes the other attention path. | Extend one-launch form by supported groups if the nearby B cost warrants it. |
| `_PP_ROWS={(1,1,False)}` full decode1 (7837) | Retained for other full contexts/page sizes; B2+ loses the direct-partials kernel. | Broaden only after occupancy/split measurements. |
| `attention_plan` / `_attn_port_variant` (5043,7006) | Portfolio tunes B1/16/32,L1/8,sliding/full. Unknown key defaults `('layer',None)`: q1 portable CUDA split kernel; q8 Hopper FA3 (local normal Triton). Supported L1/8 and arbitrary B/context/page. | Use dynamic attn_main/q8 native plans when they beat FA3; do not infer from local timing. |
| Attention split planners `_splits`, `_attn_main_plan`, `cs_plan` (2293,5058,7908) | Occupancy and page/span-derived, clamped split counts; work remains whole context/window. Keys contain dimensions/metadata, no activation values. | No correctness change indicated. |
| Gate split `GATE_SPLIT_ROWS` (374,409) | q1 only B1 both layers,B16 sliding,B32 full. Off-row computes fused qkv+gate; same arithmetic, less overlap. QKV3 projection is only requested when gate is computed separately. | Generalize by overlap benefit, preserve separate-gate execution. |
| NF selected roles (8270,8276,8282) | qkv3 T1/32; qkv_gate T32; O T1/8/16/32; unpooler T16/32. Any B/L producing the same T and role retains the port. Other T returns to original projection. `nf_project`'s native missing-plan RuntimeError is guarded by selection; no unsupported-T call. NF_FORCE is false. | Generalize a kernel family's supported M or admit additional measured shapes; don't route to an absent plan. |
| NF layout/template assertions (7991–8251) | N/K, coding metadata, contiguous BF16, launch-family invariants; model dimensions are fixed by TASK. Wide M128/256 NF kernels exist but **are not selected in S15**. | No nearby-shape failure from unselected templates; preserve exact codes. |
| Coded dense `CD_ROWS` (2972,3062) | T1 qkv_gate/pool_router only; other T uses original BF16 layout. qkv_gate bypassed by qkv3 where split is on; pool_router coded path remains. | Broaden only when decode cost remains below bytes saved. |
| Balanced `GB_ROWS`, `_GEMV_CHOICES`, pool T32 special (320,333,5104) | Balanced qkv/pool T1/8/16/32; full unpooler16/32. PoolT32 uses dedicated bal2. Unknown T<=64 runs generic split-K Triton using next-power-of-two token tile; >64 uses cuBLAS. T33/63/64/65 exercises this boundary. | Broader geometry dispatch by token tile and utilization, measured at edge sizes. |
| `PROJ_PINS`, `_MEASURED_LT`, `VDSK_SHORTLIST`, `_TUNE_TOKENS` (3149,5562,33,3108) | Only T1/8/16/32/128/256 tuned. Unknown T uses `custom`; >64 becomes cuBLAS. Same T at different B/L retains prepared descriptors; no descriptor lookup at absent M. | Expand supported plan generation by shape ranges; retain original BF16/FP32 boundaries. |
| T<=8 unpooler slice (5315) | The **call's total T=B×L** chooses1024 output rows; T9+ computes8192. Correct at decode2/7/8/9 and verify1/2. No internal chunking extends shortcut. | None: task-mandated discontinuity; baseline has it too. |
| `BEST7_DENSE_CHAIN`, prefetch, optional experts | Dense chain false in every row; no prefetch recipes selected. Optional grouped expert adapters apply only at prepared128/256. Dormant families do not reject nearby calls. | No promotion of dormant paths in this audit. |
| Prepare/assertions and model invariants |35 Python assertions: weight packing/roundtrip, fixed model dimensions, supported internal ABI, vq8 admission, template rewrites and generated geometry. No assertion that data must be one of the twelve scored shapes. | If a new geometry is admitted, change its guard/prepare together; don't weaken admission gates. |

## Case set and local verification

[cases.py](cases.py) defines65 cases:12 scored controls and53 nearby cases, seeds419731/427650.
Both sliding/full at decode2/8/24/48 and verify2/24; short-context token edges decode7/9/15/17/31/33/63/64/65
and verify15/17/31/33; contexts65536 at scored batch counts; sliding positions63/64/65 and
2040/2047/2048/2049; page64 at decode1 and native verify16/32; scaled FP8; mixed/ragged lengths;
and context96000 at decode48/verify24. Model dimensions and q_len1/8 are unchanged.

`screen.py` uses tools/local_eval in the judge image on GPU0, with the real TASK `collect` and
`check_implementation` (including near-tie routing, per-row tolerances, cache writes, FP32/FP8-derived
bounds). It sets the busiest expert range by the reference's exact criterion. Cache setup's scratch
chunk is8192 rows, retaining all context keys and distributions. Each seed is checked eagerly and
after two graph replays; all13 official TASK cases run too. No local timings are used to rank paths.

Normal mode uses S15's portable dispatch. Forced mode uses `dev/verify_best.py` only for Hopper
FA3/attention launches, and `dev/s15/local_standins.py` for N128/last-CTA launches. It retains S15's
selection rows and its real Python ABIs. The N128 stand-in is frozen S14 N64 arithmetic; last-CTA
uses direct-partial arithmetic plus the real pointer-copy/counter-reset launch. NF's original local
cluster arithmetic stand-ins remain explicit. These **do not prove native Hopper execution**.
Every S15 literal CUDA unit is independently compiled for sm90a by local_eval.

Normal and forced screens both pass all143 cases (13 official +65×2 seeds), each eagerly and
after captured replay:286 case/seed records and572 TASK correctness checks total. Each mode also
passes25 static checks (23 literal CUDA compilation units plus undefined-name/call-signature
checks). Evidence: [normal.json](normal.json), [forced.json](forced.json) and the matching
`*_local_eval.json` files. **No S15 local correctness failures were found; no candidate fix is indicated.**

The test driver initially had a duplicate `page` keyword in its own case builder, before any audit
fixture executed. The builder was corrected and both complete screens rerun. This was not an S15
failure; no candidate code or tolerance was modified.

## S15n: supported range dispatch

Candidate [cand_r3_best7_s15n.py](../../cand_r3_best7_s15n.py), SHA256
`ebceda91ed055adc0ece9161a350934dc92b93cd78277bc6b25e54ce66d890a6`.
[build_s15n.py](build_s15n.py) applies asserted edits to frozen S15;87 changed lines
(68 additions/19 removals; exact counts in `s15n.patch`). All36 source literals,
including every CUDA and attributed vendor body, are byte-identical to S15. No second
runtime implementation or fail-open alternative is appended. Four short helpers select
valid ranges; the existing kernel families execute the work.

| Lookup replaced | Supported range / dimensional guard |
|---|---|
| Best7 B/L/window row | Nearest B1/16/32 at unchanged q_len/window; no `other` route for nearby requests. |
| Small-M flags | M1 uses95 (SLOT requires exactly1); M2–64 uses69; counters remain prepared separately per flags word. |
| Expert family | T<=64 small-M;65–128 heavy/light N_full family;T>128 mixed N_full family. Both large families use the existing dynamic-M `nf_plan`, actual routing, scratch sizes and occupancy. Tested through T264. |
| Glue geometry | Nearest total-T anchor1/8/16/32/128/256, preserving the special verify1/T8 cluster geometry; larger tokens use existing row/late-load geometry. |
| Last-CTA sliding, full decode partials | Nearest B1/16/32. B1–8 uses last-CTA sliding or full direct-partials. Remaining decode groups use attn_main. Planners derive actual splits/pages/span. |
| N128 verify full | Every B>1,L8,full uses nearest16/32 setting; B1 retains its scored FA3 path. Split/partial sizes are computed from actual B. |
| Gate split | Nearest B1/16/32 per layer; q_len1 only, original gate kernel receives actual T and resets its counters. |
| Coded dense | Nearest token anchor; t1 coding extends to T2–4, then nearest original specialization. Exact BF16 codes and decode are unchanged. |
| Balanced GEMV and bal2 | Nearest seeded M for the same N/K, with actual-T masks/next-power-of-two tile. Pool anchor32 chooses bal2; dense GEMV's existing64-token resource boundary remains. |
| NF roles | Nearest anchor **among valid templates**: qkv3/d9<=8, qkv32<=32, O small<=8/16, unpool small<=16. O32 and unpool32 WGMMA mask actual M and support larger rows. Unselected anchors preserve the original S15 role choice. No unsupported plan is called. |
| Projection pins / Lt / VDSK | Nearest prepared choice per role. Lt descriptors bind M: choose a prepared ceiling, pad rows inside the timed call and trim output; above the largest plan, row chunks use the same full role. VDSK uses actual M and masks tails. No descriptor is reused with the wrong M. |
| T<=8 unpooler | Unchanged, always determined by the **whole call's T**. Lt row chunks above T8 continue using the full unpooler; they never use the shortcut. |

Normal and forced S15n screens each pass143 fixtures (13 official +65×2 seeds),
eagerly and after graph replay:572 TASK checks total. All25 static checks pass in
each mode. Evidence: `s15n_normal.json`, `s15n_forced.json` and their local_eval reports.
No S15n correctness failures were found. Hopper-specific device execution remains
unverified locally; the explicit launch stand-ins above are retained.

Scored dispatch proof: [prove_dispatch.py](prove_dispatch.py) traces each of the12
scored shapes under the same prepared state. Entire selected plan, gate split,
attention family/setting, MoE family/flags, projection family/algorithm and unpooler
width match S15. It also checks every prepared projection point against five possible
tuner choices, so the equality does not assume one local Lt outcome. The machine-readable
[scored_dispatch.json](scored_dispatch.json) records12/12 matches and candidate identities.

## Reset lab

[build.py](build.py) and [lab.py.inc](lab.py.inc) now generate a three-way payload:
**S15 / S15n / exact SGLang baseline**, all65 shapes,2 fresh seeds,8 cold alternating
CUDA-graph replays per side,1GiB read flush and measured empty-graph subtraction.
S15n is restored by a hash-checked source diff and uses S15's exact prepared weights,
Lt choices, lossless codes and counters; identical Triton functions and compiled modules
are shared. This avoids a second tuner and isolates dispatch. Both kernels still derive
all outputs from each live input. The public input generator and baseline are inlined;
no reference imports are shipped. Schema/finite checks are diagnostics; the independent
TASK gates above establish local correctness. Native H200 performance remains pending.

Both exact wrapper screens pass65 shapes×2 seeds×8 replays,13 official host tests and
2 wrapper static checks per mode. Local payload elapsed80.05s normal /60.60s forced;
these timings do not rank H200 performance.

Reproduce: run `build_s15n.py`, then `screen.py --candidate problems/muse_decode_layer/cand_r3_best7_s15n.py`
and the same with `--forced` in `CUDA_VISIBLE_DEVICES=0 tools/img python ...`.
Run `build.py`, `h200_lab.py build ...` and `screen_lab.py` / `--forced` for the exact lab.
After its native receipt lands, `read.py <receipt.json>` prints paired S15n/S15 and
SGLang-relative times. P4's performance criterion remains open until that receipt.

## Current reset identity

The unsent original lacked a top-level AST `kernel_fn`. P4 repaired only the entry; payload math, seeds and gates are identical. The original was retired under the dispatch lock after checking PID, command and source identities. `reset_superseded.json` preserves the evidence.

Repaired wrapper e5a7d14f restored as sender **587925**, priority4.9, same reset gate and full prepare budget; accepted as **a793c99eb9d447df**. h200_lab automatically absorbed89627 into an unsent combined wrapper; only that new owned sender579268 was retired. `reset_restored_queue.json` records the restored exact wrapper.

`speedup_report.py <receipt> --out reset_result.json` reports every load’s nearby/scored SGLang-relative speedup; groups use the nearest scored B with ties toward smaller B, retaining all53 nearby cases. The native TASK follow-up is ready in `native_validation/`: guarded S15n9cc90d2e vs S15,67shapes,268checks per mode. Send if the timing/control result supports retaining a range path; no promotion before native TASK validation. The B<=64 guard fixes NSMAX capacity beyond the65 timed cases. `native_validation/s15n_guarded.patch` applies onto exact S15 60254883.

## Native timing receipt and decision

Job **a793c99eb9d447df**, complete65shapes,2seeds×8coldreplays, SGLang available, no runtime/schema/finite failures. This payload has no remote TASK oracle; native correctness is a separate gate.

| Load | S15 scored speedup | S15 nearby speedup | nearby/scored | S15n scored speedup | S15n nearby speedup | nearby/scored |
|---|---:|---:|---:|---:|---:|---:|
| decode 1 | 2.2148 | 2.1146 | 0.9547 | 2.1991 | 2.2633 | 1.0292 |
| decode 16 | 1.8422 | 1.5937 | 0.8651 | 1.8393 | 1.8654 | 1.0142 |
| decode 32 | 1.6192 | 1.3252 | 0.8184 | 1.6233 | 1.4421 | 0.8884 |
| verify 1 | 1.9892 | 1.6150 | 0.8119 | 1.9954 | 1.7324 | 0.8682 |
| verify 16 | 1.8443 | 1.3513 | 0.7327 | 1.8442 | 1.8534 | 1.0050 |
| verify 32 | 1.9107 | 1.3877 | 0.7263 | 1.9081 | 1.7860 | 0.9360 |

All53 nearby cases are included in the diagnostic aggregation, grouped by nearest scored batch at the same q_len;46/16 layer weighting is retained. This is not the official score. Identical-dispatch scored S15n/S15 seed ranges by load are .97794–1.03416/.99554–1.00231/.99239–.99919/.99428–1.00065/.99491–1.00196/1.00054–1.00283. There is no independently captured A:A arm; d32 and v32 controls have small directional drift. Large nearby gains exceed that drift; scored dispatch is identical, so no scored speed claim is made.

Decision: range dispatch repairs substantial nearby cliffs, especially verify24/31/33 and decode2/24/48. Guarded S15n9cc90d2e keeps every timed case identical to the measured ebceda91, with the B<=64 native N128 capacity guard. Send native TASK validation now, then nominate the exact guarded patch to P4 only on native TASK/cache/replay pass. The timing recipe is measured; correctness remains unproven until that receipt. No arbitrary quantitative reviewer cutoff is assumed.

Native validation queued on attention as **sender1096763**, wrapper2ba5fcff, priority4, not-before1792972800; `native_validation/queue.json`. Exact patch and pending gate are indexed in `P4_HANDOFF.json`.
