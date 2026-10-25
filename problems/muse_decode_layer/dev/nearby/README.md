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

## Reset lab

[build.py](build.py) and [lab.py.inc](lab.py.inc) produce one self-contained decode-host payload,
with the exact S15 host, a public-spec input generator and the exact SGLang baseline inlined.
No reference import or external repository path is shipped. Native H200 FA3 scheduler metadata
is constructed normally. Baseline and S15 use identical inputs and the same input-derived busiest
expert range; both overwrite their own new-cache slots. Two fresh seeds,8 alternating cold graph
replays per side,1GiB **read** flush, measured empty-graph subtraction.65 shapes, one payload.

Remote schema/finite checks are diagnostics; the FP32 TASK correctness claims come from local checks
and the host's official tests, not a comparison with a rounded baseline. The payload honors prepare's
remaining deadline and reports missing cases as incomplete. The exact shipped wrapper passes normal and forced local_eval:65 shapes×2 seeds×8
replays against SGLang,13 official host tests and2 wrapper static checks in each mode. Native FA3
is substituted only locally; schema/finite gates pass on all65 pairs. Local payload elapsed134.15s
normal (first baseline JIT),67.53s forced (warm cache); neither is an H200 timing.
[validation.json](validation.json) records the exact screened wrapper hash. No native numbers are claimed before
its receipt. Decode check quota (preserving scored runs for candidates). Priority5, not-before1792972800 (2026-10-26 00:00 UTC/09:00 KST); sender **PID 1113126**, wrapper `a357ab6f8bc5a74b`.
No judge job ID yet; the gate prevents an early send. Exact identity: [queue.json](queue.json).

Receipt analysis: compare SGLang/S15 for nearby vs their matching scored controls. A lower ratio flags
a lost speedup; report raw S15/SGLang times too, separating the task-mandated T8/9 unpooler jump.
A local pass is insufficient to close P4's “not much slower” criterion until this H200 result lands.

Reproduce: `CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_decode_layer/dev/nearby/screen.py`
and the same with `--forced`; `build.py` generates the lab, `screen_lab.py` / `--forced` screen
the exact host/payload. Run `read.py <receipt.json>` after `h200_lab.py read <job>` lands.
