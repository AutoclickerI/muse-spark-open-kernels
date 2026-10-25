# Light/heavy expert branches (M=128, M=256)

Serves G1 through the decode layer's M=128/256 MoE. One kernel family (`gemm.cu.inc`, namespace `mwbr`):
u8-scale recoded WGMMA/mma with typed `__align__(1024)` smem, N8/N16/N32 tiles, two selection classes
(FM=1 expert token count, FM=2 rows in the expert's 32-row group), a one-CTA live route (all id loads in
flight, no fills; META adds the heavy-only T32 and 8<count<=16 T8 lists) and the distributed finalizer.
`sources.py` adds H_H's NT1 kernel as `pmz` (small W13 zeroes its SS rows; THR 16 or 8).

## Handoff (current)

**N16G4 handoff, H200 eea9d87b8d1640cf:** `patch_s12_light.py` **6aaf9a3d** applies onto exact
S12 final **240248b2**, producing `s12_light.py` **98ed03ef**. It includes the pinned PERMW/wide
changes and the measured N16G4 stream geometry. Requires sibling patch/build files; no project imports in
its output. `check_light_patch.py` / `light_patch_equiv.json`: **18/18 bit-equal to the measured arm**,
3 seeds x3 fixtures x2 M, all TASK gates pass. Normal `local_eval`: **13 tests +23 static PASS**
(`light_normal.json`); forced **13 tests +12 graph shapes +23 static PASS** (`light_forced.json`). Actual N16 math also has the direct equivalence/poison checks below. Whole-layer pairing is still required.


**PERMW for S12 final:** `patch_s12_permw.py` **a42a5832** applies onto exact **240248b2**,
`cand_r3_best7_s12_final.py`, producing **4a4ed79f**. Final dropped wide after whole-layer job 937801fe; this
patch restores wide as part of PERMW, using the pinned sibling `patch_s11_wide.py` **b2748e23**. It also accepts
the older a92b6619 wide cell. `check_permw.py` / `permw_equiv.json`: **18/18 bit-equal to final and 18/18 to the b88d04fe PERMW arm**,
3 seeds x3 fixtures x2 M; all TASK gates pass, maximum absolute difference zero.

Landed **b88d04fe86aa45a7** (36 fixtures/M): PERMW **0.9862 / 0.9787** vs LI4; LIW controls **0.9940 / 0.9894**.
The 1.38 / 2.13% includes the wide-layout portion; do not attribute all of it to scale/descriptor changes. PERM without
wide **0.9940 / 0.9904**. Adaptive AD64PW **1.0152 / 0.9762** has larger fixture spread (0.940-1.090 /
0.904-1.045), so PERMW was the component handoff at both M. AD128PW loses **1.1506 / 1.0995**. All TASK gates pass.
Receipt: `runs/muse_moe_w4a16/20261025T051654Z_check_b88d04fe86aa45a7.json`.
**Whole-layer promotion overrides these component timings:** integration's S13 `3dbcaccd` finds M128 PERMW
neutral; it currently retains PERMW only at M256. Do not broaden that selection from this lane's standalone wins.

New structural diagnostic: `roof_build.py` freezes the exact S12 nf CUDA. Four arms per heavy/NT1 branch and
W13/W2 phase: unchanged, trivial dequant with original copies/math, loads-only with original load schedule,
and a wide-MLP stream of the same input segments. Trivial/loads arms are diagnostic-only, deliberately wrong;
the host's returned computation stays unchanged. The table distinguishes unique bytes from scheduled rereads.

### Structural result, H200 3e8996f86bc74d45

`roof_payload.py` **9d8b264f**, based on frozen S12 final **240248b2** (nf CUDA **09dac24f**), MoE check.
12 seeds x3 fixtures/M, 288 phase cells; normal whole-MoE FP32 gates pass (max L2/row 0.004120/0.004709).
A:A signed/abs/max 0.000/0.277/4.016 us. Times are isolated phases, not critical-path increments or whole-layer wins.

| M | Branch/phase | Unchanged us | Trivial dequant us (paired ratio) | Loads-only us (ratio) | Max-MLP stream us (ratio) |
|---|---|---:|---:|---:|---:|
| 128 | NT1 W13 | 74.85 | 69.89 (0.9335) | 64.96 (0.8665) | 61.39 (0.8208) |
| 128 | NT1 W2 | 44.52 | 40.73 (0.9158) | 35.99 (0.8045) | 33.87 (0.7612) |
| 128 | heavy W13 | 56.43 | 50.62 (0.9036) | 38.13 (0.6794) | 29.18 (0.5200) |
| 128 | heavy W2 | 37.11 | 32.90 (0.8832) | 25.06 (0.6769) | 17.36 (0.4667) |
| 256 | NT1 W13 | 78.42 | 73.70 (0.9380) | 68.96 (0.8767) | 64.40 (0.8212) |
| 256 | NT1 W2 | 46.22 | 42.50 (0.9178) | 37.80 (0.8142) | 35.36 (0.7680) |
| 256 | heavy W13 | 109.47 | 94.43 (0.8628) | 70.17 (0.6428) | 50.32 (0.4648) |
| 256 | heavy W2 | 60.66 | 54.62 (0.9039) | 39.53 (0.6522) | 26.75 (0.4439) |

What this changes: **dequant alone is a 6-14% phase lever, not the explanation for a 2x gap**. It can still win,
but removing almost all of it did not halve either branch. NT1's stream reaches 4.36-4.63 logical TB/s; scheduled
weight reads are ~18-22% above unique weights. Heavy has larger staging/math and reread gaps: scheduled weight
bytes / unique weight bytes ~1.58 at M128, ~2.16 at M256. Heavy's activation-inclusive stream rates 8.6-13.6 TB/s
include L2 hits and repeated rows; **they are not HBM rates**. The timeline's 2.81-2.98 TB/s was also a logical
unique-byte proxy, not DRAM counters. The prior "issue-bound" claim was too strong.

Source/SASS parity: trivial retains all 63 heavy LDGSTS and 32 HGMMA instructions in the four-stage unroll;
all 128 HMUL2 disappear. NT1 retains its normal global LDG and 32 HMMA counts in trivial, with 128 HMUL2 removed.
Loads-only retains heavy copies; NT1's volatile activation loads keep addresses/cache policies but ptxas rolls
their loop, so static LDG counts differ. Resource reports and full counts: `roof_sass.json` / `roof_sass.py`.
The stream matches packed-weight multiplicity and cached-activation refresh reads, excluding small route,
gain/post-norm metadata and all output writes. Trivial/loads outputs are deliberately wrong and never returned
by the lab host. Local sm120: 24 normal reference gates, all 288 diagnostic cells complete, no cold-L2 timing claim.

Rebuild from this limit: `patch_s12_reuse_order.py` **b40a60bd** changes only task indexing to put N32 token
groups that share a weight tile adjacent across CTAs. It preserves the tile math, register/shared geometry,
padding and request count; intended gain is shorter L2 reuse distance. `order_build.py` compares exact S12,
PERMW, ORDER and PERMW_ORDER. Local shared-routing checks: **90/90 heavy ACT/Y/SS intermediate comparisons
bit-exact**, 3 seeds at M127/128/255/256/257. Full local screen: 48 fixtures/M/arm all TASK PASS (normal
L2/row <=0.004125/0.004947); one final BF16 replay difference in the combined arm is NT1 atomic drift.
Hopper K-loop instructions/KT128: S12 **207/208** W13/W2, ORDER **208/207**, PERMW and combined **188/188**;
8 HGMMA, 32 HMUL2 and 9 LDGSTS/stage in every arm. `order_sass.json` records the backward-loop counts.

MoE check **1bf2cc04c41a46af** landed, payload **b7a00873**; 16 seeds x3/M, all TASK PASS, A:A abs 0.483 us.
S12 196.44/270.64 us. PERMW **0.9895 / 0.9778**; ORDER **1.0010 / 1.0026**; combined **0.9927 / 0.9831**.
**ORDER loses; no integration.** Shortening the logical reuse distance did not establish an L2 gain.
Initial wrapper **6e8ddad5** was
rejected before a job id with "uses the network"; no network code exists. Only suspect token was the prose
"weight requests." (our local scanner catches requests.get/post but not this broader server match).
Changing that metadata to "weight reads." was accepted as wrapper **56149757**, with all four CUDA hashes
unchanged. The server scanner explanation is an inference from that one-token A/B.

Landed structural rebuild: `light_build.py` / `light_payload.py` **ce1ecf2e**, native CUDA **0a31e251**.
NT1 now owns only <=8-token experts; a third stream runs N16 WGMMA for 9-16-token experts, so each reads and
decodes its weights once instead of twice. Heavy N32 stays PERMW. No extra token padding. Persistent light
grids 2 and 4 CTAs/SM are separate arms, allowing NT1 residency beside the smaller ~36 KB N16 CTAs.
Local real-route screen: 48 fixtures/M/arm all TASK PASS. `light_check.py` / `light_validation.json`: **160**
captured class-transition checks, NaN ACT/Y/SS/OUT before every replay, M127/128/255/256/257 and counts
0/8/9/16/17/M; max L2/row **0.004222/0.004648**. Hopper compilation passes without spills.
MoE check **eea9d87b8d1640cf**, wrapper **29be9a1c**, landed: 16 seeds x3/M, all TASK PASS;
A:A signed/abs/max -0.060/0.438/1.584 us. All numbers below are paired component timings.

| M | S12 us | PERMW us (ratio) | N16G2 us (ratio) | N16G4 us (ratio) |
|---|---:|---:|---:|---:|
| 128 | 196.20 | 195.16 (0.9951) | 185.01 (0.9447) | **182.39 (0.9328)** |
| 256 | 270.57 | 265.78 (0.9825) | 257.63 (0.9526) | **256.43 (0.9485)** |

N16G4 wins both M. The fixture spreads are 0.799-1.010 / 0.912-1.008; full-MoE outputs often differ in
last bits because N16 WGMMA and NT1 split-K sum in different orders. Max FP32 L2/row 0.004134/0.004709,
well within TASK. This is not yet a whole-layer promotion and remains far above the unique-byte target.

### Whole-MoE structural schedule

`whole_model.py` / `whole_model.json` use the measured 3e8996f8 phase rates, with all branch service times
added before an explicit overlap assumption (never treating max(heavy,NT1) as a prediction). Unique-weight
floors at 4.58 TB/s are **102.0 /121.9 us**, so targets are **132.6 /158.5 us** for these fixtures.
Decode-once at unchanged activation traffic models **182.7 /235.0 us**; four-way activation multicast gives
an optimistic **155.8 /186.7 us** before cluster barriers, padding and occupancy costs. That is a bound using
the old address-mix rate, not a measured multicast throughput. Shared scheduling with light weights read
once models **178.6 /236.1 us**, range **143.1-214.2 /177.1-295.1** for full-to-zero residual overlap.
The planning points nearly tie; shared scheduling was selected for the stronger M128 point without assigning
an unmeasured multicast throughput. These models do not establish the <=1.3x outcome.

`shared_build.py` / `shared.cu.inc` construct one work list for adaptive N8/N16/N32 tiles across every expert.
The per-phase queue and single-launch DAG are paired arms. In the DAG, each completed W13 tile releases its
stores before incrementing the expert counter; the last tile publishes W2 readiness. Every CTA can then
consume that expert's W2 or take more W13 work. No branch-specific SM cap, software grid barrier or
residency-dependent progress assumption; no host input-value control flow. Route resets all counters on
every graph replay. The N8 copy guard avoids writing a 16-row activation tile into an 8-row allocation.

Final payload `shared_payload.py` **cd8c87fb**, CUDA **f7df4e22**. `shared_check.py`:
**450 captured checks**, invalid queue counters + NaN scratch, sizes127/128/255/256/257, empty, class-boundary,
hot and real skewed routes; PHASE and DAG **75/75 bit-equal**, three replays/case; max L2/row
0.004155/0.004782. SASS (`shared_sass.py/json`): W13/W2/DAG **125/134/139 regs**, no stack/spills,
N32 K-loop **189.5/189.25/189.25 instructions per KT128**, 8 HGMMA +32 HMUL2 +9 LDGSTS. Group wait1
is retained. Earlier out-of-line versions serialized every HGMMA despite no spills, and were discarded;
the final uses one shared W13/W2 loop body and inline dispatch, preserving async groups.
MoE check **ea794250c51744f7**, wrapper **0c8df035**, landed on16 seeds x3/M, all TASK PASS.
S12 **195.67 /269.27 us**; PERMW **193.56 /263.26** (0.9890/0.9777);
SHARED_PHASE **201.70 /279.17** (**1.0337 /1.0374**); SHARED_DAG **214.51 /297.85**
(**1.0999 /1.1069**). A:A abs/max0.439/2.192 us. H200 occupancy4/3/3 CTAs/SM for W13/W2/DAG.
**Both shared-list recipes lose; no integration.** Ready-expert scheduling adds another~13/19 us above
PHASE. The experiment does not isolate atomic scheduling from changed light math/activation traffic.
The optimistic planning estimate is rejected by this receipt; next build targets one read/decode plus
shared activation slices, retaining NT1 for the light experts.

The queue also replaces <=8-token NT1 with N8 WGMMA; this changes activation read traffic. `whole_bytes.py`
counts that explicitly on the local versions of the same fixture seeds (routing can differ on H200):
new light activation reads **90.9 /93.1 MB**, versus native phase refresh reads ~24 MB total. The planning
model does not price this cache-traffic increase or the new queue atomics/register occupancy separately;
its point estimate is optimistic. At the measured activation-inclusive stream rates, the extra bytes
alone price ~5-8 us if they fail to overlap. This uncertainty is a reason to measure the queue, not claim
a forecast or byte-floor result. No new branch recipe is promoted from this model.

**Why the old N64 form lost:** `n64_audit.py/json` extracts the exact 858ff2fa payload and recompiles its
lean N32/N64 kernels. S4 dynamic smem **52,223 ->84,991 B**, W13 static smem **4,096 ->8,192 B**,
W13 registers **91 ->116**, W2 **87 ->110**. The shared-memory limit changes **4 ->2 CTAs/SM**.
N64 doubles per-tile accumulators, can pad65-96 tokens to128 rather than96, keeps the channel-CTA activation
rereads, and adds separate heavy-class launches. Its body phases actually rise183.4->195.8 us atM128 and
266.6->280.1 atM256 (858ff2fa). These are measured costs plus source/resource evidence for an occupancy,
padding and pipeline mechanism; the old lab did not isolate a single cause. Repeating N64 without changing
those costs is closed. A multicast rebuild must share activation slices and demonstrate adequate residency.


The earlier FP16 magic proposal is audited separately, not pursued as the main lever: `magic_audit.py` /
`magic_audit.json` exhaust **65,024 FP4-pair x finite E4M3-code cases**. FP16 subnormal placement followed by
HMUL2 and an actual BF16 conversion is bit-exact in every case. Its raw FP16 bits mismatch BF16 in **63,456**
cases, so it cannot feed the current BF16 WGMMA directly. The standard affine INT4 magic recipe mismatches
E2M1 magnitudes in **96/256** pairs (2.5/3/3.5 instead of 3/4/6). This is not a demonstrated 64-instruction
BF16 decoder, and the H200 trivial-dequant bound above does not support prioritizing it over the load pipeline.

S11-final patches (2026-10-25, P3b takeover):

- `patch_s11_f4.py` **ef041fd4** applies onto `cand_r3_best7_s11_final.py` **acda6c62** (or a derived candidate
  retaining its nf_plan). Changes only nf_plan's finalizer argument 1 -> 4: the CUDA already contains finish<4>.
  Generated `s11_f4.py` **9696416e**. `check_f4.py` / `f4_bitexact.json`: bit-equal against split 1 on the SAME S11
  producer Y/SS (3 fixtures at each M), plus captured synthetic finalizers at both M and local_start=64.
  Local whole-layer normal: 13 PASS + 23 static; forced adapters: 13 PASS + 23 static (one decode-32 benchmark
  exhausted GPU0 memory while other lanes ran). That adapter screen cycles other expert engines; the actual
  nf portable math is verified by the separate cell comparisons above.
  H200 paired evidence: `5e197b5e` 0.9850 / 0.9885.
- `patch_s11_wide.py` **b2748e23** applies onto S11 final, before or after F4; changes only heavy packed-bank
  layout and its loads. Generated F4+wide **e9ce6986**. H200 `f976b09d3fc445eb` (collected): LIW vs LI4
  **0.9936 / 0.9900**, 12 seeds x3, all TASK PASS, bit 35/36 at each M (NT1 norm atomic replay difference).
  `check_wide.py` / `wide_equiv.json`: integrated F4+LIW **18/18 bit-equal** against S11 final's cell on identical
  inputs (3 seeds x3 fixtures x2 M), all TASK gates pass. Integration should pair it separately before promotion.

`f976b09d` co-residency cap LOST on H200: G1 **1.2351 / 1.4045**, G2 **1.0242 / 1.0586**, G3
**1.0285 / 1.0433**, G1p5 **1.1951 / 1.3436**, G2p5 **1.1115 / 1.0882** (M128/M256, vs LI4).
The local 0.957 / 0.944 did not transfer. W2 TMA LT2 **0.9993 / 1.0015** is neutral.
Controls 191.79 / 270.98 us; LIW 190.22 / 268.18 us. A:A abs 0.565 / 0.440 us.
Receipt: `runs/muse_moe_w4a16/20261025T044440Z_check_f976b09d3fc445eb.json`.

Landed **89608c595a9f4f6f** (`prof/lab/moe_p3b_stage.py`, send **ec6bc578**): both M, LI4 vs LIW, TYP/TYPW
(activation tile in static typed-smem; packed weights still dynamic), WP (W2 per-warp norm partials; finalizer sums
them in the old order), PAIR2/PAIR4 (one decode feeds two independent N32 accumulators; no N64 WGMMA), DESC (one
base descriptor per expert/channel tile). Local screens: M128 80/80 bit-equal on route-transition cases, M256
70/70 before the DESC-only edit; nearby sizes and local_start=64 pass. sm90a compile passes.

12 seeds x3, all TASK PASS (M128/M256 paired ratios vs LI4):

| Arm | M128 | M256 | Decision |
|---|---:|---:|---|
| LIW | 0.9924 | 0.9884 | Repeats f976b09d's win |
| TYP | 1.0012 | 0.9993 | Neutral |
| TYPW | 0.9918 | 0.9857 | Almost all gain is wide layout; static storage alone adds little |
| WP | 0.9990 | 1.0017 | Neutral; no integration |
| PAIR2 | 1.0225 | 0.9794 | M256 win but fixture spread 0.902-1.041; successor measured in b88d04fe |
| PAIR4 | 1.0130 | 1.0040 | Loses |
| DESC | 0.9978 | 0.9960 | Small repeated gain; combine in next batch |

Controls 191.00 / 270.99 us, A:A abs 0.503 / 0.465 us. PAIR2 194.93 / 264.69 us;
TYPW 189.53 / 266.82 us. Receipt: `runs/muse_moe_w4a16/20261025T050111Z_check_89608c595a9f4f6f.json`.

DESC's H200 SASS removes **31/32 ULOP3** and most descriptor UMOVs from the unrolled four-stage K loop:
W13 1449 -> 1401 static instructions, W2 1294 -> 1246 (not dynamic instruction counts). The 128-instruction dequant
core in that batch remains: 4 x (9 LOP3 + 7 SHF + 8 HMUL2 + 4 PRMT + 4 LEA). Static typed-smem by itself keeps
the descriptor ULOP3s. Proxy fences remain required after non-bulk cp.async ([NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/),
9.7.10.28.3.1 / 9.7.17.7.1); the old no-fence arm remains a diagnostic only.

Measured in b88d04fe: `prof/lab/moe_p3b_adaptive.py`: 64/128 token capacity, two pipeline slots, each decoded weight feeds
only the live independent N32 accumulators; dead token groups issue no WGMMA. Base / hoisted descriptor / wide
variants. M128 and M256 local 80/80 bit-equal plus nearby cases before the scale-permutation addition; five
paired local M256 seeds: AD64D / AD64W 0.8898 / 0.8840, AD128D / AD128W 1.0263 / 1.0201 (local rankings only).

Scale-core successor: rotate each packed 32-bit scale word left four bits in prepare (lossless same-sized u8
bank). Kernel forms even/odd BF16 scale pairs with 2 LOP3 + 1 SHF, then 4 PRMT duplicates. Its scale base is a
kernel argument so ptxas can use one register constant across all iterations (two immediate inputs otherwise
split each LOP3 into AND+OR). Dequant **128 -> 124 instructions per KT128**. SASS with hoisted descriptor:
W13 1401 -> 1385, W2 1246 -> 1238; no spills. New adaptive payload compares PERM/PERMW (N32 S4) and
AD64D/P/PW, AD128D/P/PW (live N32 groups, S2); P=rotated scale words, W=wide layout.
Final payload **3bf3b5f2**: M128 and M256 each **100/100 bit-equal** on poisoned route-transition checks, nearby
sizes and local_start=64 pass. Both capacity-64 and capacity-128 kernels compile for sm90a without spills.
MoE-host wrapper **779b8fe6**, landed job **b88d04fe86aa45a7**; results and current handoff at the top.
The 1.3x floor target remains unmet.

## Earlier integrated base (S7 -> S11)

`nf_cell.py` e86f6fb36a467233b2460c18689456d0108f37ef52a68eb7536de0a2da91ebdd: N_full at M=128 and M=256 (live route; heavy
N32 W13->W2 with the lean K loop, W13 kind 26 = gate/up rows interleaved by 8 (`nf_ilv` in prepare) with SwiGLU in
registers, W2 kind 21, launched first on the side stream || NT1 <=16 W13->W2; one join; own buffers; reads S7's H_H u8
base + NT1 banks). On S7R: `problems/muse_decode_layer/cand_r3_best7_s7n.py`
795497c8feabe97dd711857c1f3456efffaa76a3a6387b384b5aaf3fcff7b578 (`dev/s7n/`: 72/72 bit-equal to the timed arm LI_swap at
both M, normal 13 PASS + 21 static, forced 13 PASS). S11 applies `dev/s7n/nf_patch_e200b0ee.py` b22421b2 (cell block +
the nf_ilv prepare edit); the earlier lean-K patch `nf_patch.py` 686e5025 stays as S11 loaded it. The lab kinds 30-33
(wide banks, TMA-fed X) came after this cell.

## Landed: probes and finalizer split, H200 job 5e197b5e (send 453c58e2; read flush)

vs LI_swap (M=128 194.95, M=256 275.02 us), 12 seeds x3: F4 (finalizer over 4 blocks per token) 0.9850 / 0.9885, F8
0.9863 / 0.9932, F16 0.9945 / 1.0064 (bit-equal; isolated finish 12.8 -> 9.1 us at M=128). No per-stage proxy fence
(probe) 0.9935 / 0.9869. Probes (TASK FAIL by design): heavy alone 99.8 / 177.7 us, NT1 alone 127.6 / 133.7, no
finalizer 0.9627 / 0.9691. The branches barely overlap (195.0 vs 99.8 + 127.6): heavy, launched first, fills every SM's
shared memory (W13 ~57 KB x 3 per SM) and NT1's ~87 KB CTAs wait.

## Landed: gate/up interleave, H200 job 701d2867 (send 8007df6c; read flush)

vs L_swap (M=128 196.15, M=256 279.62 us), 12 seeds x3, all TASK PASS: LI_swap (W13 rows interleaved by 8 in prepare so
SwiGLU finishes in registers, kind 26) 0.9923 / 0.9838 (bit-equal; one M=256 fixture differs in the last bit in every arm
including the control: NT1 W2 norm partials are float atomics). Heavy grid /2, x2: 1.0043 / 1.0005, 1.0048 / 1.0024 (closed).

## Landed: lean K loop, H200 job 858ff2fa (send eeffa0cb; read flush)

vs N_swap (M=128 205.19, M=256 296.72 us), 12 seeds x3, all TASK PASS, bit-equal 36/36: L_swap (lean K loop) 197.16 /
279.38 (0.9613 / 0.9426). Decode once per 64-row expert tile loses: L64_swap 1.0233 / 0.9803, L64r_swap 1.0097 / 0.9983
(N64 kinds deleted). Phases (route/body/finish, us, isolated graphs): L_swap 6.5/183.4/12.9 at M=128, 6.6/266.6/13.6 at
M=256. SASS sm_90a per KT=128 stage: 254 -> 207 instructions; dequant core 128 (per 16 weights LOP3 9, SHF 7, HMUL2 8,
PRMT 4, LEA 4); the rest: fence.proxy.async (MEMBAR.ALL.CTA + FENCE.VIEW.ASYNC + 4 dummy LDS), BAR, 8 LDS, 9 LDGSTS,
descriptor UMOV/ULOP3 ~15, 64-bit pointer increments, IMAD.MOV operand copies.

## Landed: M=256, H200 job 110599ba6b6144d4 (send a45ff19d; zero_()-flushed, paired ratios only)

vs exact S6 mixed_parallel 347.13 us, 12 seeds x3 fixtures, all TASK PASS, all bit-equal to S6:
R_full 0.9227 (fork once: side = <=16-row groups N16 W13->W2, main = full groups N32 W13->W2), E_full 0.9275,
per-phase joins 0.955-0.960, one stream 1.016 (parallel branches are the whole gain), R8_full 0.9958,
live route 0.9883 (10.59 -> 6.62 us), typed-smem W2 0.9831 (sm_120 showed 22%; H200 0.6%).
Handoff: `problems/muse_decode_layer/cand_r3_best7_s7r.py` = S7 with M=256 on R_full (`dev/s7r/`): bit-equal to
S7's mixed_parallel on 72 fixtures (`dev/s7r/bitexact.json`), normal local_eval 13 PASS + 23 static.

## Landed: M=256, H200 job 14f942d57e764ebf (send 701024b9; judge-style read flush)

vs exact S6 mixed_parallel 342.80 us, 12 seeds x3 fixtures, all TASK PASS: N_full 298.68 (0.8701), B8_full 0.8768,
N_phase 0.9063, R_s6 0.9230, NR_full 0.9305, HH_u8 0.9424, B8R_full 0.9595, NRm_full 0.9725, HH 0.9941.
Handoff: `problems/muse_decode_layer/cand_r3_best7_s7n.py` = S7R with M=256 on N_full (`dev/s7n/`; mixed_parallel's
engine deleted). The M=128 half skipped on a 60 s pre-compile gate (now 25 s); resent as `49cd13f548734cd3`.

## Landed: M=128, H200 job c01b1916e2c74af8 (send 49cd13f5; judge-style read flush)

vs exact S7 H_H 234.78 us, 12 seeds x3 fixtures, all TASK PASS: N_full 207.40 (0.8842, bit-equal 36/36), B8_full 0.8812,
N_phase 0.9307, NR_full 0.9474, HH_u8 0.9702, B8R_full 0.9770, U16 0.9931 (S7's in-register word costs 0.7%), NRm 1.0025.
N_full is one cell for both M: `nf_cell.py` (heavy grid 1056 at M<=128, 2112 at M=256); `cand_r3_best7_s7n.py` routes
M=128 and M=256 through it.

## Landed: N_full tuning, H200 job 880c388117ed4e16 (send 2bb856ba)

vs N_full (M=128 207.60, M=256 299.49 us): heavy on the side stream (launched first) 0.9799 / 0.9882; priority adds
nothing; NT1 first at high priority 1.04; NT1 grid x0.75 / x0.5 1.02 / 1.07; heavy grid /2, x2 neutral. `nf_cell.py`
and `cand_r3_best7_s7n.py` now launch the heavy branch first.

## Built: `build.py` -> `prof/lab/moe_br_m{128,256}.py` (branch arms), `moe_nf_tune_m{128,256}.py` (BR_TUNE arms)

Controls: M=128 HH = exact S7 H_H (u8 `hh_cell.py`); M=256 S6 = exact mixed_parallel, plus R_s6 (= R_full).
U16 (M=128) = exact u16-word H_H: S7's in-register scale word cost. HH_u8 = H_H with this family's WGMMA.
Arms: N (NT1<=16 beside heavy N32), NR (heavy remainders on N16 beside NT1), NRm (remainders on the heavy
stream), B8 (W2 NT1<=8 + N16 for 8<count<=16), B8R; `_full` forks once, `_phase` per phase.
`check.py M` gates every arm (poisoned scratch, none/hot/remainder/single routes in one graph, nearby M,
local_start 64) then runs the lab: both pass locally (`check128.json`, `check256.json`).
MoE-host screen (`lab_screen.py`, 5 tests + 6 static) passes. Sent together: send `701024b91f79ac28` (MoE host).
