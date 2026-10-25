# S15 producer-release audit and S15e

Base `cand_r3_best7_s15.py` SHA256 `60254883` is unchanged. Experimental
[cand_r3_best7_s15e.py](../../cand_r3_best7_s15e.py), SHA256 `084e004b`, keeps every
shape plan, arithmetic expression, weight layout, cache key and precision boundary.
This lane changes programmatic launch/release/wait plumbing only.

The proposed lever has a narrower opportunity here than in the donor: most owned
S15 kernels **already release at entry**. No owned scored kernel explicitly releases
after waiting for its last TMA. Delaying existing entry releases is not an earlier
release. S15e keeps those and closes the missing MoE/VDSK and consumer-enable edges.
No “34 us” benefit is assumed for Muse.

NVIDIA's [PDL guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html)
requires every producer CTA to trigger (or exit), a PDL-enabled consumer launch, and
a dependency wait before dependent reads. The wait observes completed/flushed producer
results; triggering does **not** make outputs ready. Consumer prologues may overlap
when they use only independent inputs. Events still record full completion.

## Current trigger table

Every family in S15's twelve scored shape paths is below, plus portable/off-shape
families exercised by the all-shape screen. “Implicit exit (external)” means S15
requests ordinary library completion; it is not a claim about undocumented internal
FA3/cuBLAS SASS. Exact source locations and caveats are in
[trigger_audit.json](trigger_audit.json).

| Kernel / producer family | S15 trigger | Successor waits before producer-output read? | Successor launch / barrier |
|---|---|---|---|
| NF qkv3/qkv_gate warp and cluster GEMVs | entry | yes | PDL |
| balanced _gb_kernel / generic _gemv_kernel / pool bal2 _gi_kernel | entry | yes | PDL |
| coded_gemv V3 pool_router | entry | yes | PDL |
| NF O / NF full-unpooler warp, OL, MW | entry | yes | PDL |
| VDSK _vdsk_gemm qkv M128 | implicit exit | yes | PDL |
| cuBLAS / cuBLASLt projection main and optional split-K reduce | implicit exit (external) | yes for owned consumers | PDL after library completion |
| _attn_sumcast_kernel | entry | NO wait in portfolio S15; ordinary launch serializes | ordinary |
| attn_main portfolio q1 | entry | yes | PDL |
| partial-port full decode1 | entry | yes (only last split CTA reads Y) | PDL |
| last-CTA sliding decode1 | entry | yes (only last split CTA reads Y) | PDL |
| gate split projection / gate epilogue | entry | yes | PDL |
| vq8_main verify16/32 full | entry | yes | PDL |
| vq8_combine | entry | library completes serially; owned variants wait | ordinary library or PDL native |
| _qk_prep_kernel / _best7_old_qk_prep_kernel | entry | external FA3 serialized; portable split waits | ordinary FA3 / PDL Triton |
| FA3 attention main and internal combine | implicit exit (external) | cast is ordinary; norm waits | ordinary cast / PDL norm |
| torch BF16 cast after FA3 | implicit exit (library pointwise) | yes | PDL |
| _norm_gate_kernel | entry | owned NF waits; libraries serialize | PDL native / ordinary library |
| portable attention CUDA attn_main | entry | yes | PDL |
| portable attention CUDA attn_combine | entry | yes for owned consumers | PDL |
| portable Triton _attn_split_kernel / _attn_combine_kernel | entry | yes | PDL |
| boundary_cl b1/b2 | entry | pool/router waits; return has graph completion | PDL or none |
| route_cl | entry | moe_var yes; route_live NO, ordinarily launched | PDL small-M / ordinary large-M |
| _boundary_fast / _boundary_late / _route_fast and best5 glue | entry | owned PDL consumers wait | PDL |
| moe_var flags95/69 | implicit exit | yes | PDL |
| V9 recoded/NT2 generic moe_fused | implicit exit | owned consumers wait; library serializes | PDL owned / ordinary library |
| N_full route_live | implicit exit | event completion | ordinary launches |
| N_full heavy and light W13 gemm | implicit exit | NO S15 explicit wait; ordinary serialization | ordinary |
| N_full heavy and light W2 gemm | implicit exit | event completion | ordinary + events |
| N_full small-expert phase W13 | implicit exit | NO S15 explicit wait; ordinary serialization | ordinary |
| N_full small-expert phase W2 | implicit exit | event completion | ordinary + events |
| N_full finish<4> | implicit exit | library serialization; no owned wait inside Lt | ordinary library |

## Call paths at the six loads

Expand the named families using the trigger table above. Each row below is one of
the12 timed shapes (46 sliding +16 full layers per load); common norm/router/experts
subkernels are retained rather than hidden inside one timing total. Native Lt
heuristic choices depend on prepare, so external algorithms are named by their
measured family rather than an invented fixed binary symbol.

| Load / layer | Ordered kernels / families |
|---|---|
| decode 1 / sliding | NF qkv3 → direct qkv partials / BF16 → last-CTA sliding → gate_split → NF O → cluster b1 → coded pool/router → cluster route → moe_var → generic slice GEMV → cluster b2 |
| decode 1 / full | NF qkv3 → direct qkv partials / BF16 → direct-partial full → gate_split → NF O → cluster b1 → coded pool/router → cluster route → moe_var → generic slice GEMV → cluster b2 |
| decode 16 / sliding | balanced qkv3 → qkv sumcast → attn_main → gate_split → NF O → cluster b1 → balanced pool/router → cluster route → moe_var → NF unpooler → cluster b2 |
| decode 16 / full | balanced qkv → qkv sumcast → attn_main → gate fused into attention → NF O → cluster b1 → balanced pool/router → cluster route → moe_var → NF unpooler → cluster b2 |
| decode 32 / sliding | NF qkv_gate → qkv sumcast → attn_main → gate fused into attention → NF O → cluster b1 → bal2 pool/router → cluster route → moe_var → NF unpooler → cluster b2 |
| decode 32 / full | NF qkv3 → qkv sumcast → attn_main → gate_split → NF O → cluster b1 → bal2 pool/router → cluster route → moe_var → NF unpooler → cluster b2 |
| verify 1 / sliding | balanced qkv → direct qkv partials / BF16 → qk prep+FA3+norm_gate → gate fused into attention → NF O → cluster b1 → balanced pool/router → cluster route → moe_var → generic slice GEMV → cluster b2 |
| verify 1 / full | balanced qkv → direct qkv partials / BF16 → qk prep+FA3+norm_gate → gate fused into attention → NF O → cluster b1 → balanced pool/router → cluster route → moe_var → generic slice GEMV → cluster b2 |
| verify 16 / sliding | VDSK qkv → direct qkv partials / BF16 → qk prep+FA3+norm_gate → gate fused into attention → Lt/cuBLAS O → Triton b1 → Lt/cuBLAS pool/router → Triton route → N_full route+3 branches+finish → Lt unpooler → Triton b2 |
| verify 16 / full | VDSK qkv → direct qkv partials / BF16 → vq8 main+combine → gate fused into attention → Lt/cuBLAS O → Triton b1 → Lt/cuBLAS pool/router → Triton route → N_full route+3 branches+finish → Lt unpooler → Triton b2 |
| verify 32 / sliding | cuBLASLt qkv → direct qkv partials / BF16 → qk prep+FA3+norm_gate → gate fused into attention → Lt/cuBLAS O → Triton b1 → Lt/cuBLAS pool/router → Triton route → N_full route+3 branches+finish → Lt unpooler → Triton b2 |
| verify 32 / full | cuBLASLt qkv → direct qkv partials / BF16 → vq8 main+combine → gate fused into attention → Lt/cuBLAS O → Triton b1 → Lt/cuBLAS pool/router → Triton route → N_full route+3 branches+finish → Lt unpooler → Triton b2 |

## Changes and independent arms

[build.py](build.py) composes S15e and each grouped leave-one-out arm from the exact
frozen S15 source. The full candidate has no runtime fallback or duplicated old
implementation. Group omission is a build-time experiment.

| Producer / edge | S15e change | Consumer safety |
|---|---|---|
| Small-M `moe_var` flags95/69 | Entry release, guarded by existing PDL flag | Existing router wait retained; unpooler preloads weights independently and waits before X |
| Generic recoded V9 `moe_fused` | Entry release | Input launch stays ordinary; successor owned unpooler waits |
| Large-M `route_live` | Entry release and PDL launch | Wait after independent shared-count initialization, before ids |
| Heavy/light W13 and W2 `gemm` | Entry release and PDL launch | Wait after static address/descriptor setup, before TASK/activation reads |
| Small-expert W13/W2 `phase` | Entry release and PDL launch | Wait after mbarrier/shared initialization, before counts/ACT |
| Large-M `finish` | Entry release and PDL launch | Wait before OFF/IDS/SS/Y reads; parent side-stream joins stay full-completion events |
| VDSK qkv | Entry `gdc_launch_dependents`; launch_pdl=True | X/W are the call's original inputs; VQ8 consumer already waits |
| Portfolio q1 `attn_main` consumer | PDL launch and wait before qkv loads | Metadata/address prologue uses original inputs; producer sumcast already releases at entry |
| Glue | No edit | All selected owned kernels already release at entry and wait before dependent reads |
| cuBLAS/Lt, FA3, torch casts | No binary/library replacement | Ordinary completion boundaries preserved; no invented library wait or release |

Groups: **E**, **no_attention**, **no_moe**, **no_projections**, **no_glue**.
`no_glue` is intentionally source-identical to E and is a second no-op control.
Each source/producer edit and expanded CUDA diff is recorded in [manifest.json](manifest.json).
CUDA edits:27 added /8 removed lines across four source strings; Python VDSK/extension-name
edits are separate. Source serialization makes a raw whole-file diff misleading.

## Verification

`screen.py` runs tools/local_eval in the judge image on **GPU0 only**, reusing the
65-case nearby audit set and seeds419731/427650 plus all13 official TASK cases.
Each fixture is checked eagerly and after two graph replays by the real TASK
collect/check_implementation, including routing membership, all keys, new KV cache
writes, FP32 reference and query-derived tolerances. No timing inference from sm120.
Normal and forced modes exercise portable launches and the existing explicit
Hopper-only launch stand-ins. These cannot prove native Hopper concurrency.

The standalone whole-file static screen compiles every CUDA literal for sm90a;
[native.py](native.py) additionally compiles the four changed sources with -O3 and
records the release/wait instructions in Hopper SASS. Native concurrency and its
performance still require the paired H200 receipt.

## Batched paired H200 experiment

One decode-host payload: **A=S15, A2=A:A, E, no_attention, no_moe, no_projections,
no_glue**;12 shapes /6 weighted loads;8 fresh seeds;8 rotated/swapped cold graph
replays per arm and seed.1GiB **read** flush before every timed replay and measured
empty-graph subtraction. r=arm/S15; smaller is faster. Per-load ratio mean, standard
error and seed range are retained along with absolute per-shape times.

Every arm uses the exact same original prepared S15 plans, weights and lossless
codes. Changed modules are compiled under content hashes and rebound explicitly:
E never times S15's cached MoE/attention binary by accident. Identical Triton kernels
share only a matching full JIT dependency closure; VDSK is compiled separately.
The full-layer gates compare output L2 and route membership on every fresh seed;
local TASK checks are independent. No tolerance is weakened for a failing arm.

The payload honors prepare's remaining deadline and reports missing cases/seeds as
incomplete. No native gain is claimed before its receipt. Priority3 on decode,
not-before1792972800 (2026-10-26 00:00 UTC/09:00 KST), behind priority-1 integration
and priority0 P5. Identity and results are added below after admission.

Current admission: {"base_sha256":"602548834f7481747701469b2a85154dde214bea28076c10fefce2c46b5240cb","candidate_sha256":"084e004b797991bc6f0302fc2d04efe7157998e9e6c25b0bdee1706e26d161fa","candidate":{"normal":{"fixtures":143,"eager_checks":143,"replay_checks":143,"static":25},"forced":{"fixtures":143,"eager_checks":143,"replay_checks":143,"static":25}},"lab":{"normal":{"wrapper_sha256":"f067eb263e1a67ac5277cec91b032fa33de4891ec3a12fdba9a24b23ae3797bc","shapes":12,"loads":6,"seeds":8,"replays":8,"arms":7,"host_tests":13,"static":2,"elapsed_s":35.92},"forced":{"wrapper_sha256":"f067eb263e1a67ac5277cec91b032fa33de4891ec3a12fdba9a24b23ae3797bc","shapes":12,"loads":6,"seeds":8,"replays":8,"arms":7,"host_tests":13,"static":2,"elapsed_s":32.18}},"limitations":["sm120 screens use explicit launch stand-ins for Hopper-only attention","native Hopper concurrency/performance pending reset receipt","FA3/cuBLAS and normal CUDA event completion boundaries retained"],"optimized_hopper_compile":4,"sass_placement":"placement.json; isolated mapping sass_mapping.json"}

Queued sender **1416137**, wrapper **`f067eb263e1a67ac`**. Not-before **1792972800.0**, priority **3.0**; no judge ID yet. See [queue.json](queue.json).
