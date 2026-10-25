# P3b second-round batch

Base: S14 final `f1c77ce6`, exact nf CUDA `0a31e251`. Target remains whole MoE
132.6 /158.5 us at M128 /256. No production patch or H200 win is claimed here.

`build.py` **b3a33d2e** generates payload **67eca44e** with five arms and24 weight/router seeds x3
fixtures/M: S14, TRANS64, TRANS128, CTA2, CTA2S3. One payload also measures isolated heavy W13/W2
and packing service on a fixed fixture. These phase times are not critical-path increments.
`manifest.json` pins the payload and all CUDA sources. Generated payloads/screens/CUDA are ignored.

| Arm | Ownership and transport | CUDA sha256[:8] |
|---|---|---|
|TRANS64|One WG: tokens are register A; a64-output decoded B tile is shared across every expert-token group.16KB double-buffered B, lossless timed A packing.|3b837a91|
|TRANS128|Two WGs collectively decode128 output channels; each consumes different64-token groups.32KB double-buffered B; half as many channel tiles.|3b837a91|
|CTA2|Two consumer WGs own adjacent64-output tiles and share one native-wide activation slice inside a CTA. Packed Q/scale staging is private per WG; X is copied once.|5aff055b|
|CTA2S3|CTA2 with a three-slot copy ring for MAX64 only; larger classes keep S2/S1.|d87d1a9a|

All use ordinary CTA launches, live routes, parallel width classes, N16G4 light experts, BF16 activations,
FP32 accumulation, and F4 finalization. No cluster, global queue, residency cap, or separate producer WG.
Each packed weight tile is read and decoded once for all expert tokens, through256 tokens. Nearby257
naturally uses another tile. TRANS loads each scale word once per lane quartet and broadcasts it with
SHFL; `sync.json` proves exact Q/scale source multiplicity1. Weight banks retain packed NVFP4 formats.

TRANS W13 writes BF16 SwiGLU directly to W2's fragment layout. There is no hidden ACT-to-fragment
repack. Each warp writes a W2 norm partial; the CTA sums its columns into the existing SS64 row
format. N128 stores the128-column sum plus one zero slot. No new norm atomic is introduced.
Invalid token groups read a constant2KB zero fragment; all WGMMA groups execute uniformly. This avoids
the C7513/C7520 serialization found and removed during development. Zero fragments contain no input data.

This differs from the earlier N64 loss (`858ff2fa`): that enlarged activation staging to~85KB and kept
activation rereads. TRANS stages decoded weights in16/32KB and keeps activation fragments in registers.
It also differs from `9cd95e72`'s producer/consumer recipe: the same WGs decode and consume, with no
per-stage full/empty mbarrier protocol. These differences are hypotheses, not measured H200 gains.

## Verified locally

GPU0 only, judge image (`tools/img`):

- Each of four generated component adapters passes `tools/local_eval.py --tests-only`:5 official tests
  +4 static checks. Each forces its new plan on sm120, with matching `mma.sync` fragment math replacing
  only Hopper WGMMA. `screen_{trans64,trans128,cta2,cta2s3}.json` pins each screened source.
- `check.py` / `validation.json`:995 checks,98 input-transition cases, sizes33/127/128/255/256/257,
  all count boundaries, hot/empty/real routes, poisoned ACT/Y/SS/output and fragment scratch, captured
  replay, plus15 separately captured local_start64 checks. Worst FP32 L2/row0.004129/0.004782.
  CTA2/S3 are bit-equal to S14 on98/98 cases; transposed forms differ in accumulation/norm order.
- `entry.json`: complete weights=None entry,24 seeds x3 =72 fixtures/M/arm, every TASK gate passes;
  worst L2/row0.004132/0.005018. Local timing used no cold flush and is not promotion evidence.
- `audit.py` / `resources.json`: O3 sm90a nvcc +cuobjdump, zero stack/spills/local-memory SASS in every
  new Hopper kernel; no C7512/C7513/C7519/C7520 warning, group wait1 retained. Source/SASS hashes recorded.
  TRANS64 registers W13/W2 at capacity64:128/134; capacity256:248/250. TRANS128 capacity256:240/248.
- `sync_check.py` / `sync.json`:2,758 least-completion register/B-slot checks, bijective decoded-B and
  W13-to-W2 layouts, exact weight-source coverage; additionally19,278 S1/S2/S3 copy-ring checks from the
  retained parent checker. Wrong wait2 is rejected. These are protocol models, not hardware race tests.

The ordering argument follows [NVIDIA PTX's WGMMA synchronization contracts](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions-wgmma-wait-group):
A buffers alternate between consecutive committed groups. Wait1 retires a buffer before its next write.
Before replacing B slot k%2, every consumer WG has retired its previous reader; a CTA barrier joins those
frontiers. All decode stores precede a second CTA barrier and generic-to-async proxy fence. Epilogues follow
wait0. S3 copies retain the parent ring's all-WG retirement barrier and empty tail commits.

## Traffic and time model

`model.py/json` counts the exact fixed fixtures printed by `8ee003db`; their heavy-token distributions
are not the multi-seed averages behind the quoted1.6x/2.2x weight multiplicity. The 171/403MB figures in
`3e8996f8` are W13 activation traffic; W2 adds85/201MB. Keep those phases separate.

| M | Control padded heavy X MB (W13+W2) | TRANS64 | TRANS128 | CTA2/S3 |
|---|---:|---:|---:|---:|
|128|251.66|402.65|201.33|125.83|
|256|553.65|805.31|402.65|276.82|

All remove weight multiplicity; CTA2 halves X reads. TRANS128 also reduces X on these fixtures, while
TRANS64 trades more padded reads for smaller staging and better register residency. Padding and the
additional timed pack are included. These are scheduled bytes, not DRAM counters; L1/L2 reuse is unmeasured.

Anchoring transport service to the measured DIRECT arm and preserving its instruction cost gives
CTA2~177.8/248.7us with unchanged overlap, versus146.0/133.3us with full heavy/light overlap. TRANS128's
analogous range is201.0/281.7 to146.0/166.2us; TRANS64's is worse. Added STS/SHFL, issue behavior, occupancy,
and eliminated staging costs remain unmeasured. These scenarios do not establish the target. At M128,
fixed-fixture light service alone is~130us; light improvements or a different overlap schedule are also needed.

## Dispatch and decision

One independent follow-up is queued on the MoE host: sender **71a7126383e39952**, PID **1073793**,
priority10, `--not-before 2026-10-26T00:01:00+00:00` = **1792972860** (09:01KST), no judge job ID yet.
This matches the unchanged reset batch's existing09:01 eligibility, one minute after the requested reset
epoch1792972800, so this second-round packet cannot jump ahead of reset sender **f867a8b74a243d1e**
(priority1, PID3408321). No existing sender was changed. Exact dispatch metadata: `queue.json`;
progress: `runs/detached/71a7126383e39952.log`; frozen snapshot: `runs/sent/71a7126383e39952/lab_payload.py`.
The tool sends scored while quota remains, then checks. All four variants share one run.

**Decision rule:** send every arm independently of the reset; if reset favors S3, prioritize CTA2S3's paired
comparison, otherwise CTA2; select the fastest arm separately at each M only when paired whole-MoE ratios beat S14, then require the integration lane's whole-layer pair before promotion.
Nothing is left unsent waiting for the reset. The reset result and this packet's H200 timing remain unverified.

Rebuild: `python3 .../round2/build.py`. Screen: `CUDA_VISIBLE_DEVICES=0 tools/img python tools/local_eval.py
.../round2/screen_<arm>.py --tests-only`; captured transitions: `CUDA_VISIBLE_DEVICES=0 tools/img python
.../round2/check.py`. The builder regenerates its retained parent dependencies and pins S14's full source SHA.

Retained sources: `build.py`, `transpose.cu.inc`, `runtime.py.inc`, `check.py`, `audit.py`, `model.py`,
`sync_check.py`; their JSON siblings are current evidence, not alternate implementations.
