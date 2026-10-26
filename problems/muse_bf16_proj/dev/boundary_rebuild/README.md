# P7 boundary rebuild: decode 16/32 and verify 16/32

Current state: **the first reset receipt is unmeasured and the exact payload is queued on the attention host**;
no rebuild is promoted. Exact base: S14 `f1c77ce6`.

The existing reset batch measures decode 16/32 ready work, programmatic overlap, CL4 geometry, and the
MoE-to-unpool launch edge. It does not measure verify 16/32 boundary attribution. `reconcile.py` reproduces from the frozen timeline at commit `ec55726`
S11's weighted **combined b1+b2 completion increments**: d16 **5.149 us**, d32 **5.924 us**, v16 **12.952 us**,
v32 **17.398 us**. The 5–6 us estimate applies to the two decode loads. These are S11 observations; S14
verify attribution comes from this new payload rather than being inferred from residence.

## Rebuilds

All six candidates are generated from exact S14 by `patch.py` (SHA256 `d7814467`). Dispatch is restricted to
batch 16/32 and q_len 1/8; every other row keeps S14 behavior.

| Arm | Candidate | Mechanism | SHA256 prefix |
|---|---|---|---|
| M | moment | One 512-thread CTA per row; four moments combine branch norm, residual and next-norm statistics in one reduction; no cluster/DSM barrier | `aacb174d` |
| E | exact | One 512-thread CTA per row; branch reduction, FP32 residual, then exact residual-square reduction; final weights loaded at use | `21488a56` |
| S | serial | Identical E kernel body with the host programmatic-serialization attribute disabled; tests wait/occupancy interference | `945283a5` |
| R | route | One 256-thread CTA per row combines pool splits, norm and top8; shared-memory hand-off replaces DSM publication | `dfb47041` |
| C | chain | E boundaries + R route in the same whole layer | `ec35d75c` |
| P | publish | Decode only: same unpool GEMV recipe, tile-ready publication; E b2 consumes complete 64-column tiles before the entire grid retires | `311dba60` |
| E1 / E2 | role ablations | Same compiled E module, b1-only or b2-only state switches | same E source |
| A / A2 | controls | S14 and same-function-handle A:A | `f1c77ce6` |

All variants keep BF16 materialization points and FP32 residuals. No producer output is read early without
an acquire. M changes summation order using the same moment identity as S14's one-round cluster kernel;
E provides the two-reduction comparison. No extra global-memory norm/combine pass is introduced.

R retains sigmoid of accumulated FP32 router hi+lo, unbiased mixture weights, integer-order max keys,
and lower-expert-ID tie breaking. It writes ordinary x/ids/weights/local IDs; it does not force an unfinished
MoERouteMeta consumer ABI onto the small-M or large-M expert paths.

## Producer publication and replay freshness

P preserves S14's decode unpool geometry (BN64, splits2, NS4, TB16/32). It adds tile flags to that original
Triton kernel, not a slower replacement GEMM. Every producer CTA resets its flag to zero, completes a CTA
barrier, then issues the original PDL release. The consumer can start only after **all** producer CTAs have
reached release. Each producer fences its output stores, barriers its threads, and release-stores ready=1.
Each consumer group acquire-loads that tile flag, warp-syncs its peers, then reads partials using volatile
L2-coherent `ld.global.cg` operations. This prevents compiler hoisting and stale per-SM L1 reads.

The fixed graph workspace therefore starts a fresh publication phase on every replay; no `%gridid` tag,
inter-call output cache, host synchronization, or later-CTA epoch assumption is used. The protocol uses the
PDL launch edge to order **all resets** before consumer polling, unlike a generic cross-CTA flag scheme.
It follows [CUDA's documented dependent-launch condition](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html).
The local publication test poisons flags to -19 or 2 between changed-input graph replays to reject stale-ready
consumption. Verify rows keep A's producer and launch dependencies; P is only a duplicate A control there.

## One-line rules for attribution outcomes

| Reset / new-trace outcome | Decision rule |
|---|---|
| Boundary residence is mostly overlap and ready work is near the stream model | Do not count the hidden wait as a boundary saving; retain A unless an arm wins the whole layer beyond A:A. |
| Serialized boundary improves the layer, exposing contention or ineffective overlap | Compare S versus E; freeze serialized execution only on its independently winning shape/role. |
| Unpooler global-retirement tail exposes b2 beyond ready work | Test P versus E2 and A; keep tile publication only if its producer tax is repaid in whole-layer latency. |
| b1/b2 ready-input work itself is excessive | Compare M/E against A, then E1/E2; choose the winning numerical form and measured role rather than combining unproven gains. |
| Route critical increment is excessive after pool completion | Compare R and C; retain R only with official gate margin and a paired whole-layer win. |
| b2 waits primarily on MoE tail or a genuine cross-stream join | Attribute that time to the MoE owner; neither P nor a norm rewrite is credited for work it cannot remove. |
| Reset CL4 wins but full-row CTA loses | Keep the qualified CL4 result; do not replace it merely because E has fewer CTAs. |
| No arm wins beyond A:A or any safety gate fails | Close these concrete recipes and keep S14; do not promote component-only or noisy wins. |

**Dispatch rule: send the complete payload regardless of the reset outcome, because it includes its own S14
baseline, A:A, ready controls and A/P traces; use the outcome table to choose what to freeze, not to omit arms.**
Consequently there is no dependent, unsent variant in this working set.

## Verification and packed evidence

- Six candidates: **normal and forced** `tools/local_eval.py` in the judge image, GPU0, each 13 official tests,
  12 captured shapes, 25 static checks; all pass. E1/E2 also pass all 13 tests and 12 captured shapes in both
  modes. `screens.json.xz` contains complete receipts; `screen_summary.json` indexes exact source hashes.
  `screen.py` only uses the standard Hopper stand-ins and a full 1 GiB int64-view cold READ. Changed native
  code is actually compiled; source/flags/compiler fingerprints govern reuse of unchanged compiles.
- Native math: **256 changed-input graph replays** across T16/32/128/256, BF16 and FP32 split inputs,
  both boundary roles and route; worst component relative L2 **8.41e-5**, route IDs identical.
- Publication: **128 changed-input/stale-flag graph replays** at T16/32; worst FP32-oracle relative L2
  **1.06e-4**. Same graph addresses replay, with poisoned flags before each replay.
- sm90a SASS: all six native specializations have zero local bytes/spills. E512 uses 46 registers, M512 57,
  R256 42. Publisher keeps LDGSTS **40/48**, HGMMA **8/8**, and shared memory at T16/32; registers are
  **48/56** versus original **48/64**. Flag reset is before PREEXIT. No H200 latency is inferred from SASS.
- Full local packed lab: **8 shapes x 24 seeds x 10 arms x 8 rotated cold READ rounds**; all gates pass,
  zero route-membership changes, worst whole-layer relative L2 **0.000950**. It uses one shared S14 prepare,
  identical source-JIT closures and native handles, except kernels actually changed.
- The first full lab reached its receipt-size assertion. `local_raw.json.xz` preserves that full observation set;
  final packing was checked separately against it. `local_receipt.txt` is explicitly labeled as repacked local
  evidence; no elapsed time or H200 result is invented. Kernel arithmetic did not change during packing fixes.

The final archive is **8,685 characters**. Every paired seed is retained as baseline time (0.01 us) and ratio
(0.0001; rounding error <= 0.00005, or 0.005%). Source hashes use 16-character prefixes; full hashes are in
`build_manifest.json`. Complete graph edges and stage labels survive. Exact O/b1/pool/route/unpool/b2 kernel
intervals/resources and the other stages' exact start/end envelopes support critical-path attribution. The
identical verify P trace is omitted; verify A remains. Ready A/M/E/R medians and sample counts are retained.
`check_archive.py` verifies all these bounds and round trips. This is local sm120 evidence only.

## Queue and handoff

New owned sender **1298945**, prefill priority **1.2**, not-before **1792972800 / 2026-10-26 09:00 KST**.
Wrapper SHA256 `2a76a8c1c1a265d2f7b6b417b4e9893552c6d35f75eaf873ca3614b2a28b97b4`;
payload SHA256 `1c47422e13faffce4de8661af4eb11ffe07c826ba92ff93d046a881d936bd2bb`.
Log: `runs/detached/2a76a8c1c1a265d2.log`; worker snapshot: `scheduled.json`. Original reset job: **a05f986396d744e7**; its packed result was displaced, as recorded below.
Earlier owned senders **3517006** (priority1.0, reset attribution/confirmation) and **975835** (priority1.1,
decode-1 chain) stay unchanged. No decode quota is used.

After dispatch, read the job with `tools/h200_lab.py read JOB`, then
`python3 problems/muse_bf16_proj/dev/boundary_rebuild/report.py RECEIPT.txt --out h200_rebuild.json`.
Use whole-layer ratios across seeds and the A:A control, with rounding bounds, to freeze winning rows.
Any P win is specific to the measured S14 producer; integration must port publication onto the current
producer and re-pair it, not displace a faster projection recipe on the strength of this experiment.
No winning integration patch exists before the H200 receipt.

Rebuild the self-contained payload with `python3 problems/muse_bf16_proj/dev/boundary_rebuild/build_lab.py`.

## Reset receipt and retry

Original job **`a05f986396d744e7`** completed, but the lab is **unmeasured**: later prefill-host prepare tuning and profiler output displaced every lab marker from the 16,000-character receipt tail. Host correctness passed; paired ratios, A:A and lab gates are unavailable. No arm is promoted or closed.

Exact-byte retry: attention-host import lab, 550-second budget, sender **4131824**, priority **1**; payload SHA256 `1c47422e`, wrapper `748b31fb`. Accepted follow-up job **`9f6849fcc12a448d`**. All existing local screens still cover the unchanged kernels. The attention host finishes its own tuning before this lab (no later prepare); its prior receipt has 1,956 characters after import tuning. `reset_result.json` records the receipt and dispatch proof. Apply the existing decision table only to the retry's recovered paired controls and gates.
