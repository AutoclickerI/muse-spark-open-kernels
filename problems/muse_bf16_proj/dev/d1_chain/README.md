# P7 decode-1 glue second round

Build a full decode-1 chain attribution on **S14 `f1c77ce6`** while the 09:00 KST reset batch waits.
Retry5a95984a completed with biased A:A controls; no preload gain is qualified. The boundary/confirmation reset job is **4c86853a953b4eca**. This lab is independent of its outcome.

`patch.py` builds five exact-S14 candidates, restricted to decode-1 sliding/full. Every branch/partial from a
predecessor remains after the dependency wait. Inline PTX pins immutable loads where nvcc otherwise sinks them.

| Arm | Change |
|---|---|
| A / A2 | S14 and a second graph using the same function handles |
| D | Prior decode-1-full immutable-preload patch; historical control, unchanged |
| W | b1 + route + b2 retain all immutable weight vectors/metadata before wait |
| W1 / WR / W2 | The same compiled W kernel handles, one role enabled at a time |
| L | Alpha/beta/post-weight and route metadata early; final `prew` explicitly loaded after wait |
| H | L2 prefetch requests before wait, all actual weight values loaded after wait |
| S | W plus the original timed-input residual loaded early in b1 |
| C | W with four CTAs per row in boundaries and route rather than eight |

b2's residual comes from b1, so S leaves it after the wait. An early-release chain does not prove that earlier
producer ready. All original arithmetic and precision remain; C changes norm reduction order.

## Existing receipt trace: what is actually known

`receipt_trace.py` recovers canonical observations from git into `existing_receipts.json`.
The available per-role timestamp trace **9845398be06b4c37 is S13 `958f1258`, not S14**. Its decode-1 sliding
original arm has b1 / route / b2 residence **6.048 / 11.456 / 6.336 us**, with **4.000 / 7.872 / 3.552 us**
overlapping their immediate producer, leaving completion tails **2.048 / 3.584 / 2.784 us**. These are overlap
proxies, not measured instruction-level wait cycles. No S14 per-role timestamp trace was present in the existing
PDL receipts. The new lab therefore captures S14 directly; it does not relabel the S13 trace.

The actual S14 paired receipt **0f74021a2c53445e** gives prior D full latency ratio **0.98129** (A:A **0.99939**),
16 seeds. Those graphs did not share unchanged JIT handles; the reset confirmation remains required. The broad
first-weight-register-tile failure was a projection pipeline change (48 -> 150 registers, LDGSTS 40 -> 8),
not evidence that the glue-vector preloads themselves have excessive register pressure.

## Compiled ordering and resources

`audit_sass.py` compiles actual sm90a sources and checks the selected CL8/CL4 one-round specializations.
All selected kernels have **zero local bytes / spills**. Before-wait LDG/CCTL instructions are retained in
`sass_validation.json`; the ordinary S14 CL8 kernel has none. Ordering is established from the actual SASS,
not from source placement. In L both early and late value loads use volatile PTX: ordinary const loads written
after the wait were hoisted before it in the first prototype, so that prototype was replaced.

| Kernel | S14 | W | L | H | S |
|---|---:|---:|---:|---:|---:|
| CL8 boundary registers | 57 | 57 | 57 | 63 | 59 |
| CL8 route registers | 64 | 64 | 64 | 64 | 64 |
| CL4 boundary registers | 80 | 80 | 80 | 80 | 80 |

The C4 arm reuses W's native module. D also shares it because its GPU source is byte-identical. W1/WR/W2
share W's Python namespace and native module, with separate role-state switches. Every unchanged Triton
kernel and its dependency closure is borrowed from A to avoid the earlier JIT-handle pairing error.

## Local evidence

Screens run in the judge image on GPU0 using `tools/local_eval.py`, with a cold READ of the same 1 GiB via
an int64 view to avoid an 8 GiB uint8.sum temporary. The standard best7 stand-ins exercise unsupported Hopper
attention/GEMM launch branches; actual native cluster glue executes unchanged on sm120.

- Five full candidates: normal and forced, each **13 official correctness tests + 12 captured shapes + 25 static
  checks**, all pass. Source fingerprints permit reuse only for unchanged, previously compiled native sources;
  changed sources are compiled. Final source hashes are in `build_manifest.json`; complete screen receipts are in `screens.json.xz` and their compact index in `screen_summary.json`.
- W1/WR/W2: separate normal and forced `local_eval` runs of every official test and captured shape, with one
  shared prepare per mode and role switches restricted to the same decode-1 rows.
- All five actual native variants: **192 changed-input graph replays**, including BF16/FP32 partials, router
  inputs, changing local-start values, zero rows, and the sliced N1024/S8 unpooler used at decode 1. W/L/H/S are
  bit-exact; C's worst native relative L2 is below 0.001; route IDs match exactly.
- Full packed lab: **two windows x 32 seeds x 11 arms x 10 rotated cold READ rounds**, all paired gates pass,
  zero route-membership changes, worst whole-layer relative L2 **0.001546**. Six arms have five-replay CUPTI
  traces plus all graph nodes/edges and programmatic edge ports. b1/route/b2 also have same-handle, ready-input
  standalone graphs to separate actual work from producer overlap. Unprofiled event controls expose CUPTI
  overhead. The archive retains every seed, kernel observation and graph edge; only event times are quantized
  to 0.001 us. Local archive **6,135 characters**, safely below 9 KB; `check_archive.py` verifies the round trip.

GPU0 exhaustion occurred when several lanes' full-layer prepares overlapped. Those failed validation attempts
were kept local and rerun once capacity returned. No checks or fixtures were weakened. Local ratios are not
used to pick H200 winners.

## Dispatch and integration rule

**Dispatch rule: send this entire lab after the existing P7 reset batch regardless of its result; it measures raw
S14 with its own prior-D control and therefore has no reset dependency.** It uses one import-time, synthetic
prefill lab (550-second budget); decode quota is left to integration and P10b.

Queued sender **975835**, prefill wrapper SHA256
`d490ac854840bcd39b4b5716092b7118d12ab73077c5aece6266d819eb34ac6a`, priority **1.1**.
`not_before=1792972800` (**2026-10-26 09:00 KST**). Original reset job: **841077efefb343a7**; it contains no LAB diagnostic, as recorded below.
`scheduled.json` records the worker; log: `runs/detached/d490ac854840bcd3.log`.
Payload SHA256 **`a84f0d56828e38b8b83d93cf6ef6974b5865fe84b3cd7b746816379beccf9e55`**.
The prototype patch builder is `patch.py`, SHA256 **`b6d4954e193fc576d32a6fb6dba9c9771ef982be6cede60aefd7a7240723a4f7`**,
and applies onto exact S14 `f1c77ce6`. It is not a qualified integration win.

After results, `report.py RECEIPT.txt --out h200_chain.json` reconstructs every observation, paired per-shape
and 46/16-weighted decode-1 ratios, role completion increments, producer overlap and immediate DAG predecessors.
Only freeze an arm/shape with clean gates, no unexplained membership changes, and a paired gain that exceeds
A:A spread across seeds. Compare W1/WR/W2 with W before combining wins. Then screen a concrete patch onto
S14 and hand it to integration for the current incumbent's whole-layer pairing. Keep or drop the separate
metadata-only patch from the reset result; this lab does not assume it won. No winning patch exists yet.

Build/recheck:

```
python3 problems/muse_bf16_proj/dev/d1_chain/patch.py
python3 problems/muse_bf16_proj/dev/d1_chain/build_lab.py
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_bf16_proj/dev/d1_chain/check_lab.py
python3 problems/muse_bf16_proj/dev/d1_chain/check_archive.py
```

## Reset receipt and retry

Original job **`841077efefb343a7`** completed, but the lab is **unmeasured**: the prefill receipt contains no LAB marker (the earlier displacement diagnosis was not established). Host correctness passed; paired ratios, A:A and lab gates are unavailable. No arm is promoted or closed.

Exact-byte retry: attention-host import lab, 550-second budget, sender **4131717**, priority **1**; payload SHA256 `a84f0d56`, wrapper `4de4a050`. Accepted follow-up job **`5a95984a640d410e`**. All existing local screens still cover the unchanged kernels. The attention host finishes its own tuning before this lab (no later prepare); its prior receipt has 1,956 characters after import tuning. `reset_result.json` records the receipt and dispatch proof. Apply the existing decision table only to the retry's recovered paired controls and gates.

## Retry decision

Retry `5a95984a640d410e`: all gates pass, but A:A S/F 0.98549±0.00313 / 1.01603±0.00539 (95% mean CI). W S/F 0.98648/1.00369 and prior D full 0.98549 do not qualify beyond control drift. No arm is promoted or family closed. A balanced same-capture remeasurement remains open; the new MoE payload uses that protocol. Full retained verdict: `receipt_decision.json`.
