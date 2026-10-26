# P5 sliding round 2

This batch targets decode16/32 and verify1 sliding critical-path increments in
the whole S14 layer. It is not promoted. Base is exact S14 final f1c77ce6;
S15's full-verify/projection integration is a separate paired decision.

`kernels.py` extends the measured last-CTA reduction to eight query tokens,
with token-aware output/gate indexing and one warp per real Q/K/V row in
verify preparation. Query subgroups1/2/4/8 distribute the last-owner tail.
At verify1, QT8 cuts the per-warp preparation vector from20 float2 values
(QT1) to6, and the last-owner epilogue from16 rows/warp to2. Native ptxas
registers fall from254 to139 for those last-CTA specializations. These are
compiler/index facts, not latency results. The increased KV duplication is
measured against QT1/2/4 and the unchanged layer control.
The same prep is tested with compact early-history gathering and DSM output
reduction. Every path reads all window keys, appends every new KV row,
normalizes every query and captures as one attention launch.
S14's projection and gate-split choices remain identical in all arms: full-QKV
rows fuse the gate here; split-gate rows retain the existing gate GEMV/apply.
The lab reports both attention and gate increments and judges whole-layer
latency, so work shifted into the gate cannot count as a win.

`runtime.py` defines the complete three-shape experiment matrix. `build_packet.py`
packs all arms, exact S14, unchanged-JIT A/A2 controls, and the full public
checker into one attention-board import-time run. Sixteen fresh seeds x12
rotated cold-READ rounds per shape; each arm/seed must pass the public
whole-layer TASK checker with20% margin. CUPTI records QKV, attention, gate and
O critical increments. It retains replay samples, error drift and resources
in a compressed archive. The exact reset DSM shader is retained as an
experimental control at verify1, so a prep change cannot claim an unrelated win.

The payload is independent of reset outcomes: it directly compares last-CTA,
early-history and DSM for all three new shapes. It is queued behind the three
existing P5 reset senders. No first-round queue entry is replaced or absorbed.

## Reset-dependent follow-ups (ready, unsent)

Decision rules: **history E/OLD beats A:A noise with no QKV/gate displacement → send selected-QT history confirmation; otherwise drop it. DSM C/A beats A:A noise and all gates pass → send only qualifying batches in the DSM whole-layer confirmation; otherwise drop it.**

- **History decision:** if 0cec6706's best E1/E2/E4 beats OLD4 in whole-layer
  ratios beyond A:A noise and does not shift the gain into QKV/gate, render
  `followups.py history --selection '{"qt":<winning QT>}'` and send its
  twenty-four-held-out-seed whole-layer confirmation; otherwise drop this
  recipe and use the last-CTA/DSM outcomes from round2.
- **DSM decision:** for each B where f1c270e6's native DSM arm beats its A/A2
  control beyond noise and passes all gates, render `followups.py dsm
  --selection '{"<B>":<winning C>,...}'`; send the selected shapes' paired
  S14 whole-layer RESET/COOP/half/double-gang confirmation, twenty-four fresh
  seeds. If no batch qualifies, do not send this packet.

Both follow-up generators ship only used kernel families. They are payloads,
not queued senders. Their shape selections and minimum measured benefit are
explicit: no result-dependent choice is silently inserted into the live batch.
They use seeds disjoint from both the reset and round2 matrix. Exact reset
history/DSM CUDA bytes remain controls. The follow-up generator's original
mixed-cluster tests use actual C1 launches onsm120 as the explicit local
multi-CTA DSM stand-in; native residency is checked in the H200 lab.

`layer_patch.py --selection '<shape-to-arm JSON>' --out <file>` emits an
integration candidate onto exact S14 and ships only selected CUDA families.
Default proposal covers decode16/32 and verify1 last-CTA, and must earn native
paired evidence before integration. Patch choices can be composed onto S15
by integration after its reset leave-one-out result; this lane does not assume
that result in advance.

## Local admission

- `check_variants.py`: every matrix arm at its actual target shape, BF16 and
  strided FP32 partial inputs, changing graph inputs, eight repeated replays,
  counter reset; verify also short history, pages1/64, nonunit KV scales.
- `sass_check.py`: all three kernel families x8 specializations at sm90a O3;
  early-history gather and transpose precede ACQBULK in every specialization.
- `screen.py` / `screen.py --forced`: exact board packet through local_eval,
  with forced new dispatch on nearby sliding TASK cases. Native WGMMA uses
  its actual sm120 MMA math shim locally.
- `check_lab.py`: exact compressed payload with forced Hopper whole-layer
  glue, every arm at all three shapes, two fresh seeds x2 cold rounds. DSM
  native clusters explicitly use genuine C1 math and resource stand-ins onsm120;
  native multi-CTA placement/latency remains unverified until the H200 result.

Outputs live under ignored `out/`; final source hashes, admission counts and
queue identities are in `admission.json`. The local GPU cannot establish an
H200 win. Keep no follow-up send if its decision rule fails.

## Build and send commands

```sh
python3 problems/muse_attention_decode/dev/sliding_round2/build_packet.py
python3 problems/muse_attention_decode/dev/sliding_round2/layer_patch.py
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_attention_decode/dev/sliding_round2/check_variants.py
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_attention_decode/dev/sliding_round2/sass_check.py
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_attention_decode/dev/sliding_round2/screen.py
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_attention_decode/dev/sliding_round2/screen.py --forced
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_attention_decode/dev/sliding_round2/check_lab.py
```

A selected follow-up payload is sent via `tools/h200_lab.py send <payload>
--host muse_attention_decode --import-budget 650 --phase import --note <result>`.
The main round2 candidate is a single fully embedded board wrapper; send that
through `tools/okc_run.py send`. Neither path requires editing a source file
when a reset result selects a cluster or query partition. The wrapper queue
release is09:01KST, matching the existing P5 reset batch, with lower priority,
so the second-round packet could not jump the history reset packet's time gate.
After all first-round P5 receipts completed, its priority became2 to put
remaining structural measurements ahead of component-board diagnostics.

## Queue and final artifacts

Second-round packet **295a2792**, accepted as **27a914f72bfb47c8**;
sender **1029989**, current priority2 (initially4), not-before
**1792972860 (2026-10-26 09:01 KST)**. The one-minute offset from reset
1792972800 matches the earlier P5 batch's time gate and keeps this packet behind
it. Sender log:
`runs/detached/295a2792bb092012.log`.

The first three reset packets completed as **ae6cd27877cc4ee2** (global),
**b05713ecf6574ad9** (DSM) and **402819c28279466d** (history).
Their decisions and currently queued responses are below. Historical
admission hashes in `admission.json` describe the original packets; current
confirmation hashes and selected-shape gates are in `response_admission.json`.

Default integration proposal `out/s14_round2.py`: **d2711401**. Its builder
ships only the selected last-CTA family and requires exact S14 **f1c77ce6**.
Kernel sources: last-CTA **9118a17c**, early-history **c7a41b46**, cooperative
DSM **0f250847**. The exact old DSM control is **e43cb513**, and the held-out
history confirmation uses exact reset shader **0e07854b**.

No native speedup is claimed. Promotion waits on many-seed paired H200
whole-layer results and the integration lane's six-load score.

After collection, `report.py <runs/...receipt.json>` decodes the compact
archive and prints every arm's paired ratio vs A and A2, across-seed standard
error, and QKV/attention/gate/O displacement. Its confirmation ranking requires
the gain to exceed the observed A:A drift and three standard errors. It makes
no promotion or submission automatically. The decoder was exercised on the
real local archive; those local timings do not establish native rankings.

## Receipt responses

The frozen reset admission above remains in `admission.json`; current
response sources and queue identities are in `response_admission.json`.
DSM receipt **b05713ecf6574ad9** selects **B1:C16 and B16:C8** only:
stage ratios0.71927/0.84911 vsA,0.74918/0.86657 vsA2. B32's bestC4
loses1.07381, so it is excluded. The24-held-out-seed whole-layer packet
is **768d8105**, sender **114722**, priority1, attention import hosting.
It packs RESET/COOP/HALF/DOUBLE; selected-shape local admission passes
all gates, two fresh seeds x2 cold rounds, with native DSM explicitly C1
locally. It completed as **6c26b14fdd28458b**: v16 cooperative variants
lose1.063-1.153, RESET is neutral; v1 nominal COOP0.94983 is **unmeasured**
because identical C16 DOUBLE/COOP1.03901 (SE0.00300) fails its control.
No new P4 promotion is made. Machine audit: `dsm_confirmation.json`.

The v1-only repeat is **6a0b3f0987e94ba5**, wrapper **51847d1b**,
35 disjoint seeds x14 rounds, with A/A2/A3 and the C16 duplicate retained.
Future generated packets now recapture graph pools and rotate capture
ownership per seed; CUPTI increments still describe the first capture only.
Seven arms x2 fresh seeds pass the actual local recapture/TASK/cache smoke.
The reporter detects identical-recipe aliases and prevents promotion when
their displacement exceeds baseline drift plus three standard errors.

Follow-ups now validate each arm's new cache immediately after that arm
replays. The original round2 packet is immutable and retains its original
gate order. Follow-ups also export an explicit `lab` for h200_lab admission
and cap their retained output at8800 characters (generic wrapper limit9000).
These fix admission/measurement gaps without changing CUDA math.
`followups.py selected --selection '{"16,1":["last",1,8]}'` prepares
one selected native winner per round2 shape with24 held-out seeds; send
only shapes that pass the receipt's A/A2 and correctness decision.
The three-shape local selected-mode smoke passed; its default arms are
functional examples, not measured selections.

The global lab **ae6cd27877cc4ee2** has no captured-path TASK gates and is
unmeasured. Its exact-kernel repaired4-shape/8-arm packet is **da61a8f2**,
sender **112961**, priority5; see `../cluster_sliding/README.md`.
History **402819c28279466d** is closed by its decision rule: bestE4/OLD4
0.997414 (SE0.001244) does not exceed A:A1.009555 (SE0.006105;
range0.98824-1.09546) or3SE. All16 fresh-seed gates pass, and clocks prove
the prewait work. No history confirmation is sent. Its prepared generator
now retains exact OLD4 CUDA3815beab atQT4 instead of the round2 shader;
the corrected packet passes local whole-layer admission.
Round2 and resident-history reset decisions still await receipts.

Final local admission: **100 fresh-input/partial/counter cells**, **24 sm90a
specializations**, **80 exact index-coverage cases**. Exact board wrapper:
normal/forced each **16 TASK /12 graph shapes /7 static checks**. Default
whole-layer integration proposal: normal/forced each **13 TASK /12 graph
shapes /24 static checks**. Main payload and both conditional confirmations
pass their complete selected-arm matrices with forced layer glue, two fresh
seeds and two cold rounds, full public TASK checks with20% margin, and one
attention launch per changed arm. DSM multi-CTA placement is explicitly a
C1 stand-in locally. No local timing is used to select a native winner.
