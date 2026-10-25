# P5: early sliding history on S14

Target: decode-1 sliding critical increment <=6–8 us in the whole layer. Exact
patch base is `cand_r3_best7_s14_final.py`, SHA256
`f1c77ce692055c89474eab38bd2b45ec1ea34c94cf5a93d1992dbbcf448a3cac`.
This experiment is not promoted; H200 performance is pending.

## What changed

`history_build.py` derives the measured non-spinning last-CTA implementation
(`3815beab` CUDA, paired H200 `7e93cc81af17489a`). Its history gather already
uses volatile inline-PTX cp.async before PDL wait: all 16 initial copy
instructions precede the first ACQBULK in sm90a SASS. Consequently the older
warp-kernel c0fd644c phase breakdown does not establish that the newer winning
last-CTA path still exposes 5.3 us of post-QKV history latency.

The new path finishes the one history tile and its LDSM.T/PRMT V transpose
before the PDL wait. RoPE metadata runs while the asynchronous gather is in
flight. Page-table/sequence loads use explicit volatile PTX. Only QKV, its new
cache row, normalization and the attention math consume producer outputs.
Raw V storage becomes KN/VN after transpose, eliminating its long lifetime.
One N128 tile per split covers all history with 16 splits; the new row uses the
unchanged virtual-token math. The S14 split-gate producer choice remains shared across every arm; when it
is active, its gate GEMV/apply remains a separately attributed gate stage.
Both attention and gate increments are retained. The launch rejects windows exceeding ns*128,
never skips keys, and retains the measured last-CTA publish/reduce/reset.

Dynamic shared memory falls from 53,504 to 28,928 bytes. cuobjdump reports
2,048 bytes static allocation (larger than ptxas's 1,024-byte usage report);
use the conservative totals 55,552 ->30,976 bytes for residency reasoning.
Register allocation and the exact instruction-order evidence are retained in
`sass_proof.json`. `residency.json` checks the active 200-CTA qkv3 producer:
two producer plus two compact attention CTAs fit the static SM budget
(225,792 shared bytes, 49,152 registers, sixteen warps). This is static
admission, not measured co-residency.
The lab measures QKV durations as well as attention's critical increment.

`patch.py::patch(source)` is a reviewable Python builder onto exact S14.
It enables only B1/q_len1/sliding. All history reads and preparation occur in
the timed call. Scratch counters are allocated once eagerly and reset by the
last CTA, with no metadata, combine, reset or prefetch launch in the graph.

## One batched H200 packet

`build_lab.py` builds the attention-board candidate and compressed lab payload.
The timed candidate uses the new kernel only at decode-1 sliding; the lab
synthesizes the public decode-layer weights, runs exact S14 prepare once, and
shares all baseline-selected plans, banks and unchanged JIT objects.

Seven arms, sixteen fresh seeds x12 rotating cold replays:

- A/A2: identical S14, including its direct-partials decode-1 path.
- OLD4: the measured old last-CTA, 128 CTAs.
- E4/E2/E1: compact early history, 128/64/32 CTAs (2/4/8 query heads per CTA).
- L4: same compact layout, history gather/transpose after producer wait.

E2 tests the middle residency/duplicate-read tradeoff. All changed arms must
capture exactly one attention launch. Each receives an actual projected-QKV
TASK attention/cache gate, finite checks on every fresh whole-layer replay,
the full public whole-layer TASK checker with 20% margin on every seed,
1%/exact-route admission against OLD4, and paired whole-layer ratios vs both A
and OLD4. CUPTI retains stage intervals and critical-path increments. Separate
instrumented kernels record history issue/readiness/transpose, PDL wait,
QKV preparation, attention math and last-CTA epilogue. The diagnostic clock
buffer is cleared before each replay so retired CTAs cannot retain stale
last-owner timestamps. Clock kernels do not provide paired performance.

The packet leaves existing reset senders f1c270e6 (DSM attention sweep) and
0ccc2548 (old qualified host / full decode experiment) untouched.

## Rebuild / admission

```
python3 problems/muse_attention_decode/dev/early_history/history_build.py
python3 problems/muse_attention_decode/dev/early_history/patch.py
python3 problems/muse_attention_decode/dev/early_history/build_lab.py
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_attention_decode/dev/early_history/sass_audit.py
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_attention_decode/dev/early_history/check_lab.py
```

Normal/forced whole-layer screens use `screen.py --candidate out/s14_early_history.py
--out out/normal` and `--forced --out out/forced`; directories must exist first.
`check.py early 1|2|4` and `check.py late 4` exercise the actual sm120 math shim
with BF16/strided FP32 producer inputs, pages 1/64, edge keys, nonunit scales,
changing graph inputs and eight consecutive counter-reset replays. sm120 does
not execute native Hopper WGMMA or establish H200 overlap.

Current admission hashes, counts and queue identity are recorded in
`validation.json` after the final source's screens finish. No new H200 job ID
or paired ratio exists before reset. Next: collect this packet, select by
whole-layer E/OLD and E/A (beyond A:A noise), check QKV displacement, and
integrate a qualified winner onto the S14/S15 composition.

The first local payload attempt rejected OLD4 solely on an invented 3% A-relative
drift cutoff. Replaced that unsupported rule with the complete frozen public
TASK checker (20% stricter bounds), retained A-relative drift, and required
E/L arms to match OLD4 within 1% with exact routes. All seven arms pass both
fresh local seeds; E4/E2/E1/L4 match OLD4 exactly. This changes validation to
the actual contract and adds a stronger same-algorithm control; no result
was promoted from the failed screen. Full native sixteen-seed checks remain pending.

## Final packet / receipt status

- CUDA: **0e07854b**; generated S14 layer candidate: **eedd5fb6**.
- Standalone diff `s14_early_history.patch`: **5f1d2d43**, applies onto **f1c77ce6**;
  `git apply --check` passes. It is experimental, not an integration promotion.
- Immutable attention-board packet: **0cec6706**, payload **656dea2a**.
- Live reset worker **3549805**, priority3, not before **2026-10-26 09:01 KST**;
  `runs/detached/0cec67066689787d.log`. No H200 job ID yet.
- Existing workers **2411468 / 3038721** still own **0ccc2548 / f1c270e6**;
  no duplicate or replacement sender was created.
- Final normal/forced layer screens: **13 TASK / 12 graph / 25 static**, both pass.
  Attention packet: **16 TASK / 12 graph / 8 static**, all pass. Component
  variants: **40 cells**, all pass. Exact compressed payload: seven arms,
  two fresh seeds x2 cold rounds, all full-reference checks with20% margin pass.
- H200 performance: **unmeasured**. The prior qualified last-CTA control
  `7e93cc81af17489a` was **0.95868 vs S11**, **0.98017 vs direct-partials**,
  critical increment **15.10 ->11.10us**. These numbers are not this experiment's.

All new checks and resource facts are in `validation.json`, `sass_proof.json`,
`residency.json`, `component_validation.json` and the local receipts. Next is
this single packet's paired native result and its QKV/attention/gate timelines;
select and hand integration a winner only after that evidence lands.
