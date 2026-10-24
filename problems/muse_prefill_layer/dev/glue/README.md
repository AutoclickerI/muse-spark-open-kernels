# Prefill glue portfolio

Ready: `problems/muse_prefill_layer/cand_glue_prefill.py`.
Frozen SHA256 **2a2e3914ea26756fc2df828f8c58cc88d0157171dcff880dec50c55290ebdbef**.
One242894-byte file; no repository imports. Parent is board1.434x
`cand_score_portfolio_routing.py`, SHA2b215f40e76ed7ef42361a154d877fd8436f7d891f261836f860353bd09da9ba,
receipt a30fb411b0c74ffb. No parent/queued source edited, no judge job or commit by this lane.

## Inventory and scope

[Direct H200 table](../../../../docs/reference/prefill_glue.md) covers both a30fb411 and
99820fa6, every visible named glue kernel and all three loads. Missing top12 profile
entries stay N/A; zero only marks a kernel absent from the selected path. Local expert
pair/touched-expert counts are not in the profiles, so their byte floors remain ranges.
The boundary pair dominates visible activation glue:851.7–855.2us at16K, versus671.1us
traffic floor, about159ms /125ms per first turn. Separate dequant is another118–121ms in
these profiles, already part of expert-stage accounting; do not subtract it twice.
The previous235ms remainder is not all removable overhead. Dense/dequant/expert algorithm
searches remain with their owning lanes; this run measures new glue interactions only.

## Choices and semantics

- `('stock',)` retains the full incumbent producer and consumer.
- O/unpooler `('gemm',tile)`: tiled BF16 GEMM, BF16 rounding, per-output-tile norm partials;
  the last CTA for each token block uses acquire/release counters and completes full-row
  branch RMSNorm, residual update, next RMSNorm and BF16 output. One GEMM/epilogue kernel,
  no spin/global resident barrier/second norm kernel. Scratch BF16 values are still written
  and read: this removes a launch, not all irreducible row-reduction traffic.
- Boundary `('vector',rows,warps)`: retain the proven library GEMM, change the existing
  two-RMS residual kernel's row/warp geometry and coefficient reuse.
- Pooler `('gemm',tile)`: same last-CTA completion with the actual pre-expert norm gain,
  preserving the GEMM's BF16 boundary. Cloned expert consumers gather already-normalized
  rows or alias their normalized input; every expert/post/residual norm remains exactly once.
  The tuner compares producer plus real gather consumer, counting the extra materialization.
- Router `('gemm',tokens,sort)`: swap-AB BF16 hi+lo products accumulated in the same FP32
  accumulator; sigmoid, exact top8 selection, renormalized weights and local atomic slot
  metadata in its GEMM epilogue. No512-wide logit materialization or separate top8 kernel.
  Token tiles8/16/32, iterative-max or sortable64-bit topk (including exact index tie break).
- Router `('sort',rows,warps)`: retain the proven FP32-output GEMM and replace only its
  top8 consumer. Model routing uses sigmoid and weight renormalization, not softmax.

GEMM tiles `(BM,BN,BK,warps,stages)` are `(64,128,64,4,3)`, `(64,256,64,8,3)`,
`(32,128,64,4,3)`. Boundary vector choices1row/8or16warps,2rows/8warps,4rows/16warps.
Hopper's router uses3 stages; sm120's identical BF16 math uses1 because its3-stage MMA
pipeline exceeds local shared memory. The actual3-stage WGMMA path is cross-compiled.

## Tuner and interface

H200 prepare seeds the receipt-proven15 dense,3 expert and8 attention choices. It queries
Lt descriptors on synthetic inputs and catches/logs any unavailable optional ordinal.
This avoids rerunning the old dense/expert/attention portfolio and reserves the budget
for glue. Local sm120 retains the parent's portable attention/expert controls.

`_glue_boundary(x,stream,st,'o'|'unpooler') -> (BF16 hidden,FP32 stream)`;
`_glue_route(h,local_start,st) -> (ids,probs,slot_e,slot_pos,counts)`;
`_glue_pool(h,st) -> (pooled,already_normalized)`. When normalized, call
`_glue_ready_experts_selected`; otherwise call the unchanged `_experts_selected`.

Own synthetic banks only, three fresh inputs, read-only128MiB L2 flush, CUDA graphs,
four-pair coarse trials,20-pair finalists, shared230s deadline with per-key budgets.
The1st true GEMM epilogue is tried early. Independent FP32 references use the BF16
activation boundaries and2.4% global/12% row admission (0.8 times the layer's3% minimum).
Residual checks isolate the update instead of letting the carried stream hide error.
Router checks use the judge's score/strong-expert/local-count allowances, exact histogram
and slot-permutation integrity. All optional graph/oracle/timing/finalist exceptions
reject/log; wrapper failure keeps the incumbent. No judge input or output is retained.
Every trial/final time and all12 picks fit in `GLUE_LAST_SUMMARY` printed last.

## Verified locally: sm120 /GPU0

- Official final file: **5/5 tests,8/8 fresh-input graph shapes**, **48.111s prepare**.
  All78 optional configurations and36 finalists were reached; no rejection in this screen.
  The6906-character compact table survives the final16000-character receipt tail.
- Direct M33 andM16384:60 configurations,540 fresh correctness admissions/replays,
  1200 cold timed pairs;18 counter buffers return to zero. All pass.
- Forced complete layer with GEMM epilogues enabled simultaneously:5/5 official tests.
- Forced Hopper prepare reaches the actual seed branch; strict Lt stand-in covers10
  ordinal queries/forwards, with old raw entry points bound to the real native module.
  Six actual3-stage router branches are forced through the one-stage portable kernel.
- Normalized `humming_masked`, `dg_w8`, `raw11` consumer graphs match their original
  norm+expert paths (0.236%,0%,0% relative L2 in the final safety run).
- Case construction, capture, numeric, timing, finalist and tuple faults reject independently;
  expired deadline and wrapper AttributeError keep the incumbent.
-60 actual-signature SM90 specializations compile:48 WGMMA,12 MMA.sync; max shared69632B.

At16K, whole producer+consumer local paired medians: O2005.0->1997.8us (ratio.9969),
unpooler5122.6->5085.2us (.9972), router536.6->530.5us (.9886). Pooler keeps stock.
The true GEMM epilogues lose locally; only small vector/topk gains appear. These are
shared sm120 observations, not an H200 gain. Replay overhead is included on both sides.

Evidence: `runs/screen/glue_prefill/` (ignored): inventory.json, screen.log,
`muse_prefill_layer.cand_glue_prefill.json`, validate33/validate16384.json,
forced_tests.json, safety_final.log, compile90_final.log.
Use `CUDA_VISIBLE_DEVICES=0 tools/img python` with `tools/local_eval.py <candidate> --json <path>`;
direct tests are `dev/glue/validate.py [candidate] --tokens 33|16384`, `safety.py [candidate]`,
and `validate.py [candidate] --compile90`. `build.py --output <new filename>` is deterministic
and refuses to overwrite a changed candidate. Do not regenerate the frozen file.

H200 execution/choices/speed/cold prepare remain unverified. In serving, O's TP all-reduce
must precede its nonlinear boundary: its one-GPU GEMM fusion cannot cross that collective.
The vector boundary is usable after the all-reduce. Unpooler fusion requires the existing
expert reduction complete; retain every serving collective. Concurrent replay needs
independent counter workspaces. Serving agreement and integration remain unverified.
