# Small-M best7 glue cells

One native CUDA launch per fused stage, BF16 weights and activations, FP32
accumulation/router hi+lo/residuals. No expert format changes or prepared
expert expansion. Initial scratch contains counters only; every replay rewrites
all input-dependent values and resets all arrival counters.

`best7_glue_cell.py` appends to `cand_r3_best7_scale.py`. Its
`BEST7_GLUE_CELLS[(batch,q_len,window)]` is empty until a whole-layer H200 win.
Values:

- `pool`: boundary 1 in the pool/router GEMV prologue, then FP32 routing and
  pre-expert norm in its last-CTA epilogue. One output tile writes the residual.
- `tail`: unpooler GEMV plus boundary 2 in its last-CTA epilogue.
- `all`: both of those cells.
- `epi`: o GEMV + boundary 1, pool/router GEMV + routing/pre-expert norm,
  and unpooler + boundary 2. This avoids the repeated full-row prologue norm.

The native MMA path is shared by SM90 and SM120; only the attention calls need
local portable stand-ins. Arrival counters use device-scope acquire/release
atomics after every producer thread's device fence. Losing CTAs return; there
is no resident-grid assumption or spin barrier. Token tiles are 8 or 16; the
single-token norm epilogue uses the whole CTA. Each graph call site needs its
own FusedGlue instances if it can execute concurrently.

## Measurement

`prof/lab/best7_fused_glue.py` freezes the current best7_scale source and its glue
table. Projection choices and automatic attention choices are explicitly fixed
to receipt c0e74bed, using the incumbent's native Lt/cuBLAS/custom calls. Both
arms use the exact best7_scale recoded NT1 expert reader and original host NVFP4
bank; the bank's layout is checked against the current packer first. The host's
phase-instrumented expert device kernel is not timed in either arm.

Missing dense, router, gate and norm weights use seed941732 and model
scales/mean-aligned routing. The host supplies real expert weights/scales and
post-expert norm. These are synthetic-dense whole-layer measurements, not
claims about the exact decode checkpoint. Cases: decode1/16/32 and verify1,
sliding/full, context80000/page1/soft/agentic. Three changed input streams at the
same graph addresses, same inputs across all arms, EP owner changes, 6 AB/BA
replays with a cold256MiB flush before each replay, A:A per case. Pool, tail,
all and epi are compared separately against the same incumbent. A whole-layer
relative L2 gate of1%/worst-row5% applies to hidden and residual *updates*;
independent FP32 routing enforces the layer's score, keep, count and distinct
expert gates. Output tensors/counters are checked again after timing.

## Admission and status

- Native component validation:72 changed-input graph checks, two token tiles,
  M1/8/16/32, BF16/FP32 producer ABI, split counts1/2/16, FP32 projection and
  norm/router oracles; max final relative L2 6.55e-5. Every counter reset.
- Forced Hopper full-layer screen:96 pairs (8 cases x4 cells x3 fresh streams),
  max hidden/residual-update relative L2 0.004515, no route flips. Native glue
  executes locally; attn_main and FA3 launch stand-ins exercise Hopper glue.
  Host5/5 tests and6/6 static checks pass. Payload/drop-in name and signature
  checks pass. Native SM90a8 specializations compile with no register spills.
- H200 job80b70f7f20fa41c6 loses every arm by1.51–17.63x. The pool prologue
  repeats a full-row norm576 times per token; last-CTA boundaries serialize
  up to8 rows into one CTA. Local work-deletion ablations reproduce this; PDL
  on/off is neutral. This mapping is closed and disabled. It does not rule out
  fusion with a cooperative reduction. [Mechanism and evidence](audit/README.md).

Build: `python problems/muse_decode_layer/dev/fused_glue/build.py`.
Local native oracle: `tools/img python problems/muse_decode_layer/dev/fused_glue/validate.py`.
Whole lab: `tools/img python problems/muse_decode_layer/dev/fused_glue/screen.py`.
`admission.json` binds the sent source and local evidence; the layer table stays empty.
