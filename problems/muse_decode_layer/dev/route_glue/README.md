# Route-only and verify-load glue cells

The target is the decode-layer overall score. The previous small-M full-glue
lab is separate and immutable (`80b70f7f20fa41c6`). This experiment isolates
route removal while retaining incumbent projections, and extends epilogue
fusion to the verify loads with material glue time.

`profile_glue.json` extracts the best7_reset `c0e74bed` receipt: combined
boundary/route GPU activity is11.0/11.7us at M8 sliding/full,14.1/13.8 at M128,
and22.9/22.9 at M256. These include PDL waits; only paired whole-layer time
can establish saved critical-path time.

## Kernels

- `route`: NT1 builds biased top8-of256 and weights in each CTA's private
  shared-memory summary. Pool/router partials and BF16 hi+lo rows are exactly
  the incumbent producer's; pre-expert RMSNorm is fused into expert BF16 input
  loads. CTA0 exports global IDs/weights/localIDs. Every CTA computes the same
  deterministic summary; there is no routing grid barrier. At M1 the whole
  CTA computes the RMS rather than leaving seven warps idle. The packed
  NVFP4/E4M3 recoded expert body is unchanged. At M256 the incumbent remains
  masked Humming: routing is folded into its pooled-input and deterministic
  token-position preparation; normalization remains in its gather prologue.
- `pool_epilogue`: BF16 pool/router GEMV plus last-CTA route/pre-expert norm;
  both boundaries and the expert backend stay incumbent. This is the second
  route-fusion approach, measured independently of a boundary replacement.
- `tail`: unpooler plus last-CTA boundary2 at M8/128/256. Its counter resets
  each call; no waiting for unscheduled CTAs. M>=128 uses16-token tiles.
- `both`: route plus tail, so their whole-layer interaction is measured.

The drop-in's table is empty. No change to best7_scale itself is authorized by
local timing; the H200 receipt decides which cells to hand to integration.

## Lab and validation

Non-expert weights use seed941732/model scales and mean-aligned router; first32
real host experts/scales/post-expert norm are retained. The packed host bank is
checked against the decode packer before reuse. M256 Humming transforms only
those32 experts and keeps the FP32 global scales. No expert bank is dequantized
in prepare. Both arms receive identical stream, attention input, EP owner and
KV addresses. Controls use current best7_scale glue, c0e74bed fixed projection
choices and q8 public layer attention, with the same backend in every arm.

Twelve cases: decode1/16/32 and verify1/16/32, sliding/full, context80000/page1,
soft/agentic. Three fresh graph input streams and EP owners,6 AB/BA pairs with
a cold256MiB flush before every replay, A:A per case. Every comparison requires
exact global route IDs before and after timing and checks the independent FP32
route score/count gates. Whole hidden and residual-update L2 gates are1%/5%row.

`check_route.py` directly executes the native prologue on72 changed-input graphs
(M1/8/16/32/128/256 x splits1/2/8/16 x3 draws); global/local IDs match exactly,
max numeric drift2.29e-5. `screen.py` forces Hopper Python attention paths with
portable attention launches and executes the real new CUDA/Triton glue locally.
The screen fails if the payload fails even when the unchanged host passes.
Build with `build_moe.py` then `build.py`; construct the MoE wrapper with h200_lab.
The kernel's raw-pointer ABI avoids Torch headers in its CUDA translation unit.
Humming weight transformation must run on the main thread.

## Sent and checked

MoE-host send **81ee15ab3a2d29e0**, accepted job **8a3d71811a324c36**.
Immediate detached log reports acceptance without rejection. Receipt is pending.
Payload SHA `ded132cf8f7bbdf0af85e52dd4f66ff5272e39ac40e46c4adef841dda85c208e`.
Disabled cell `best7_route_glue_cell.py`, SHA
`c0e845637a2cfc658752dbd8c12fa186d74928beb70a8e696eb4c4af94a03248`.

The final whole-layer screen passes108 comparisons across all12 cases with
zero route flips, max hidden/residual-update L2 0.002728. Native route72 and
Humming pooled/route/position54 fresh graph replays pass; host5 tests/6 static
checks pass. Explicit SM90a builds cover8 projection kernels and the route-MoE
kernel, all without register spills (route-MoE128 registers versus original124).
The payload also passes name/signature checks. Local cells all lose, so the
best7 integration table remains empty. Cached payload12.9s; cold H200 setup and
whole-layer timing remain unverified. `admission.json` binds every sent hash.

Local setup failures were fixed before sending: slice the host's256 experts to
this layer's32; run Humming transformation on the main thread; subtract static
shared memory from the opt-in limit; use a value-returning per-thread fence plus
`tl.debug_barrier` rather than Triton's unsupported void inline assembly.
