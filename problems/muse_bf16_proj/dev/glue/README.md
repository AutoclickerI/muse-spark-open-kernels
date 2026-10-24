# GEMV and decode glue fusion

Read when merging the proj-host H200 lab into `cand_r3_best` (frozen base812f5f36).
Solo/GPU0, no commits, no decode submissions. Jihoon sends candidates. No judge polling:
send the lab, build the handoff, check receipts once per turn and end if only results remain.

The old CUTLASS recipe set is closed by45f58e7009c440eb: all144 choices/shape lose at
M128/256. This closes those stock schedules/tiles, not split-K/TMA/WGMMA as a family.
This experiment targets the12–25us boundary/routing/launch remainder in the decode lane's
lever table. The router uses sigmoid, biased top8 and renormalization; it is not softmax.

## Kernels and precision

`projection_glue.py` is a self-contained torch/Triton block, max added line195 characters.
No repository imports, NVFP4 expert bank, activation quantization or output cache.

- `b1`: O GEMV, post-attention RMSNorm, residual gate and pre-FFN RMSNorm in one launch.
- `pool`: one grid for pooler and router hi/lo GEMVs, pre-expert RMSNorm, biased top8,
  route weights and wrapped EP8 local ids. Router rows are tiled across CTAs; each hi/lo
  pair accumulates into the same FP32 accumulator, then the last CTA merges K partials.
- `b2`: unpooler GEMV, post-FFN input scale/RMSNorm, residual gate and next pre-attention
  RMSNorm in one launch. M<=8 computes only1024 projection columns, with zeros elsewhere.

The GEMV epilogues publish FP32 partials, fence all producer threads, and increment a
per-token-tile counter. The last arriving CTA finishes the row norm/top8 and resets
that counter to zero. There is no spin, residency assumption, separate reduce/add,
or per-call counter memset. Initial counter zeroing in prepare is synchronization
setup only; every input-dependent partial/output is overwritten in every timed call.
For multi-token tiles the norms use512-column chunks and vectorize tokens. Four exact
moments combine the post-norm and residual norm; a second chunk pass writes outputs.

O/unpooler outputs and pooler output round to BF16 before stock's next boundary.
Normalized activations remain BF16, streams/scales/accumulators FP32. All K and every
router row are processed. Layer tests retain the original attention, KV, experts and
routing checks. These component fusions assume the board's identity collectives;
serving must complete its O/expert all-reduces and unpooler gather before global norms.
No serving module is changed by this experiment.

## Selection and fair measurement

300 phase/config cases across M1/8/16/32/128/256,96 unique native specializations:
token tile1/4/8/16, output rows32/64, splitK1/2/4/8, K step64(pool)/128(boundaries).
Coarse4 pairs; top3 finalists get20 pairs. Three fresh inputs include a one-column
layout probe, outlier residual channels and changed local_start. Numeric gates use an
independent FP32 oracle with stock BF16 rounding, <=1% against the incumbent, and
residual deltas with the carry removed. Routing enforces the published gap, mandatory
local expert, local route-count and distinct-expert checks; selected scores are checked
at1% against the independent oracle. It does not require identical near-tied ids.

The lab copies the exact current best's projection/PDL/prefix tuners, scoped with
`_gbc_` names, and prepares those paths on the H200. Its dense weights are the proj
host's loaded bank; norm/gate/bias vectors are generated from model distributions
when that host does not supply them. This is explicit synthetic calibration, not
an exact decode-checkpoint claim. Original best remains untouched. The lab's control
excludes attention/MoE but uses their exact projection/glue producer/consumer chain.

Cold1GiB float32 L2 flush, graph replay, fresh3, swapped order, median20 pairs.
Full-chain controls use exactly **one shared synchronization buffer/reset**, as the
current forward does. Individual-stage results do not enable runtime dispatch.
Every M additionally measures tail-only, fused-front+tail, and fused-front-only;
**only a passing full-chain winner** enables the wrapped layer. Losing native trials
are still timed in the diagnostic. Required old-path resets are included. Optional
construction, compile, capture, numeric, timing, finalist and setup Exceptions reject
and log. No new default is enabled before H200 evidence. The shared caller deadline
stops trials; the lab targets230s after its entry, bounded by the host's285s deadline.
An in-flight JIT is not preemptible; actual cold H200 prepare remains to be measured.

## Drop-in interface

Append the block to the lane's single file and preserve its current functions:

```python
GB_H200_RECIPES = {}  # fill only from the receipt; empty keeps the original kernel
_gb_original_prepare = prepare
_gb_original_kernel = kernel_fn
kernel_fn = gb_wrap_kernel(_gb_original_kernel, project, stage_attention, experts_v9)

def prepare(weights):
    started = time.monotonic()
    state = _gb_original_prepare(weights)
    callbacks = gb_callbacks(state, project, boundary, route_and_norm, o_and_pool)
    rows = gb_prepare(state, callbacks, started + 240, recipes=GB_H200_RECIPES)
    for row in rows:
        print(row, flush=True)
    return state
```

`recipes[(M,'b1'|'pool'|'b2')]` is a list of `(BT, BN, Ksplits, BK, warps)` tuples.
`gb_prepare` returns log rows and installs `state['_gb_glue']`. Its phase trials retain
a best valid native candidate even when slower; the final combined-chain gate chooses
between the incumbent and the three combinations above. `gb_wrap_kernel` calls the
original kernel for unselected/nearby M. It keeps qkv, attention and experts unchanged.
`gb_prefix(attn,stream,state,data)` returns `(x,ids,weights,local_ids,stream1)`;
`gb_tail(moe,stream1,state)` returns `(hidden,stream2)`.

## Local evidence (sm_120; not H200 timing)

- 300 real configurations,900 fresh graph checks; all counters return to zero.
- All96 unique variants compile for sm90, including real WGMMA where the compiler
  selects it. Maximum shared storage is recorded in `glue_sm90.json`.
- Full best-layer fixture forces all18 phase paths:13/13 tests,12/12 graphs,
  11/11 static checks;5.740s cached prepare. These execute native portable Triton,
  not a math stand-in. H200 execution is still distinct.
- The lab forces Hopper-only incumbent/Lt glue locally through actual portable Lt
  execution; module paths, all projection families, graph capture and finalists run.
- Failure injection covers construction, launch, capture, oracle, coarse/final timing,
  expired budget and empty shortlist. Mock timings exercise all winning dispatch
  branches only; they are never reported as performance evidence.

Reproduce with GPU0 `tools/img python` and `test_kernels.py --full`, `compile90.py`,
`test_tuner.py`, `test_lab.py --hopper-glue`. `build.py` assembles `.py.inc` source
fragments into the standalone block and payload; fragments aren't importable modules.
`build_control.py` extracts the frozen best, preserving local argument scopes and
global declarations. `build_fixture.py` builds a never-send full-layer test.
Evidence: `runs/screen/verify_proj/glue_*`. Final proj-host screen9/9 tests+30/30 graphs+5 static checks,45.116s cached prepare.
All19 production definitions/constants exactly match the final admitted layer fixture.
The compact final lab summary is under6KB. On shared sm120 no combined chain wins;
closest is M1 tail-only81.920->83.968us (paired1.019). Larger-M losses are substantial;
these are not H200 claims. New kernels use regular launches; neighbour PDL overlap and
actual whole-layer H200 performance require a decode integration receipt.

Frozen block SHA256c763555f9908158879fb9bf65f766e03b650967487a8f97d325400613d5c92f9.
Frozen payload SHA25634143ac6134efe1f0d0c98ae95e0d3d25425fa536339b734c80f5c190a88e0ac.
Host SHA2564b4182cb1d0ff580916f7ed6fbe3a9b236264917bea014fa7a09f111168bfcf4,
sent priority0, local waiting request468733; no judge job assigned at the local check.
Do not poll or resend. H200 results are pending. Use `read_receipt.py <receipt>
--recipes <file>` to verify its payload hash and emit only phases belonging to a
positive combined-chain result. Retest those choices in the decode layer before
claiming a layer or serving gain.
