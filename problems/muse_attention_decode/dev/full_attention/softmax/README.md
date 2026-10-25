# S14 full-verify softmax issue study

Base: S14 final **f1c77ce6**, whose selected full verify16/32 CUDA remains
**c262e4c5**. No attention variant from this lane is promoted.

One batched proj-host run is held until **2026-10-26 09:00 KST**, dispatch
**030864131ce8d08f**, PID3171016. Its generated file is
`problems/muse_bf16_proj/lab/lab_softmax_mix_lab.py`; payload
**e9423a0da334bf04**. There is no judge job ID yet. `reset_admission.json` pins
the exact wrapper, sources, screen and send-time gate. The dispatcher will collect
the receipt after the reset. All hosts reported zero remaining quota.

The earlier attention dispatch **f5b9af868f449de9** was never accepted. Its
PID3061078 was verified by command, cwd and SHA, then canceled under the dispatch
lock. It consumed no check. No further send is authorized tonight; this scheduled
batch goes to proj, as the orchestrator requested.

## What the SASS says

`audit_sass.py` compiles actual sm90a code and disassembles a128-thread softmax
probe: one M64N64 tile, P packing and arbitrary O rescaling. Counts below are
static instructions per thread, before branch frequency; the complete probe also
includes input/output addressing, stores and setup. QK/PV and the producer are
excluded. `sass.json` preserves raw opcodes and source-line categories.

| Form | MUFU.EX2 | FADD | FFMA | FP8 packing | P×ones QGMMA | Probe instructions |
|---|---:|---:|---:|---:|---:|---:|
| ref |34|38|34|16|0|344|
| mix25 |26|62|58|16|0|400|
| mix50 |18|86|82|16|0|456|
| tc25 |26|28|58|16|2|376|

Reference breakdown:34 max-stage FMNMX plus4 SHFL.BFLY; another2 FMNMX update
the running maxima. Probabilities require32 scale FFMAs,32 MUFU exponentials,
32 sum FADDs and16 F2FP packing instructions. Running-max rescaling contributes
another2 MUFU instructions. O rescale contains32 FMULs behind VOTE.ANY and a
conditional branch. The new counters measure whether that branch is taken.

The ~907-cycle H200 softmax phase is not evidence that all907 cycles are SFU
work. Replacing8/16 MUFUs adds56/112 probe instructions, including24/48 FFMAs.
Thus mixed exponentials can shift the bottleneck into issue or dependencies;
the paired run must decide. tc25 removes34 FADDs relative to mix25, adding two
native `QGMMA.64x8x32.F32.E4M3.E4M3` instructions and their synchronization.

Folding scale into quantized Q does not remove the per-element operation:
`s*scale+bias` becomes `s+bias`, one FFMA replaced by one FADD. It also moves the
query quantization boundary. This batch prioritizes the34-FADD reduction and
explicit overlap rather than claiming that scale folding is a large issue cut.

## Variants and arithmetic

- `ref`: same attention math and schedule; unchanged combine gate load moves
  after its reduction to avoid the incumbent's unrelated16/8-byte spill.
- `mix25`/`mix50`:25/50% of probability exponentials use FP32 nearest-integer
  Cody-Waite reduction and the existing degree3 minimax; the others use MUFU.
  Running-max exponentials remain native. Worst measured exp relative error
  over[-125,8] is0.000187975. Masked terms round to exact FP8 zero.
- `gap`/`mixgap`: one initial800-cycle offset for WG1, with native or25% mixed
  exponentials. These are phase-stagger diagnostics, not enforced ping-pong.
- `tc25`:25% mixed exponentials plus native FP8 P×ones row sums. Each lane
  carries one quarter of the row sum, preserving the established final
  four-lane reduction. The denominator uses quantized P, as PV does; no keys or
  cache values are skipped. The shared ones tile is512B in a1KiB allocation.
- `ping`/`ping25`: explicit alternating consumer phases, native or25% mixed.
  WG0 QK/PV overlaps WG1 softmax; after `bar.sync 3,256`, WG0 softmax overlaps
  WG1's next QK/PV. A second phase barrier completes the alternation. The
  producer and its per-warp gather/transpose ownership are unchanged. Both
  groups release the prior V slot before the cross-WG barrier, avoiding an
  ST2 ring deadlock. Each MMA phase drains before its phase boundary.

Inspiration for splitting hardware/software exponentials: the primary
[FlashAttention implementation](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/utils.py).
This is independently written CUDA/PTX using our existing polynomial coefficients.
The earlier all-polynomial recipe9407b461 remains a measured loss.

## Verification and failures retained

The exact eight variants pass32 independent attention fixtures: page64 with
KV scale1/64, B16/context4096, and B16/B32 full80K. Each checks TASK output/cache
bounds, graph replay, strided three-FP32-partial input and clock/plain bitwise
parity. Separate probes check polynomial error, packed-P layout, running max,
row sums and arbitrary O rescaling against independent PyTorch calculations.
tc25's local P×ones stand-in widens E4M3 exactly and sums in FP32; the actual
native M64N8 WGMMA is compiled and will be gated on H200 before timing.

All16 complete plain/clock CUDA units pass the unchanged no-spill/no-C75 gate.
Main kernels use159–168 registers; the32-byte frame remains explicit. Combine
uses63 registers. Native resources are in `native_plain.json`,
`native_clock.json`, `ping_native.json`, `ping_native_clock.json`.

The original50% mixed expansion triggered C7511 despite zero spills. Tying
each probability pair before writeback fixes the register lifetime without
adding waits or changing the async QK/PV schedule. The first ping-pong build
carried pending O accumulators across reconvergent phase control flow, causing
C7515/C7517. Completing each MMA phase before that boundary fixes the cause;
both groups still overlap with each other. These failures stayed local.

The first forced wrapper compared an unexecuted captured graph output; replaying
and synchronizing before comparison fixes this harness error. No bound changed.
`initial_resource_failure.json` and `first_boot_failure.json` preserve the evidence.

The six-variant attention wrapper passed normal and forced local_eval,16 TASK
tests and7 static checks each; its actual boot covered both full80K shapes.
After quota exhaustion the expanded eight-variant weighted proj wrapper passed
9 TASK tests and3 static checks, and its real prepare executed the full synthetic
lab with three seeds ×16 rotating cold replays, A:A and detailed clocks.
There is no local-timing promotion claim.

## H200 decision and integration

The scheduled lab reports paired ratios, per-seed ranges, A:A, plain/clock wall
times and per-tile max, exp/sum, packing, alpha/sum, rescale, QK/PV wait and issue
cost. It also records exact O-rescale calls/taken counts. Its microphase clocks
add overhead; compare them beside plain wall time, not as a causal sum.

`build_patch.py <variant>` generates a thin patch onto exact S14 final: only
VQ8_CUDA and the extension's content key change. Existing dispatch still selects
only full verify16/32. Generated mix25 and ping25 patches are experimental;
their identity JSONs say promotion=false and `git apply --check` passes.
Neither has a whole-layer H200 result. Select a component winner first, then
screen and pair the whole layer on identical inputs and many seeds.

Reproduce onGPU0 in the judge image:

```
python build.py
python build_ping.py
python check.py
python check.py ping ping25
python audit_sass.py
python build_lab.py --ping
```

Only after those source hashes pass the native gates and local_eval should the
payload be sent. The current scheduled source is already screened and frozen.
