# Grouped NVFP4 RS WGMMA

Free lane. The current experiment is `pipeline.cu` / `pipeline.py`; H200 performance is
unverified. Candidate promotion and serving wiring wait for agreement and a seed-mean win.
The scalar/plane adapter remains in decode `dev/parked/cand_r3_mixed_wgmma.py`.

## What the receipts establish

| Receipt / scope | M128 us | M256 us | Conclusion |
|---|---:|---:|---|
| Scalar296fdf45 |2060 vs V9280|2648 vs480|Correct, rejected|
| Plane01e74ee6 |1845 vs V9280|2361 vs479|Correct, only~10% better than scalar|
| Decode adaptersde6a56fc,12-seed means |masked~341, raw17647|masked448, raw17771, V9NT2545.5|Masked is the new M256 bar|
| Vector4ee35d3dd5374c40 |pending|pending|Already accepted when this turn began; could not batch|

At16K, plane W13 takes49.2ms for39491 routed pairs (~54TFLOPS), so changing the packed
representation did not repair the mainloop. Prefill controls70206a2a all pass: at512/753,
raw17 costs400/532us against Marlin595/788; at3012/16384, masked and dg_w8 are the leaders.
This lane does not replace those producers without a matched win.

W13-only ablation276739d2 uses exact R0 IDs, fresh public weights and cold128MiB graph replays:

| Scalar W13 experiment | M128 us | M256 us |
|---|---:|---:|
| Grid132 |2215.7|3066.8|
| Grid264 |1322.7|1758.4|
| Grid528 |911.4|1197.2|
| No conversion, grid264 |574.3|756.6|
| Constant weights/no loads, grid264 |322.8|462.9|
| No MMA, grid264 |1264.1|1669.6|

Conversion removal helps~2.3x; removing MMA helps only~5%. Strong grid dependence and the
remaining no-load cost make latency hiding and mainloop synchronization the next targets.
These are controlled timings; they do not isolate every remaining cycle to HBM latency.
Original SASS has no spills and eight K16 MMAs per wait0, not one wait per MMA.

## Pipeline change

`pipeline.cu` retains the vector converter: E2M1 bit permutations and raw E4M3 layout
permutations only in prepare, generic format lookup built inside each timed CTA, FP32 GS
in epilogues. Gate/up GS can differ. No prepared expanded weight or scale bank exists.

- K64/128/256 stage sizes and3–6 stage cp.async rings. Preload S-1 stages, recycle the
  previous stage after the newer group's wait1. Maximum packed Q+SF prologue is27648B/CTA,
  versus4608B in vector's original two-stage K64 ring; gathered X is async too.
- Gather source indices are computed once per task. One CTA barrier per K stage; no
  barrier inside its K64 conversion/MMA chunks. Scale/weight loads arrive with the stage.
- Two register-A banks overlap conversion and WGMMA. The last wait0 drains the task;
  steady-state overlap uses wait1. The K128 wait0 control differs only in overlap.
- Split-K2/4 partitions K without repeating its weight reads. Disjoint FP32 partials are
  reduced before SwiGLU or norm; added partial-buffer traffic is included in full timings.
- Grid264/528/1056 diagnostics and full seed comparisons. Work/offsets/hot order remain
  GPU-derived on every replay. No inter-CTA spin barrier.

The first fully unrolled K256 version triggered ptxas **C7513** and silently serialized
all16 MMAs. Input-only A-register keep points plus a non-unrolled loop over pairs of K64
chunks avoid that allocator/pipeline interaction. The final simpler MMA wrapper has no
C7513. `dump_pipeline.py` compiles every actual sm90a specialization and rejects spills,
C7513 or unexpected wait0 instructions. All ten currently have zero spills; overlap
variants contain two static wait1 sites plus the final wait0, and no per-MMA waits.
Static CTA-barrier sites are LUT initialization, stage entry and task end.
Native/portable images explicitly target sm90a/sm120. The sm120 math stand-in shares the
loads, async ring, register fragment layout, GS epilogue and split-K reduction.

| ID | N | K/stage | Stages | Split K | Overlap | Prologue packed Q+SF bytes |
|---|---:|---:|---:|---:|---|---:|
|P0|16|64|4|1|wait0|6912|
|P1|16|128|4|1|wait1|13824|
|P2|16|128|4|1|wait0 control|13824|
|P3|16|128|6|1|wait1|23040|
|P4|16|256|4|1|wait1|27648|
|P5|32|128|4|1|wait1|13824|
|P6|32|256|3|1|wait1|18432|
|P7|16|128|4|2|wait1|13824|
|P8|32|128|4|2|wait1|13824|
|P9|16|128|4|4|wait1|13824|

## Lab and validation

`prof/lab/moe_pipeline_seed.py` packs stage/grid/W13 diagnostics and full12-seed comparisons
into one proj-host run. The earlier vector run was already accepted and was left running.
Controls come from the decode adapters' raw-E4M3 V9 NT1/NT2, masked Humming and raw17, with
independent gate/up FP32 GS. R0 has exact209/414 pairs,18/21 experts,hot55/119; weights and
activations are fresh public draws, not the captured host's unexported weights.

Each variant is measured on the same three fresh inputs per seed and eight rotated/reversed
cold128MiB graph replays. Reports give per-seed values and mean/median per fixed variant;
no per-seed best is used as a winner. Gates: finite, relative L2<=1%, row<=5%. Rejected variants
are labeled and excluded from timing. W13 additionally checks every active expert/channel.
Raw17's R0 full-MoE timing and W13+SwiGLU phase use its actual swizzled input/output layout;
the phase excludes gather. W13 B/s is nominal packed Q+SF requests/unique bytes divided by
elapsed time, not hardware DRAM traffic. Split-K reduction is included in pipeline W13.

`check_pipeline.py` passes120 refreshed graph checks against an independent FP32 oracle:
all ten configurations at8/128/256 with distinct gate/up GS, plus signed/zero/subnormal
scales and empty/hot/single-pair routing. Worst relative L2~0.000635. Native Hopper math and
the341/448us mean targets remain unverified. Local timing ranks neither Hopper nor serving.

Reproduce inside tools/img: `check_pipeline.py`, `dump_pipeline.py`, `build_pipeline_lab.py`;
build the proj wrapper with tools/h200_lab.py, then screen with tools/local_eval.py. The wrapper
catches lab exceptions, so inspect LAB_ERROR/INCOMPLETE and every gate in addition to the nine
host tests. Static checks also run directly on the standalone payload because the batched
wrapper embeds payload source strings. Current hashes/screens/queue identities: `admission.json`.

The earlier canonical V6 and Humming layout inversion tests remain exact, including FP32 GS;
see `check_layouts.py`. CUTLASS host TMA repairs pass locally but its mixed GEMM stays quarantined,
out of omni and outside this experiment. No new CUTLASS slot or candidate was sent by this lane.

A winner can replace serving expert GEMMs, gathering already-normalized input and retaining
per-pair outputs and EP collectives. Separate gate/up GS are supported in the new epilogue.
Serving and complete-layer pipeline agreement wait for an H200 producer win.
