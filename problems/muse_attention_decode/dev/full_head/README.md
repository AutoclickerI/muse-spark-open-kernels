# P5 single-head N128 full attention

`build.py` splits the measured full128 stage-4 pair (`9a23ca9a`) into one KV head per192-thread CTA:
128 native FP8 WGMMA consumer threads and64 gather/transpose threads. Stage4 uses110,592 dynamic shared
bytes plus4,096 static bytes;168 registers/thread. Two resident CTAs/SM are the target, replacing the
incumbent's single384-thread CTA containing two KV heads. Each head has an independent page/KV pipeline,
consumer barriers and stream-K portion. The default grid is2×SMs. Partial slots deliberately retain128
rows, with only64 live, so the established scratch/clock ABI remains exact. Combine visits one head/segment.

CUDA SHA-256 `3422fb9c7358313a98e92b75eadb1b319ee8604537907a9c36a93b8ac44640de`.
Producer ownership is exact: each of two warps owns32 keys in each64-key slab, and gathers, waits for,
then transposes only those keys. FP32 projection partials are accumulated and rounded toBF16 inside the
same prologue. No projection-sum launch is introduced. The previous producer-ownership fix and register-only
max fold are inherited from the paired winner.

Verification onGPU0:

- `check.py`:24 checks = four batch/context fixtures × BF16/three strided FP32 partials × three fresh
  input sets, each with eight graph replays. Actual single-head persistent stage2 pipeline runs, including
  batch32's64 independent head segments. Allpass. The initial query-head mapping still used2*hp+hh;
  correcting it tohp fixes the initial ~1.0 output error before any remote send.
- Native sm90a O3 compile: stage2/3/4 each168 registers,32 stack bytes,zero spill stores/loads; noC75xx.
  The unchanged combine retains16-byte spill stores/8-byte loads; that is not hidden as main-kernel spill freedom.
- Exact wrapped attention-board candidate `db96f359`:normal andforced local_eval each16 TASK tests,
  12 captured scored shapes,10 static checks. The forced import boot runs both80K full verify shapes through
  the actual stage2 head pipeline and strided partials/clocks; its N128 two-head control explicitly uses
  frozenN64 launch glue onsm120. This does not claim local execution of the Hopper-only two-head N128 ring.

`board_probe.py.inc` pairs one-head stage3/4 (grid1×/2×SM) against the measured two-head N128 stage4 winner,
three fresh80K seeds × eight rotating cold-L2 replays, including an A2 control. It reports native occupancy,
relative output errors, FP32-partial gates and phase clocks. This is native stage evidence only.
Whole-layer promotion still requires the paired original-S11 lab after quota resets.

`build_board.py` produces `out/p5_native_head_board.py` (`db96f359c6250299`):the timed attention candidate
uses one-launch full/sliding decode `35c257f8`, and the measured stage4 full-verify winner atB16/32; other
verify shapes retain the frozen attention incumbent. The single-head experiment runs only in its import lab.
Attention-board job pending. Every layer-lab host is quota-exhausted; the combined official whole-layer
candidate and16-seed global-reduction pair (`0ccc2548`,host`8e091fcb`) is queued for09:01KST October26.
