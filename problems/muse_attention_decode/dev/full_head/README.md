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
**6d47827c9ecf4a0a lands: reject the single-head verify variant.**
Stage4 grid2×SM ratios1.24010/1.31018 atB16/32 against measured N128 stage4; stage3 grid2×SM1.26484/1.38887;
stage4 grid1×SM1.40658/1.51770. A2 .99900/.99418. All native gates pass (max output drift .014),
and residency is genuinely2 CTAs/SM. Yet total per-tile cycles rise to3,341/3,217, softmax1,441/1,335,
publish653/637 and producerissue1,166/1,126. The occupancy hypothesis is falsified for this design.
The experimental attention candidate passes16/16 at1.31826x; its one-launch decode sliding times8.72/14.47/18.78us
and full decode30.61/169.78/328.48us miss their targets. This is not whole-layer promotion evidence. Every layer-lab host is quota-exhausted; the combined official whole-layer
candidate and16-seed global-reduction pair (`0ccc2548`,host`8e091fcb`) is queued for09:01KST October26.


## Decode1 full specialization: closed loss

`build_q1.py` / `q1_driver.py` specialize the actual persistent pipeline to one full-decode token:
eight real query rows, one cooperative norm/partial-input pass, no padded global stores, and8-row partial
slots instead of128. CUDA`607dc8a1`; native stage2/3/4 each154 registers,zero stack/spills; combine64regs,
zero stack/spills. `check_q1.py`:24 independent cases ×8 graph replays, including80K context and strided
FP32 input. Wrapped native probe`80c3d0b0` normal/forced16 TASK/12 graph/10 static eachpass.

H200 **58d6a59603564014**, eight seeds ×8 paired native replays: original22.760us, A2.99029;
global last-CTA34.244us/1.50458; specialized head stage3 32.406us/1.42396;
stage4 31.586us/**1.38787**, SE.00473. Both native andpartial-input gates pass (max relative error.00592
against FP8 control). Stage4 native main reaches18.72us fromfirstCTAentry, preparation4.128us,
4–5tiles/CTA,4,715cycles/tile; softmax906.5,publish695.25,producerissue1,714.4.
This falsifies a prologue-only explanation of decode1 full latency. Retain the earlier direct-partials
win for decode1 full; neither new global nor native-head full path is promoted. The experimental attention
candidate passes16/16 at1.33494x, which is not whole-layer evidence.
