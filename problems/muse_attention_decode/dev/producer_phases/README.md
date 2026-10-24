# Hopper producer diagnosis and paired attention lab

The be19da94 producer passes but loses at B16/B32 full: 427.8/713.7 us,
with only 6.3/8.5 us in combine. The mechanism below is established by
machine code; phase fractions remain unmeasured on H200.

`audit.py` compiles the exact archived producer for sm90 and sm90a, checks
that `verify8.cu` equals the archived CUDA string, and reads cuobjdump SASS
and resources. `sass_evidence.json`, `baseline_mma_excerpt.sass` and the
two small probe SASS files preserve the result with CUDA13.0.88:

- One `mma.sync.m16n8k32...e4m3` lowers to **12 FP8-to-FP16 unpack
  conversions and two FP16 HMMAs on sm90a**. The identical PTX becomes
  one native FP8 QMMA on sm120. The earlier FP8-peak model does not
  describe this Hopper kernel.
- The producer uses223 registers/thread, no stack/spill bytes and no
  SASS LDL/STL. Static counts across all branches/prologue/tail include
  408 unpack conversions and160 FP16 HMMAs; these are not dynamic counts.
- 128 threads/CTA and rounded224 registers/thread permit two CTAs/SM:
  eight resident warps,12.5% of Hopper's64-warp limit. Shared memory is
  64KiB dynamic +1KiB CUDA reservation and would allow three CTAs.
  [NVIDIA's Hopper resource limits](https://docs.nvidia.com/cuda/hopper-tuning-guide/)
  support the model; the payload also prints CUDA's actual occupancy query.

At nominal80K, each17-way CTA covers at most4736 keys:606208 K+V bytes
for one KV head. B16/B32 grids are544/1088 CTAs, or2.06/4.12 occupancy
waves over132 SMs (three/five launch waves). Agentic max86K increases the
largest CTA to5120 keys/655360 bytes. Nominal producer logical bandwidth
is0.77/0.92TB/s; no DRAM counter establishes HBM saturation. Detailed
accounting is in `resources.json`.

The lab adds per-warp clock64 totals for gather issue/page-index prefetch,
gather wait/handoff, transpose, QK, softmax/rescale, probability conversion,
and PV/layout. It also records prologue/CTA elapsed cycles. Phase medians
and p90 are cycles per stage; CTA cycles are unnormalized. Clocks change
instruction scheduling: plain and instrumented wall times/registers/
occupancy print side by side. All local instrumented outputs match their
plain controls bitwise. Clock intervals are diagnostics, not stall counters.

`widen.py` builds the next structural variant: keep the global cache FP8,
widen K/V exactly once per stage for all four query warps, preload widened
query fragments, and pack FP16 probabilities once for reuse in all PV
MMAs. Queries still round to permitted E4M3. The query scratch becomes
widened-K scratch after its register loads and the stage barrier. This
reduces sm90a registers to141/133 for128/64 keys, with84/44KiB dynamic
shared memory at three stages. The64-key choice permits three CTAs/SM.
No dequantized weight or cache copy persists outside the timed kernel.

The explicit sm90a-lowering stand-in on sm120 is in `force_lowering.py`.
The widened64-key variant is about tied at B16 and4% slower at B32 locally;
128 keys loses. Native sm120 also loses. These results **do not establish
an H200 or FA3 win**; no further attention candidate was sent. Native H200
clocks and same-input FA3 controls decide whether to tune this variant or
move to native Hopper WGMMA. The rescale/deferred switches are not promoted
while9828bfa041a143a9 remains unresolved.

The second payload compares `verify1_sliding_pick` against the exact decode
public FA3 call, including current and best5 QK glue, cache store and norm/
gate. It covers BF16 and1/4 FP32 partials, fresh graph values and the task's
post-store FP32 gate. It excludes the projection GEMV and whole-layer PDL
schedule, so a standalone saving still needs a layer composition check.

Local evidence in `admission.json`:30 producer fixtures (24 fresh graphs
plus6 exact80K cold graphs), corresponding instrumented gates, nine paired
ABI graphs; all pass. The FP32 gate functions are frozen from the task,
and cached lab gates are checked against the unchanged local task gate.
Native FA3 cannot execute on sm120: its Python/Triton glue is forced through
a portable operation; its local timing is not used as an FA3 measurement.
Both payloads pass undefined-name/signature checks and native sm90a builds.
Warm full control-flow exercises take4.8s/0.22s and print3601/997 characters,
inside each80s/4500-character share. Cold H200 compilation remains unverified.

Scheduled batch: send key `e0712c33ab1cb715`, proj host, priority0.8,
`--not-before 2026-10-25T09:00:20+09:00`. The detached log confirms the
hold and no early judge request. There is no judge job ID until dispatch.
Read the `verify8_producer_phases` and `verify1_sliding_fa3_pair` sections
once a receipt exists; no attention setting changes before a measured win.

Reproduce in the image:

```
python problems/muse_attention_decode/dev/producer_phases/build.py
python problems/muse_attention_decode/dev/producer_phases/audit.py
python problems/muse_attention_decode/dev/producer_phases/check.py
python problems/muse_attention_decode/dev/producer_phases/check_pair.py
python problems/muse_attention_decode/dev/producer_phases/check_lowered.py
python problems/muse_attention_decode/dev/producer_phases/exercise_lab.py
```
