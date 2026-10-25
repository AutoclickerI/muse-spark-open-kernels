# P5 native full-verify N128 tiles

Structural extension of frozen S11 vq8 (`c262e4c5`): 128 cached keys per tile instead of 64, with the same
two KV heads per persistent stream-K CTA, FP32 ex2, native FP8 QK, LDSM.T+PRMT V transpose and register P
for RS WGMMA. The virtual new-token tile remains N64. Each cached tile amortizes max/sum rescaling,
producer barriers and page/gather bookkeeping across twice as many keys. Stage 3/4 replace the incumbent's
N64 stage 6; larger N128 rings cannot fit Hopper shared memory.

`build.py` generates the source from the frozen incumbent and shared native QK helpers. `NSMAX=64`
keeps the selected layer shapes (at most 32 segments) within the shared-memory budget, including the
separate clock build. The measured original CUDA was `e728b282`; the current corrected source is listed below.

Verification:

- The exact N128 QK/softmax/RS-PV and two-head transpose helpers run on GPU0 in `check_tile.py`:
  16 masked/unmasked × scale × fresh-graph fixtures pass against independent PyTorch attention, error <.045.
- The persistent N128 ring exceeds sm120's per-CTA shared memory even at stage 2. `smoke.py` therefore
  forces only its exact Python layer/ABI/graph/CUPTI glue with the established S11 portable native stand-in.
  Both full-verify shapes complete; this does not claim local execution of the Hopper persistent pipeline.
- Native sm90a builds compile without C75xx. Original main uses 168 registers, 112 stack bytes and 8-byte spill
  stores/loads. The instrumented build has 16-byte spill stores / 24-byte loads. The unchanged combine
  retains its 16-byte spill stores / 8-byte loads. These small spills remain explicit, not hidden.
- A local tile test initially exceeded CUDA's 48 KiB static shared allocation limit; switching its test
  buffer to opted-in dynamic shared memory fixes that failure before any H200 send.

`build_lab.py` writes `prof/lab/full128_timeline.py` (`9fb09275`), which compares actual whole S11 graphs
at full verify B16/32 over eight seeds × eight rotating read-cold replays. It records CUPTI attention
completion increments, A2 noise, attention cache/reference gates after real QKV projection, and native
phase clocks for stage 3/4. The actual N128 pipeline is gated on H200 before timing.

H200 **0171093705614698**, prefill, original payload `9fb09275`: stage-3 attention gate returns NaNs,
so **no valid timing**. The lab stops before stage 4; the unchanged host itself passes its gate.

Correctness diagnostic **e0c7d58b8cb54271**, MoE host, payload `734bf8aa`:

- Native N128 tile oracle passes (relative errors .02237/.02198), isolating the failure from QK/softmax/PV math.
- Both stage 3/4 pass context 20/127/128/257. At context 80K, stage 3 produces NaNs; forcing
  WGMMA waits to complete still fails. Stage 4 passes the sampled cases but does not establish race freedom.
- Root cause found in the producer ownership: gather/page-fetch assigned each warp 32 contiguous keys,
  while transpose assigned each warp 16 keys in each 64-key slab. Transpose therefore consumed another
  warp's async loads, and cp.async.wait_group covers only the issuing warp. More buffering hid the race.

The current build fixes **page fetch and K/V gather** to keys `16*warp + local16 + 64*slab`, matching
transpose exactly, with no extra CTA barrier. A per-warp set check proves page/gather/transpose ownership
identical and complete. Ownership-fixed CUDA SHA-256: `3dcd60016b7a5adae96b32168434c5ada185027c7b144dadc48d621d2e76e7f2`;
16 portable tile checks pass again, the sm90a build completes without C75xx, and the forced layer/graph
stand-in completes both shapes. No claim of a fixed native pipeline until the new H200 gate lands.

Ownership-fixed pipeline landed in **7e93cc81af17489a** (decode host), 16 seeds × eight cold replays:
all native attention gates pass and all fresh replay outputs are finite. The warp-ownership fix is confirmed.
The pipeline is nevertheless a large performance loss:

| Full verify | S11 increment us | Stage3 ratio / increment us | Stage4 ratio / increment us | A2 |
|---|---:|---:|---:|---:|
| B16 | 152.87 | 1.26726 / 258.62 | 1.30340 / 269.82 | 1.00003 |
| B32 | 265.25 | 1.40674 / 494.66 | 1.45280 / 521.22 | 1.00118 |

Native phase clocks show 3,071–3,175 softmax cycles per N128 tile and total 4,563–4,769 cycles.
This is not promoted. The unchanged official S11 host passes 13/13 at 1.71559x; that score is not a variant result.

A second concrete root cause: the generic max-reduction loop kept the 16-element softmax array in local
memory instead of registers. `build.py` now folds compile-time 8/4 elements and uses literal remaining max
operations. Native stage4 SASS LDL/STL counts fall **60/87→4/3**; stack **112→40 bytes**, spill stores/loads
**8→4 bytes**, registers remain 168. Corrected CUDA SHA-256
`9a23ca9a98ddeaff252bad309d0f1c774b0aa0f72144c6fe4f74a39dddf45507`.
All 16 exact tile oracle fixtures pass again. This is a machine-code result, not yet an H200 speed claim.

`build_winner_lab.py` batches official validation of last-CTA B1 sliding (`fead4a00`) with the register-fold
full128 pair, reusing actual prepared host weights/state (no redundant S11 tuner). It writes
`prof/lab/full128_regmax_pair.py` and `problems/muse_decode_layer/lab/lab_p5_last_and_regmax.py` (`2d48b941`).
Proj/prefill/MoE quotas are exhausted; this check uses a remaining decode check to finish the promotion gate
and measure the identified in-stage fix. Normal/forced local screens precede send.

Full-verify polynomial experiments are closed losses in [full_poly](../full_poly/README.md).
