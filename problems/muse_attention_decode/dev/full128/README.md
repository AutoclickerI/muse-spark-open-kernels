# P5 native full-verify N128 tiles

Structural extension of frozen S11 vq8 (`c262e4c5`): 128 cached keys per tile instead of 64, with the same
two KV heads per persistent stream-K CTA, FP32 ex2, native FP8 QK, LDSM.T+PRMT V transpose and register P
for RS WGMMA. The virtual new-token tile remains N64. Each cached tile amortizes max/sum rescaling,
producer barriers and page/gather bookkeeping across twice as many keys. Stage 3/4 replace the incumbent's
N64 stage 6; larger N128 rings cannot fit Hopper shared memory.

`build.py` generates the source from the frozen incumbent and shared native QK helpers. `NSMAX=64`
keeps the selected layer shapes (at most 32 segments) within the shared-memory budget, including the
separate clock build. Generated CUDA SHA-256 `e728b2829fb903240f3a5b4b49a8dd8c59c885e10e1fae3e4b6d6147d2639222`.

Verification:

- The exact N128 QK/softmax/RS-PV and two-head transpose helpers run on GPU0 in `check_tile.py`:
  16 masked/unmasked × scale × fresh-graph fixtures pass against independent PyTorch attention, error <.045.
- The persistent N128 ring exceeds sm120's per-CTA shared memory even at stage 2. `smoke.py` therefore
  forces only its exact Python layer/ABI/graph/CUPTI glue with the established S11 portable native stand-in.
  Both full-verify shapes complete; this does not claim local execution of the Hopper persistent pipeline.
- Native sm90a builds compile without C75xx. Main uses 168 registers, 112 stack bytes and 8-byte spill
  stores/loads. The instrumented build has 16-byte spill stores / 24-byte loads. The unchanged combine
  retains its 16-byte spill stores / 8-byte loads. These small spills remain explicit, not hidden.
- A local tile test initially exceeded CUDA's 48 KiB static shared allocation limit; switching its test
  buffer to opted-in dynamic shared memory fixes that failure before any H200 send.

`build_lab.py` writes `prof/lab/full128_timeline.py` (`9fb09275`), which compares actual whole S11 graphs
at full verify B16/32 over eight seeds × eight rotating read-cold replays. It records CUPTI attention
completion increments, A2 noise, attention cache/reference gates after real QKV projection, and native
phase clocks for stage 3/4. The actual N128 pipeline is gated on H200 before timing.

H200 job **0171093705614698**, prefill host: pending. No integration patch before a paired win and full
candidate verification. Full-verify polynomial experiments are closed losses in [full_poly](../full_poly/README.md).
