# P5 single-launch sliding attention

Structural replacement for sliding attention in frozen S11 final (`acda6c62`). One launch consumes the
actual QKV projection (BF16 or strided FP32 partials), performs Q/K RMSNorm and RoPE, appends the new KV,
gathers every causal key in the 2,048-key window, reduces split outputs through cluster DSM, and applies
output RMSNorm and the optional sigmoid gate. S11's separate gate-split choice is retained by the layer lab.
There are no metadata, global-counter reset, or combine launches.

`build.py` reuses only the vq8 low-level native FP8 primitives and projection-input loaders; `kernel.cu.inc`
contains the new algorithm. Each CTA owns one request/KV head/GQA subgroup/window split. A hardware cluster
owns all splits for that output. Historical KV prefetch precedes the PDL dependency wait. The new tokens use
an N8 QK and K32 PV tile. Cached tiles use N64 or N128 QK, LDSM.T+PRMT V transpose and register P for RS WGMMA.
The DSM reduction distributes output rows over cluster CTAs, with a final cluster barrier protecting lifetime.

`driver.py` queries actual occupancy before capture and chooses a resident cluster from shape/architecture.
It uses neither sequence values nor input data for dispatch. Batch 1 exposes four GQA subgroups per KV head;
larger batches keep all eight GQA heads together. Native variants have 124–168 registers and no spills or
C75xx warnings in the sm90a compile. Generated CUDA SHA-256: `e43cb5134924fff5e3f246759922b557259291e830804892a67697ee27381ca1`.

Verification completed locally on GPU0:

- `check.py`: all attention TASK cases and all six scored sliding shapes, N64/N128, BF16 and three strided
  FP32 splits, fresh-input graph replay, page-size/scale/boundary cases: **48 checks, zero failures**.
- `smoke.py`: the exact embedded whole-S11 payload, all six sliding shapes, two seeds, real QKV projection,
  original attention cache/output gates, CUDA graph stage mapping and CUPTI summaries. Every shape completes.
- Local timing is not H200 evidence. The first lab wiring bug (variant functions bound to the wrong globals
  dictionary) was caught locally and fixed before sending; the timeline now observes the actual attention node.

H200 measurement: `build_lab.py` writes `prof/lab/cluster_sliding_timeline.py` (`6236a0f6`). Job
**7906118231f04058** on the prefill host is pending. It fixes S11's selected projection/expert/prefetch plans,
compares A/A2/N64/N128, uses four seeds × eight rotating cold-L2 replays, and reports original graph
CUPTI completion increments. `CS_XZ85` is a lossless compressed JSON archive of paired seed times, resource
plans, output drift, gates and attention node durations. No structural kernel is promoted without this result
and subsequent whole-layer correctness/promotion checks.

Integration's already-measured decode-1 direct-partials win is separate:
[handoff](../vq8/HANDOFF.md), commit `d4d2e76`, patch `d454609d`, onto S11 `acda6c62`.
