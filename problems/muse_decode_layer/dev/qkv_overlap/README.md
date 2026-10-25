# QKV / attention KV-group overlap ceiling

Two KV heads give two independent groups of eight Q/gate heads. For equal groups,
an ideal two-stage pipeline saves `min(Q,A)/2`. This assumes full standalone
throughput while overlapping and zero extra launches, reductions or synchronization.
QKV main+reduce names are matched to isolated qkv H200 profiles; attention includes
its combine where present. Denominators are the FOCUS sensitivity benchmark medians.

| Decode load | QKV S/F us | Attention S/F us | Ideal saved S/F us | Overall gain from this load |
|---|---:|---:|---:|---:|
| 1 | 14.50/14.10 | 9.90/18.30 | 4.95/7.05 | 0.873% |
| 16 | 14.00/13.80 | 12.50/106.50 | 6.25/6.90 | 0.522% |
| 32 | 14.40/14.50 | 17.00/195.60 | 7.20/7.25 | 0.414% |

All three decode loads together: **1.820% overall**.
Using best7_reset's own benchmark denominators instead: 1.857%.
At 50% realization, the
overall gain is 0.897%.
The bound gives no GPU-resource availability or scheduling guarantee. M1/16/32
qkv and attention compete for the same SMs and memory; launch cost is not divisible
by head count. Splitting the current Lt split-K/reduction schedule can add cost.
The built experimental path below preserves complete Q/K/V/gate groups,
per-head QK RMS and cache publication. H200 timing remains unmeasured.
Source hashes and exact inputs are in `bound.json`.

## KV-group pipeline design and built path

`runtime.py` implements the two groups for decode M=1/16/32. `build.py` emits
`best7_kv_group_cell.py`; `BEST7_KV_GROUP_CELLS` is empty until H200 admission.
The original BF16 QKV rows are losslessly gathered once into two 1152-row banks:
512 Q, 64 K, 64 V and 512 gate rows per group. Timed calls compute both GEMMs.

1. The main stream computes group 0 QKV and records a CUDA event.
2. A prepared side stream waits on that event and runs group 0 attention while
   the main stream computes group 1 QKV, then group 1 attention.
3. The main stream joins the side stream before the unchanged O projection.

The attention producer `_kg_attention` fuses Q/K RMS, optional RoPE and new
FP8 cache stores into split attention. Each CTA reads only its group's QKV rows;
only its final split writes the new row, and historical reads stop before that
row. Groups write disjoint KV heads. `_kg_combine` reduces partials, applies the
per-head output norm/gate and writes its disjoint half of the 1024-column output.
It visits every historical key and includes the new token. No host/data-derived
scheduling, global spins or replay-persistent state is used. Captured CUDA events
provide release/visibility and a replay-safe fork/join; neither attention grid can
occupy the GPU while waiting for QKV work queued behind it.

Existing PDL releases Triton successor kernels early so their independent weight
prefetches can run before `gdc_wait`. It does not publish one KV group out of a
cuBLASLt projection: the Lt call and its split-K reduction publish the whole
output. The pinned best7 native attention sources (`CUDA_SRC`,
`ATTN_PORT_CUDA_SRC`) and `PG_RAW` contain no GDC or programmatic launch attributes.
Thus this fork/join adds actual attention compute beside the second GEMM; PDL's
prefetch overlap is not counted again. Shared SM/HBM use and added launches remain
potential costs; the 1.82% bound is still optimistic.

BF16 weights/QKV and FP8 KV remain in their stored formats. Normalized query and
PV MMA fragments use FP16, matching best7's native attention fragment precision;
reductions/softmax use FP32. The first BF16-fragment prototype missed the strict
whole-layer comparison at 1.022% L2. Matching the incumbent's FP16 fragments fixes
that comparison without changing its gate: 18 forced-Hopper whole-layer draws
pass, zero route flips, maximum L2 0.9706%. Native graph replay also passes 36
independent FP32 attention/cache cases (maxL2 0.8683%, max row1.3003%), including RoPE on/off, poisoned outputs
and new-cache rows, fresh inputs, and serial-vs-overlap exact equality. Current
numbers/hashes are in `validation.json`, `layer_validation.json`, `manifest.json`.
These are local correctness results; no H200 performance claim or dispatch change.

Reproduce with `CUDA_VISIBLE_DEVICES=0 tools/img python
problems/muse_decode_layer/dev/qkv_overlap/validate.py`, then `screen_layer.py` at
the same path. The local screen disables only FA3's unsupported schedule builder;
the overlap device kernels run on the local GPU. Whole-layer screening forces the
Hopper Python path with the repository's portable attention stand-ins.
