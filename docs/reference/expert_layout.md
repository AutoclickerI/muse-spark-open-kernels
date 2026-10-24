# The NVFP4 expert bank for serving

Serves G1/G2. Read for bank ownership, raw scale storage and the cost of another layout.
Current dispatch and gates are in [serving_levers.md](serving_levers.md).

## Production bank

The serving loader currently keeps **one stock Marlin packed bank**. It leaves
`_muse_v4_state=None`; Marlin handles every M, including257–752. Serialized
aliases are released after Marlin repacking. No dequantized BF16/FP8 NVFP4 weight
copy is retained. Original BF16 dense weights and the permitted FP32-router
hi/lo decomposition are separate from NVFP4 experts.

The requested V9-small/Marlin-large hybrid is not bound yet. The V9-only binder
creates a different bank and omits Marlin's packed parameters. A shared reader,
or a bounded extra layout admitted against actual KV/graph residency, is needed.
A full second bank does not fit the judge workload's memory budget. Partial
copies are allowed wherever measured faster and capacity permits; none is
currently admitted.

## V9 tile layout, retained unbound

The MoE lane's `layers/muse_spark_moe.py` keeps raw E4M3 block-scale bytes and
converts them in-kernel. Its earlier custom scale encoding has been removed.
The repack changes physical order only; NVFP4 weights remain packed.

W13 pairs eight gate rows with the matching eight up rows in each16-row tile.
W2 uses natural16-row tiles. Each tile contains32768 code bytes and4096 scale bytes:

| Tensor | Shape per local layer | Bytes | MiB |
|---|---:|---:|---:|
| W13 tile bank | `[32,512,36864]` |603979776|576|
| W2 tile bank | `[32,256,36864]` |301989888|288|
| One bank | |905969664|864|

`gs13_gate`, `gs13_up` and `gs2` remain separate FP32 sidecars. The checkpoint has
unequal gate/up scales; folding them together is incorrect. Across60 layers,
these sidecars total23040 bytes/GPU. The norm sidecars are also small.

`muse_spark_serving.run_v9_dense` caps calls at256 and bypasses the standalone
module's grouped dispatch. Grouped and Humming are never selected in serving:
real-weight H200 evidence rejects Humming/V11 agreement, and grouped loses
under skew/in-layer. The independently owned kernel module is preserved.

All producers expose dense `[T,8,H]` rows with identity-or-minus-one slot maps.
Serving owns post-expert norm, weighted sum and EP reduction. Standalone dense
finalize uses CuTe below32 tokens and stock fused Triton above; an explicit
fused-AR workspace retains reduction ownership.

## Memory

On each EP8 rank,60 routed layers contain32 local experts:

```text
3 * 4096^2 * 32 * 60 = 96636764160 parameters/GPU
96636764160 * 9/16 = 54358179840 B = 54.358 GB = 50.625 GiB/GPU
```

The9/16 bytes comprise packed E2M1 plus one E4M3 byte per16 weights. Stock Marlin
and the V9 bank have essentially this same footprint; there is no multi-GB scale
saving. A second complete layout costs another54.358GB, leaving11.13GB of a
119.85GB static budget before dense weights, drafter, graph pools and KV.

The conservative full-history target KV count is15872B/token/GPU. The active
bounded-SWA allocator has a different cell and fixed live-window reserve; see
FINDINGS, “KV residency and long-context capacity,” for actual allocation math
and graph/workspace uncertainty. Do not treat42.1GB as non-KV residency: it is
spare HBM before KV under the source-count baseline.

## Verification

`test_serving_expert_storage.py` exercises the production loader at12 shapes
M1..2048, including257/511/512/513/752/753, with one packed bank. M128..753 is
bit-exact against stock math. At1024/2048, maxabs<=0.0078125 and graph/eager
variation stays within stock's measured control. Remote routes return zero.

Current full-model agreement/TTFT evidence is centralized in serving_levers.md.
Local sm120 and sm90a compilation do not establish H200 cooperative execution,
TP8/EP8 FA3 agreement, full-checkpoint residency or the hidden-workload TTFT gate.
