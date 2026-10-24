# The NVFP4 expert bank for serving

Serves G1/G2. Read for bank ownership, raw scale storage and the cost of another layout.
Current dispatch and gates are in [serving_levers.md](serving_levers.md).

## Production bank

The serving loader currently keeps **one stock Marlin packed bank**. It leaves
`_muse_v4_state=None`; Marlin handles every M, including257–752. Serialized
aliases are released after Marlin repacking. No dequantized BF16/FP8 NVFP4 weight
copy is retained. Original BF16 dense weights and the permitted FP32-router
hi/lo decomposition are separate from NVFP4 experts.

The V9-small/Marlin-large hybrid now has a shared-bank reader, awaiting H200
performance admission. `muse_spark_expert_bank.py` joins the stock Marlin code
and scale arrays into one allocation per W13/W2; the original contiguous
Marlin tensors remain byte-identical views. `muse_marlin_v9_reader.py` reads
those same allocations directly and inverts Marlin's scale-byte transform
inside the kernel. It changes V9's weight reader, preserving its routing,
expert arithmetic and dense-output ABI. No second expert layout, timed repack
or prepared BF16/FP8 weight copy is used. `MARLIN_READER_ADMITTED=False` keeps
stock as default until the new reader's own timing wins; old canonical-bank
V9 receipts cannot admit a different memory-access pattern.

When admitted, the policy is V9 at M<=256 and Marlin above256, explicitly
including257–511. A full second bank does not fit the judge workload's budget.
Partial copies are allowed where faster and capacity permits; none is needed
by this reader. A serialized per-call one-layer repack was correct but rejected:
at c32 its modeled traffic floor377us/layer exceeds the114us break-even.

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

The standalone canonical-bank kernel supports separate FP32 `gs13_gate`,
`gs13_up` and `gs2`. The serving reader instead follows stock Marlin's actual
convention: its already-BF16-rounded gate global is shared between both W13
halves. The sidecars invert stock's exact power-of-two bias; the packed bank
and stored scales are unchanged. Comparing against the original independent
FP32 globals initially caused a0.021484375 mismatch. Matching stock's convention
restores native-V9/direct-reader bit equality. The norm sidecars are small.

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
saving. The joined Marlin bank remains905969664B/layer,54358179840B for60
layers, with zero additional resident expert bytes or timed repack scratch.
Joining temporarily allocates a W13/W2 array while stock's loading aliases
remain; those aliases are released before serving. A second complete layout costs another54.358GB, leaving11.13GB of a
119.85GB static budget before dense weights, drafter, graph pools and KV.

The conservative full-history target KV count is15872B/token/GPU. The active
bounded-SWA allocator has a different cell and fixed live-window reserve; see
FINDINGS, “KV residency and long-context capacity,” for actual allocation math
and graph/workspace uncertainty. Do not treat42.1GB as non-KV residency: it is
spare HBM before KV under the source-count baseline.

## Verification

`test_shared_expert_bank.py` forces the production joined-bank loader and real
reader at13 shapes M1..2048, including257/511/512/513/752/753. M<=256 has
relative L2<=0.00538/maxabs0.03125 against stock; the Marlin path through753 is
bit-exact. At1024/2048, maxabs<=0.0078125 and graph/eager variation stays within
stock's control. Shifted EP ownership and remote-zero cases pass.
`test_marlin_v9_reader.py` additionally checks all127 positive finite E4M3
encodings (including zero/subnormals), exact stored bytes, native-V9 bit parity,
and eager/three changed-input graph replays at M1/8/32/128/256. The derived CUDA
source compiles for sm90a. Full-width TP1/TP2 server gates are recorded in
serving_levers.md. H200 admission lab source is `6a3168ef5d749e10`.

Current full-model agreement/TTFT evidence is centralized in serving_levers.md.
Local sm120 and sm90a compilation do not establish H200 cooperative execution,
TP8/EP8 FA3 agreement, full-checkpoint residency or the hidden-workload TTFT gate.
