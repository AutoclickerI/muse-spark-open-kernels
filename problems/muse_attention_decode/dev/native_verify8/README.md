# Native Hopper verify8 producer

Lab experiment; FA3 stays in decode settings. H200 `a1679f9c6f6b4c70` (proj): correct,
but loses to FA3, B16 275.1 us at 33 splits vs 210.8, B32 524.3 vs 395.1; QK and P
dominate the per-tile cycles (FINDINGS "Attention").

`body.cu.inc` maps all64 query rows (eight tokens x eight GQA heads) to the
M dimension of one native FP8 WGMMA m64n64k32 tile. QK uses two native MMAs
across64 dimensions. The 64-key online softmax is local to four lanes per
query row: max and denominator require no cross-warp reduction. Its output
accumulator has two rows/thread and32 FP32 values/thread. Probabilities are
rounded to allowed E4M3 in shared memory, and V is transposed once per tile
into K-major byte layout. PV uses two more native FP8 MMAs. The128-thread
consumer group has its own128-thread barriers; the four gather warps run
independently through full/empty mbarriers and a4/8-stage cp.async ring.
The global cache stays FP8. New-token KV is quantized inside the last split;
only its owner writes the global cache, so readers have no cross-CTA store
hazard. The original parallel split combine, RMSNorm and gate are retained.

`audit.py` compiles both complete64-key interior tiles at sm90a and checks
machine code. Native WGMMA:776 instructions/64 keys,12.125/key; explicit
once-per-stage FP16:1224,19.125/key. Both counts include state load/store,
transpose, softmax and P packing; FP16 also loads its query register fixture.
This static fixture comparison chooses option A; it is not a dynamic
instruction counter or proof of speed. Native SASS has four
`QGMMA.64x64x32.F32.E4M3.E4M3`, no FP16 HMMA or FP8 unpack instructions.
Both options' native producers and the tile probes are preserved in
`sass_comparison.json`; only the selected native design runs in the new lab.

The production kernel is119 registers/thread with zero spills; clocked
kernel127, also zero spills. Each CTA is256 threads,53,248/86,016 dynamic
shared bytes at4/8 stages plus1KiB static CUDA reservation. H200's64K
register budget permits two CTAs/SM for both forms,16 resident warps in
aggregate (eight compute and eight gather). Runtime CUDA resource/occupancy
queries print in the payload.17 splits give544/1088 CTAs at B16/B32,
2.06/4.12 occupancy waves over132 SMs. At nominal80K, max4736 keys and
606,208 raw K+V bytes/CTA;8 stages carry65,536 raw KV bytes/CTA.

`layout_check.cu` checks4096 accumulator coordinates against installed
CUTLASS `CLayout_64x64` and4096 byte addresses against
`Layout_K_SW64_Atom<uint8_t>`. It catches a shared-layout or accumulator
mapping mismatch independently of the portable math path. All match.
Actual H200 WGMMA descriptor execution and mbarrier scheduling still need
hardware validation; the sm120 shim cannot establish those results.

Local admission:

-16 task tests,12 scored fresh-input graphs, five static checks including
  native sm90a builds of all CUDA strings in the local-only screen wrapper.
-15 additional captured graph/ABI fixtures: six stage4/8 short/scaled/edge/
  coherent fixtures and nine B1/16/32 exact80K BF16/1/4-partial fixtures.
  Short cases include empty splits; page sizes1/64 and scale1/64 are covered.
-Clocked output matches plain output bitwise on all six fixtures. Native
  Hopper Python dispatch is forced locally; CUDA replaces only unavailable
  WGMMA math with the equivalent mma.sync operations in the same layout.
  Global post-store FP32 errors on80K are0.019-0.032.
-No local timing of the shim is treated as native Hopper speed or as an FA3
  comparison. FA3 Python/Triton glue runs with the existing portable math
  stand-in when exercising the complete lab locally.

`pack_lab.py` rebuilds the original phase payload, retains baseline clocks
at B16/B32 and adds native vs same-input public FA3 for `(splits,stages)`
`(9,4),(17,4),(17,8),(33,4)`. Native consumer clock medians cover wait,
transpose, QK, softmax, P and PV; producer clocks cover issue/backpressure.
Plain and clocked wall times/resources are both printed. It reserves35s
for the original phase section, then passes the actual deadline to native.
Warm full payload control flow takes7.22s and3542 characters; the verify1
pair takes0.33s/997 characters, within each80s/4500-character share.

Send `b098ec3e0876b70a` ran as `a1679f9c6f6b4c70`: the combined producer payload plus
the paired verify1 sliding adapter vs exact current/best5 decode FA3 glue.

The ready adapter is `muse_decode_layer/dev/verify8_native_wgmma.py`, function
`attention_verify8_native(qkv,data,setting=(17,4)) -> [B*8,1024]`. It accepts
BF16 or contiguous FP32 projection partials, reducing/rounding inside both
producer and epilogue. It is not wired into the decode settings table.
Promote only after the same-input H200 post-store gate and FA3 win, followed
by a whole-layer composition check. Conditional rescale is closed by9828bfa0;
its denominator experiment was not isolated from the slower q8 family.

Reproduce:

```
python problems/muse_attention_decode/dev/native_verify8/screen.py
# Run the following inside tools/img:
python problems/muse_attention_decode/dev/native_verify8/check.py
python problems/muse_attention_decode/dev/native_verify8/audit.py
python tools/local_eval.py problems/muse_attention_decode/dev/native_verify8/local_screen.py
```
