# Sliding attention, one launch

`build.py` writes the self-contained attention-board candidate
`../../cand_sliding_singlelaunch.py` and the dense-layer adapter
`../../../muse_decode_layer/dev/sliding_attention.py`. Run the builder after edits.
The adapter accepts BF16 QKV or the layer projection tuple `(partials, splits)`, plus layer metadata, and returns
`[batch*q_len, 1024]`; it replaces the entire attention stage after QKV projection.

The CUDA kernel normalizes Q/K, applies interleaved RoPE, quantizes/stores FP8
KV, attends every key in the 2048 window, and applies per-head output RMSNorm
and sigmoid gate. Cache readers stop at `seq_lens`; new tokens are quantized
in each reader's shared memory. Only one owner writes each global new row.
Thus cross-CTA cache-store ordering is unnecessary, including q8 causal attention.

The eight MMA rows are eight query tokens at one query head for q8. For q1
at B>1 they are the eight GQA heads at one KV head, avoiding eight redundant
KV streams. B1 q1 uses one head per CTA for parallelism. Each pipeline stage
holds 256 FP8 KV rows; four warps process disjoint 64-key tiles. The CTA merges
its four warp partials, and an elected final split CTA merges the global
partials and applies norm/gate. There is one GPU launch including reduction.
Counters hold no input-derived values after return and begin at zero.
FP32 projection partials are summed and rounded to BF16 in the prologue; the
layer adapter therefore needs no materialization or cast launch.

Default split counts: B1 q1/q8 = 8; B16/B32 q1 = 4/2; B16/B32 q8 = 1.
Full q1 retains the incumbent split/cluster `attn_main`; full q8 retains FA3
on Hopper. The portable full q8 path is exercised locally, with the actual
Hopper norm/store/epilogue Python glue separately forced through a tensor
stand-in for the unavailable FA3 binary.

`check_integration.py` checks all six layer adapter shapes with fresh-input
graph replay and the task's post-store reference; it also asserts one CUDA
kernel for B1 q1/q8 and forces the full q8 Hopper glue. `tune.py` gates and
measures local split settings. The official local screen covers all 16 tests,
all 12 graphs, undefined names, signatures, and both sm90a CUDA source builds.
Only a judge receipt establishes H200 speed or promotion.

For the v9-derived portfolio, `attention_dropin.with_sliding_attention(source)`
adds the `('singlelaunch_sliding', None)` choice without promoting it. Once the
H200 receipt clears a shape, set only that shape's `attention_plan` row in the
decode lane's measured settings. Full-layer end-to-end performance remains a
separate integration check.
