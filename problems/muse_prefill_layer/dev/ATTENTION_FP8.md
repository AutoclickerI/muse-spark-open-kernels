# FP8 prefill attention candidates

The new path is enabled only for full attention with at least 2,048 queries per request.
Other Hopper shapes keep v3 FA3. QK and PV both use E4M3 tensor-core inputs; softmax
maxima, denominator, accumulation and cache descales remain FP32. Probabilities are
scaled by 448 before E4M3 conversion, and the output divides out that scale. Every
causal key is processed. The gathered variant copies FP8 KV losslessly inside the
timed call and retains no data-derived state across calls.

Candidates:

- `cand_attention_fp8.py`: direct paged reader, BQ16/BN128/GROUP8, 8 warps, 2 stages;
  measured v3 experts.
- `cand_attention_fp8_contig.py`: the same tile after lossless contiguous FP8 gather;
  measured v3 experts. Main attention choice pending H200 measurement.
- `cand_attention_fp8_contig256.py`: gathered BN256 diagnostic alternate. Local sm120
  winner, but the Hopper compiler spills; lower queue priority.
- `cand_canonical_attention.py`: canonical V6 pipelined reader plus gathered BN128 attention.
- `cand_grouped_attention.py`: canonical V6 grouped RS128 reader plus gathered BN128 attention.
- `cand_prefill_autotune.py`: both canonical expert readers, plus synthetic prepare-time
  selection among FA3/direct FP8 BN128/gathered BN128/gathered BN256 at 32K/64K prefix.
  The short-prefix bucket retains FA3. Only shape/config choices survive prepare().

## Local timing only

`attention_fp8_bench.json`: RTX PRO 6000 Blackwell Max-Q, sm120, complete attention
CUDA graphs, five replays with a 128 MiB cold-L2 flush; includes KV gather.
These are not H200 or FA3 speedups.

| 16K query prefix | Existing BF16 Triton | FP8 direct BN128 | FP8 gathered BN128 | FP8 gathered BN256 |
|---|---:|---:|---:|---:|
| 32,768 | 20.257 ms | 13.357 ms | 12.890 ms | 11.988 ms |
| 65,536 | 38.839 ms | 25.066 ms | 23.721 ms | 23.411 ms |

A second soft-key sweep (`attention_fp8_bench_tiles.json`) rejected smaller M and
16-warps layouts: about twice the local latency. Reducing stages to one did not
remove BN256 Hopper spills. Clock/shared-device variability limits cross-run comparisons.

## Hopper compile evidence

`attention_fp8_compile.log` and `attention_fp8_compile_tiles.json` compile the actual
Triton kernels for sm90, without running them. BN128 emits six FP8 WGMMA instructions
per key-loop iteration; BN256 emits ten. No FP8 mma.sync instructions.

- Main gathered BN128: 24 KiB dynamic shared memory, 255 registers, no stack spills.
- Gathered BN256: 40 KiB dynamic shared memory, 255 registers, 424 bytes stack/thread.
- Smaller BQ8/BN128: 200 registers, no stack, but roughly doubles local latency.
- 16-warps BN128: 128 registers and 128 bytes stack/thread, also doubles local latency.

Compiler resource evidence makes BN128 the safer first H200 experiment; only a judge
receipt can decide the H200 winner.

## Validation

The forced direct and gathered BN128 paths pass all five official local test shapes,
plus changed-input graph replays with page64/non-unit cache scales and unequal
request lengths. The final exact sources are screened with `tools/local_eval.py`
by `attention_fp8_screen.py`; final logs and JSON use `_final` suffixes.

`attention_fp8_verify_long.json` captures complete 16K query blocks at 32K and 64K
prefix. Two fresh replays replace query, page table, K/V and lengths; one uses random
keys and one uses stock-like soft shared-direction keys. All output entries are finite.
For each replay, 17 sampled query positions in every head are checked against FP32
attention over every visible key. Relative L2 is 0.86–1.19%. This is sampled numerical
validation, not an exhaustive FP32 comparison of all 16K output rows.

`attention_fp8_verify_hopper.py` forces the real Hopper Python dispatch on sm120 and
uses the portable Triton kernel only as the FA3 stand-in. Both page64/non-unit-scale
full and unequal-length sliding cases pass fresh full-layer graph replays. Actual
new FP8 attention instructions execute on sm120, and their sm90 WGMMA form compiles.
`attention_fp8_verify_combo.py` exercises the combined canonical expert + new attention
production branch at full L2048/page64/non-unit scales with two fresh graph replays.

## SGLang port

Place the kernels in a source module under `python/sglang/kernels/ops/attention/` and
call them from `FlashAttentionBackend.forward_extend` in
`python/sglang/srt/layers/attention/flashattention_backend.py`, replacing its full-layer
long-query FA3 `flash_attn_with_kvcache` call after cache writes. Keep the stock KV pool
and FP8 descales. No expert-bank change is needed for attention; its gathered FP8 KV
buffers are per-invocation scratch (20 MiB combined at 80K context, 2 KV heads, D64).

The board adapter assumes equal L per request and local HQ16/HKV2. Serving must supply
request query offsets from `cu_seqlens_q`, per-request query lengths and context lengths,
and constexpr head counts/strides from `layer.tp_q_head_num` / `tp_k_head_num`, so mixed
batches and TP1/TP2 exercise the same algorithm. Return BF16 attention outputs to the
existing per-head RMSNorm/gating path. Preserve SGLang's Q scaling convention exactly:
the board folds 5.4730064863875 into softmax_scale after the normalized Q is rounded
E4M3. The FP8 K and V descales belong after QK and PV respectively. Port selection
awaits H200 timings and serving agreement; no serving files were edited in this round.


## Complete prepare-time selector

`cand_prefill_autotune.py` includes the unchanged expert autotuner from canonical base
SHA256 `9b21cf17386d46de2da7f606c2c024bb26461935dd4bf37a0ec253b0b0e10a14`.
It also tunes four complete attention implementations on synthetic E4M3 queries and
soft shared-direction FP8 keys at L16384/prefix32768 and65536. Each choice is a complete
CUDA graph, timed three times after128MiB L2 flushes. Outputs are compared to the FA3
choice with a4% relative-L2 math guard. Both the expert and attention tuners print all
measurements and decisions. State retains only selected reader/config names; synthetic
queries, KV, routes, outputs and CUDA graphs are released.

For serving, tune once per GPU during initialization/warmup and share the resulting
shape/config choices across layers. Repeating the same attention tuning on all60 layers
would add startup cost without measuring a different attention algorithm. The expert
reader guard still needs validation against the loaded layer's represented weights.


Frozen complete-selector SHA256: `71593f9c5e5467f616ea26d3f8dc7778e3f277106216b62df4dc193797680d9a`. Exact-source official five
local tests pass. Forced-native and forced-SM90-glue synthetic tuning also pass; the
warm-cache local stand-in prepare takes 9.46s. Selected and all four attention
paths pass two fresh full-layer production graphs at L2048/prefix32K/page64/scale1/64.
See `prefill_autotune_validation.json` and `verify_prefill_autotune.log`. H200 execution,
H200 timing and serving agreement remain unverified.
