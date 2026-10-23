# Prefill attention focus

The H200 prefill control is FA3 from `cand_v11` receipt `6a9cd943fe404d74`.
Its profile attention kernel achieves:

| load | FA3 us | attention TFLOPS | % of 989 TFLOPS |
|---|---:|---:|---:|
| turn1 753 slide/full | 25.8 / 414.8 | 244.8 / 597.7 | 24.8 / 60.4 |
| turn4 753 slide/full | 61.7 / 1618.0 | 409.5 / 621.5 | 41.4 / 62.8 |
| first 16K ctx0 full | 900.3 | 610.7 | 61.7 |
| first 16K ctx32K slide/full | 309.3 / 4531.4 | 444.4 / 606.6 | 44.9 / 61.3 |
| first 16K ctx64K full | 8431.4 | 586.8 | 59.3 |

TFLOPS uses 4·Q·K·16·64 (QK plus PV), with causal/window key counts;
small 753-token sliding is launch/metadata limited, while long full attention
leaves roughly 37–41% of dense BF16 peak unused.

`cand_attn_prefill.py` and the serving mirror provide FP8 Q/K and FP8-PV,
lossless page gather into contiguous storage, online FP32 softmax, exact causal
and 2048-window masks, and GQA reuse. Interface: q `[T,16,64]` FP8, paged
K/V `[pages,page,2,64]` FP8, page table, int32 `cache_seqlens` and `cu_seqlens_q`,
scales, softmax scale, window; return BF16 `[T,16,64]`.

Local sm_120 graph/reference tests pass (full/window last-row error 2.65–2.72%,
page-64 finite output). H200 timing, FA3 comparison, and serving workload gain
remain unverified. Hash: `498c8c444a014a738326bd768782fe767a8fd73bdcbf235ac58d4086d6180cca`.
