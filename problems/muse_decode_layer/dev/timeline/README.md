S11 final H200 graph timeline

This lane measures the original `cand_r3_best7_s11_final.py` (`acda6c62`) at
all twelve scored shapes: decode/verify × batch 1/16/32 × sliding/full.
It changes no candidate kernels or dispatch choices. `selected_prepare.py`
constructs the same selected S11 state and retains its selected expert,
attention and projection gates; it omits compilation/tuning of unselected
contenders so both the host and the lab fit prepare’s 300-second cap. The prefill host supplies
its real layer weights; inputs use public generation, seed 172903, 80K context,
soft keys, page size 1, fixed lengths at batch 1 and agentic lengths otherwise.
These are one fixture per shape, not held-out decode-host routing or a multi-seed
promotion experiment.

Reproduce:

```bash
python3 problems/muse_decode_layer/dev/timeline/build.py
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_decode_layer/dev/timeline/screen.py
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_decode_layer/dev/timeline/screen.py --forced
python3 tools/h200_lab.py send problems/muse_decode_layer/dev/timeline/lab_timeline.py --host muse_prefill_layer --note "S11 graph timeline, all 12 shapes, CUPTI + cold READ + observer controls"
python3 tools/h200_lab.py read JOB
python3 problems/muse_decode_layer/dev/timeline/report.py JOB
```

`build.py` freezes the source and `activity.cpp` into a self-contained payload.
The generated payload is ignored; building verifies the frozen candidate hash.
`screen.py` runs the payload inside `tools/local_eval.py` and then all 13 official
local correctness cases. The forced screen uses the existing portable native
attention stand-ins from `dev/verify_best.py`. The CUDA 13 graph query and CUPTI
recorder run unchanged on sm_120. Local timings are not H200 evidence.

Measurement and limits

- CUPTI concurrent-kernel activities provide original CUDA graph node IDs,
  streams, start/end timestamps and launch geometry. Host-side graph snapshots
  at S11's stage boundaries add no GPU nodes. CUPTI executable IDs differ from
  capture IDs; the lab performs the same upper-32-bit graph-ID rekeying as the
  image's `torch.cuda._graph_annotations._rekey_annotations`.
- Every replay, including warmups and controls, follows `lab_cold_l2()` (the
  wrapper's 1 GiB **read**). Five profiled replays and five unprofiled replays
  check CUPTI observer effects. The trace with median first-to-last node span is
  kept in full; node-duration MADs and all five wall spans expose variation.
- A second graph omits the host graph queries. Five alternating paired replays
  compare it against the mapped graph on identical tensors. Exact route equality
  and finite outputs are required; floating differences and S11's own repeat
  drift are recorded. Atomic reduction order can change floating outputs across
  captures. The official reference correctness gate remains in local_eval.
- Wall is first CUPTI node start to last node end. Event wall includes replay
  framing; both raw and empty-graph-corrected event medians are reported separately. The union of node intervals plus the
  no-node gaps equals wall; the sum of durations includes overlaps. CUPTI node
  boundaries are the permitted activity alternative to CTA globaltimer stamps;
  they include dispatch/drain and do not pinpoint individual CTA entry/exit.
- Byte counts are **static compulsory unique logical reads**, calculated from
  actual weight shapes, generated lengths and routed expert IDs. They are not
  DRAM transactions. Dense weights retain BF16; expert weights retain FP4 codes
  and byte scales. Multi-node MoE partitions use actual expert token counts.
  Per-node internal partial reads with incomplete layout information are lower
  bounds or explicitly unresolved. Write traffic, redundant reads, L2 reuse,
  compression and HBM transactions are not measured.
- “Below 50%” is an **average-rate proxy**: integrate intervals where the sum
  of active nodes' logical bytes / node duration is below 2.29 TB/s. Neither
  kernel activity timestamps nor logical byte counts identify instantaneous
  HBM utilization. The lab does not claim a measured DRAM-duty trace.
- Positive empty intervals are measured no-kernel gaps. PDL waits can happen
  *inside overlapping node intervals*; CUPTI activity alone cannot split them
  from useful instructions or distinguish launch/dependency causes. Signed
  QKV→attention and pool/router→routing envelopes are reported, not relabeled
  as actual wait durations.
- The continuous-stream model is total compulsory reads / 4.58 TB/s plus
  `L(qkv→attention) + L(router→routing)`. The byte term is evaluated. Unavoidable
  dependent latency requires a data-ready/first-consumption experiment; an
  observed graph gap is not that minimum. Thus the byte-only result is an
  optimistic model, not a proven attainable ceiling. Stage residuals identify
  headroom envelopes, not causal measurements of time with no HBM traffic.

Receipt accounting

[jobs.json](jobs.json) keeps every H200 attempt. First attempt
`2cc7b07dd29c4360` (`3862a1aa`) produced **zero timeline shapes**: host prepare
136.4 s + full S11 prepare 144.8 s = 281.2 s, beyond the lab's deadline margin.
Its 1.4313× host score concerns the unchanged prefill candidate. This is an
unmeasured timeline, not a negative kernel result. The selected-only setup
removes the unselected work that exhausted that budget.

Results

H200 receipt `c6bd22e2084c4809`; [raw receipt](receipts/c6bd22e2084c4809.json).

Single public fixture per shape, using prefill-host weights. Decode 1 has **zero local expert routes** on this fixture; its wall time is not representative of the board’s calibrated routing. Each load is 46 sliding + 16 full layers. CUPTI wall runs from the first node start to the last node end; kernel sum double counts overlapping nodes, busy union does not. No-kernel time is the complement of the union.

| Load | Wall ms | Kernel sum ms | Busy union ms | No-kernel ms | <50% proxy interval ms | Read GB | Stream floor ms |
|---|---|---|---|---|---|---|---|
| decode 1 | 4.110 | 7.400 | 4.061 | 0.050 | 1.456–1.456 | 8.976 | 1.960 |
| decode 16 | 9.061 | 14.795 | 8.996 | 0.065 | 2.499–2.499 | 25.540 | 5.576 |
| decode 32 | 12.676 | 16.902 | 12.618 | 0.058 | 1.005–1.005 | 38.259 | 8.354 |
| verify 1 | 6.683 | 8.532 | 6.628 | 0.055 | 2.991–3.248 | 14.471 | 3.160 |
| verify 16 | 18.916 | 19.278 | 18.536 | 0.380 | 2.791–2.954 | 48.075 | 10.497 |
| verify 32 | 24.233 | 28.696 | 23.914 | 0.319 | 5.615–5.779 | 59.979 | 13.096 |

`<50% proxy` integrates the sum of active nodes’ *average logical read rates*. It is not measured HBM utilization. The interval excludes unknown split-partial intervals at its lower end and treats their rate as zero at its upper end. The stream floor is the logical compulsory read count / 4.58 TB/s. An ideal continuous-stream latency is `floor + L(qkv→attention) + L(router→routing)`, where unavoidable dependent latency remains unmeasured. Observed gaps below are scheduling intervals, not a measurement of those unavoidable minima. Writes, cache hits, redundant reads, dequantization, and intrinsic attention compute are not represented by this read-only floor.

Per shape (µs; bandwidth is logical TB/s):

| Shape | CUPTI span | Event raw / corrected | Σ kernels | Busy union | No-kernel | Overlap | <50% proxy | Read MB | Stream floor | capture A/B | CUPTI/off |
|---|---|---|---|---|---|---|---|---|---|---|---|
| d1S | 63.62 | 66.78 / 63.87 | 116.22 | 62.78 | 0.83 | 53.44 | 19.36 | 139.5 | 30.47 | 1.0140 | 1.0206 |
| d1F | 73.98 | 77.31 / 74.40 | 128.35 | 73.28 | 0.70 | 55.07 | 35.33 | 159.8 | 34.89 | 1.0033 | 1.0211 |
| d16S | 115.26 | 119.14 / 116.22 | 184.29 | 114.18 | 1.09 | 70.11 | 52.64 | 297.7 | 65.00 | 0.9919 | 1.0062 |
| d16F | 234.94 | 240.22 / 237.31 | 394.88 | 234.02 | 0.93 | 160.87 | 4.83 | 740.4 | 161.66 | 1.0021 | 0.9965 |
| d32S | 160.35 | 164.10 / 161.18 | 220.48 | 159.49 | 0.86 | 60.99 | 18.14 | 454.5 | 99.23 | 0.9949 | 1.0043 |
| d32F | 331.23 | 335.42 / 332.51 | 422.46 | 330.11 | 1.12 | 92.35 | 10.62 | 1084.6 | 236.82 | 0.9998 | 1.0008 |
| v1S | 96.99 | 99.65 / 96.74 | 125.44 | 96.06 | 0.93 | 29.38 | 57.47 | 198.9 | 43.44 | 1.0039 | 1.0199 |
| v1F | 138.85 | 141.73 / 138.82 | 172.58 | 138.08 | 0.77 | 34.50 | 37.79 | 332.5 | 72.59 | 0.9982 | 1.0142 |
| v16S | 280.61 | 281.73 / 278.82 | 282.98 | 274.12 | 6.49 | 8.86 | 52.48 | 712.3 | 155.52 | 0.9990 | 1.0116 |
| v16F | 375.52 | 379.42 / 376.51 | 391.33 | 370.43 | 5.09 | 20.90 | 33.76 | 956.9 | 208.93 | 0.9930 | 1.0022 |
| v32S | 323.26 | 325.09 / 322.18 | 391.78 | 317.50 | 5.76 | 74.27 | 99.81 | 768.3 | 167.75 | 1.0024 | 1.0083 |
| v32F | 585.15 | 585.31 / 582.40 | 667.17 | 581.79 | 3.36 | 85.38 | 74.27 | 1539.9 | 336.22 | 1.0032 | 1.0081 |

Per-stage span, compulsory input reads, effective read rate and fraction of 4.58 TB/s:

| Shape | Stage | Span µs | Completion increment µs | Read MB | B/span TB/s | Roof % | Span − read/roof µs |
|---|---|---|---|---|---|---|---|
| d1S | qkv3 | 7.26 | 7.26 | 20.988 | 2.889 | 63.1 | 2.68 |
| d1S | attn | 17.63 | 15.84 | 0.584 | 0.033 | 0.7 | 17.50 |
| d1S | gate | 14.30 | 1.57 | 16.796 | 1.174 | 25.6 | 10.64 |
| d1S | o | 15.17 | 2.14 | 16.779 | 1.106 | 24.2 | 11.50 |
| d1S | b1 | 5.66 | 2.02 | 0.147 | 0.026 | 0.6 | 5.63 |
| d1S | pool_router | 24.64 | 19.87 | 75.514 | 3.065 | 66.9 | 8.15 |
| d1S | route | 10.78 | 4.19 | 0.128 | 0.012 | 0.3 | 10.76 |
| d1S | moe | 12.19 | 3.78 | 0.016 | 0.001 | 0.0 | 12.19 |
| d1S | unpooler_slice | 3.87 | 4.06 | 8.397 | 2.169 | 47.3 | 2.04 |
| d1S | b2 | 5.34 | 2.88 | 0.197 | 0.037 | 0.8 | 5.30 |
| d1F | qkv3 | 8.32 | 8.32 | 20.988 | 2.523 | 55.1 | 3.74 |
| d1F | attn | 25.47 | 23.90 | 20.851 | 0.819 | 17.9 | 20.92 |
| d1F | gate | 24.93 | 3.87 | 16.796 | 0.674 | 14.7 | 21.26 |
| d1F | o | 14.27 | 1.98 | 16.779 | 1.176 | 25.7 | 10.61 |
| d1F | b1 | 2.91 | 2.02 | 0.147 | 0.051 | 1.1 | 2.88 |
| d1F | pool_router | 21.92 | 20.00 | 75.514 | 3.445 | 75.2 | 5.43 |
| d1F | route | 10.24 | 2.88 | 0.128 | 0.013 | 0.3 | 10.21 |
| d1F | moe | 11.49 | 3.90 | 0.016 | 0.001 | 0.0 | 11.48 |
| d1F | unpooler_slice | 3.94 | 4.22 | 8.397 | 2.133 | 46.6 | 2.10 |
| d1F | b2 | 5.28 | 2.88 | 0.197 | 0.037 | 0.8 | 5.24 |
| d16S | qkv3 | 7.36 | 7.36 | 21.234 | 2.885 | 63.0 | 2.72 |
| d16S | attn | 18.27 | 16.16 | 9.339 | 0.511 | 11.2 | 16.23 |
| d16S | gate | 14.98 | 1.76 | 17.072 | 1.140 | 24.9 | 11.25 |
| d16S | o | 19.07 | 5.66 | 16.810 | 0.881 | 19.2 | 15.40 |
| d16S | b1 | 7.65 | 2.40 | 1.671 | 0.219 | 4.8 | 7.28 |
| d16S | pool_router | 25.12 | 21.18 | 75.760 | 3.016 | 65.8 | 8.58 |
| d16S | route | 11.07 | 3.39 | 1.787 | 0.161 | 3.5 | 10.68 |
| d16S | moe | 43.55 | 35.42 | 85.075 | 1.953 | 42.7 | 24.98 |
| d16S | unpooler | 18.50 | 19.14 | 67.240 | 3.635 | 79.4 | 3.81 |
| d16S | b2 | 19.17 | 2.78 | 1.704 | 0.089 | 1.9 | 18.80 |
| d16F | qkv_gate | 10.94 | 10.94 | 38.011 | 3.473 | 75.8 | 2.64 |
| d16F | attn | 112.35 | 107.33 | 339.099 | 3.018 | 65.9 | 38.31 |
| d16F | o | 109.95 | 5.50 | 16.810 | 0.153 | 3.3 | 106.28 |
| d16F | b1 | 10.24 | 2.40 | 1.671 | 0.163 | 3.6 | 9.88 |
| d16F | pool_router | 30.91 | 21.86 | 75.760 | 2.451 | 53.5 | 14.37 |
| d16F | route | 11.62 | 3.10 | 1.787 | 0.154 | 3.4 | 11.23 |
| d16F | moe | 71.52 | 62.30 | 198.321 | 2.773 | 60.5 | 28.22 |
| d16F | unpooler | 18.34 | 18.85 | 67.240 | 3.667 | 80.1 | 3.65 |
| d16F | b2 | 19.42 | 2.66 | 1.704 | 0.088 | 1.9 | 19.05 |
| d32S | qkv_gate | 11.36 | 11.36 | 38.273 | 3.369 | 73.6 | 3.00 |
| d32S | attn | 24.80 | 18.50 | 19.104 | 0.770 | 16.8 | 20.63 |
| d32S | o | 12.64 | 6.34 | 16.843 | 1.332 | 29.1 | 8.96 |
| d32S | b1 | 9.06 | 2.82 | 3.244 | 0.358 | 7.8 | 8.35 |
| d32S | pool_router | 29.89 | 23.10 | 76.022 | 2.544 | 55.5 | 13.29 |
| d32S | route | 11.49 | 3.30 | 3.556 | 0.310 | 6.8 | 10.71 |
| d32S | moe | 78.88 | 70.53 | 226.765 | 2.875 | 62.8 | 29.37 |
| d32S | unpooler | 20.83 | 21.28 | 67.371 | 3.234 | 70.6 | 6.12 |
| d32S | b2 | 21.95 | 3.14 | 3.277 | 0.149 | 3.3 | 21.24 |
| d32F | qkv3 | 7.62 | 7.62 | 21.496 | 2.822 | 61.6 | 2.92 |
| d32F | attn | 202.11 | 199.36 | 677.018 | 3.350 | 73.1 | 54.29 |
| d32F | gate | 29.31 | 2.40 | 17.367 | 0.592 | 12.9 | 25.52 |
| d32F | o | 23.04 | 5.86 | 16.843 | 0.731 | 16.0 | 19.36 |
| d32F | b1 | 8.80 | 2.94 | 3.244 | 0.369 | 8.0 | 8.09 |
| d32F | pool_router | 29.12 | 23.23 | 76.022 | 2.611 | 57.0 | 12.52 |
| d32F | route | 10.82 | 3.36 | 3.556 | 0.329 | 7.2 | 10.04 |
| d32F | moe | 71.33 | 63.20 | 198.453 | 2.782 | 60.7 | 28.00 |
| d32F | unpooler | 19.65 | 20.38 | 67.371 | 3.429 | 74.9 | 4.94 |
| d32F | b2 | 21.06 | 2.88 | 3.277 | 0.156 | 3.4 | 20.34 |
| v1S | qkv_gate | 10.91 | 10.91 | 37.880 | 3.471 | 75.8 | 2.64 |
| v1S | attn | 21.54 | 18.02 | 1.050 | 0.049 | 1.1 | 21.31 |
| v1S | o | 5.95 | 5.66 | 16.794 | 2.822 | 61.6 | 2.29 |
| v1S | b1 | 6.08 | 2.11 | 0.885 | 0.146 | 3.2 | 5.89 |
| v1S | pool_router | 23.68 | 20.54 | 75.629 | 3.194 | 69.7 | 7.17 |
| v1S | route | 10.02 | 3.07 | 0.902 | 0.090 | 2.0 | 9.82 |
| v1S | moe | 36.93 | 29.47 | 56.697 | 1.535 | 33.5 | 24.55 |
| v1S | unpooler_slice | 3.94 | 4.32 | 8.454 | 2.147 | 46.9 | 2.09 |
| v1S | b2 | 4.90 | 2.88 | 0.655 | 0.134 | 2.9 | 4.75 |
| v1F | qkv_gate | 11.14 | 11.14 | 37.880 | 3.402 | 74.3 | 2.87 |
| v1F | attn | 37.15 | 30.78 | 21.318 | 0.574 | 12.5 | 32.50 |
| v1F | o | 6.40 | 6.02 | 16.794 | 2.624 | 57.3 | 2.73 |
| v1F | b1 | 6.78 | 2.08 | 0.885 | 0.130 | 2.8 | 6.59 |
| v1F | pool_router | 24.83 | 21.38 | 75.629 | 3.046 | 66.5 | 8.32 |
| v1F | route | 10.18 | 3.42 | 0.902 | 0.089 | 1.9 | 9.98 |
| v1F | moe | 64.35 | 56.64 | 169.944 | 2.641 | 57.7 | 27.25 |
| v1F | unpooler_slice | 4.32 | 4.54 | 8.454 | 1.957 | 42.7 | 2.47 |
| v1F | b2 | 4.90 | 2.85 | 0.655 | 0.134 | 2.9 | 4.75 |
| v16S | qkv_gate | 13.47 | 13.47 | 39.846 | 2.958 | 64.6 | 4.77 |
| v16S | attn | 23.43 | 24.23 | 12.088 | 0.516 | 11.3 | 20.79 |
| v16S | o | 7.39 | 8.00 | 17.039 | 2.305 | 50.3 | 3.67 |
| v16S | b1 | 6.34 | 7.14 | 6.390 | 1.008 | 22.0 | 4.94 |
| v16S | pool_router | 26.08 | 26.53 | 77.595 | 2.975 | 65.0 | 9.14 |
| v16S | route | 2.88 | 3.49 | 2.377 | 0.825 | 18.0 | 2.36 |
| v16S | moe | 170.62 | 171.23 | 482.361 | 2.827 | 61.7 | 65.30 |
| v16S | unpooler | 20.19 | 20.61 | 68.157 | 3.375 | 73.7 | 5.31 |
| v16S | b2 | 5.98 | 5.92 | 6.423 | 1.073 | 23.4 | 4.58 |
| v16F | qkv_gate | 14.27 | 14.27 | 39.846 | 2.792 | 61.0 | 5.57 |
| v16F | attn | 151.62 | 151.90 | 341.635 | 2.253 | 49.2 | 77.02 |
| v16F | o | 7.10 | 7.55 | 17.039 | 2.399 | 52.4 | 3.38 |
| v16F | b1 | 5.41 | 6.30 | 6.390 | 1.182 | 25.8 | 4.01 |
| v16F | pool_router | 26.66 | 27.04 | 77.595 | 2.911 | 63.6 | 9.71 |
| v16F | route | 2.91 | 3.52 | 2.377 | 0.816 | 17.8 | 2.39 |
| v16F | moe | 138.14 | 138.53 | 397.427 | 2.877 | 62.8 | 51.37 |
| v16F | unpooler | 19.62 | 20.06 | 68.157 | 3.475 | 75.9 | 4.73 |
| v16F | b2 | 6.30 | 6.34 | 6.423 | 1.019 | 22.2 | 4.90 |
| v32S | qkv_gate | 20.80 | 20.80 | 41.943 | 2.016 | 44.0 | 11.64 |
| v32S | attn | 28.99 | 29.82 | 18.277 | 0.630 | 13.8 | 25.00 |
| v32S | o | 9.02 | 9.44 | 17.302 | 1.917 | 41.9 | 5.25 |
| v32S | b1 | 8.16 | 8.70 | 12.681 | 1.554 | 33.9 | 5.39 |
| v32S | pool_router | 32.45 | 33.12 | 79.692 | 2.456 | 53.6 | 15.05 |
| v32S | route | 3.71 | 4.29 | 4.736 | 1.276 | 27.9 | 2.68 |
| v32S | moe | 181.95 | 182.53 | 511.730 | 2.812 | 61.4 | 70.22 |
| v32S | unpooler | 25.28 | 25.73 | 69.206 | 2.738 | 59.8 | 10.17 |
| v32S | b2 | 8.83 | 8.83 | 12.714 | 1.440 | 31.4 | 6.06 |
| v32F | qkv_gate | 20.83 | 20.83 | 41.943 | 2.013 | 44.0 | 11.67 |
| v32F | attn | 264.32 | 264.64 | 676.617 | 2.560 | 55.9 | 116.59 |
| v32F | o | 9.06 | 9.31 | 17.302 | 1.911 | 41.7 | 5.28 |
| v32F | b1 | 7.97 | 8.35 | 12.681 | 1.592 | 34.7 | 5.20 |
| v32F | pool_router | 33.57 | 33.92 | 79.692 | 2.374 | 51.8 | 16.17 |
| v32F | route | 3.68 | 4.22 | 4.736 | 1.287 | 28.1 | 2.65 |
| v32F | moe | 209.73 | 210.11 | 624.976 | 2.980 | 65.1 | 73.27 |
| v32F | unpooler | 24.67 | 25.09 | 69.206 | 2.805 | 61.2 | 9.56 |
| v32F | b2 | 8.64 | 8.67 | 12.714 | 1.472 | 32.1 | 5.86 |

Signed stage dependency envelopes (negative means overlapping nodes, including possible PDL waits):

| Shape | QKV end → attention start µs | Pool/router end → routing start µs |
|---|---|---|
| d1S | -1.792 | -6.592 |
| d1F | -1.568 | -7.360 |
| d16S | -2.112 | -7.680 |
| d16F | -5.024 | -8.512 |
| d32S | -6.304 | -8.192 |
| d32F | -2.752 | -7.456 |
| v1S | -3.520 | -6.944 |
| v1F | -6.368 | -6.752 |
| v16S | 0.800 | 0.607 |
| v16F | 0.288 | 0.608 |
| v32S | 0.832 | 0.576 |
| v32F | 0.320 | 0.543 |

Each stage owns only its extension of the preceding stage-completion frontier. This removes double counting of PDL overlap; it is still a scheduling model, since reads may start before that frontier.


Largest three completion-increment residuals (mean µs per 62-layer load; model, not measured off-HBM time):

- **moe: 1998.9 µs.** Exposed completion increment minus compulsory reads at 4.58 TB/s; may include instruction work, barriers, repeated L2 reads, memory ramp/tail, PDL waiting and overlap.
- **attn: 1703.9 µs.** Exposed completion increment minus compulsory reads at 4.58 TB/s; may include instruction work, barriers, repeated L2 reads, memory ramp/tail, PDL waiting and overlap.
- **pool_router: 461.5 µs.** Exposed completion increment minus compulsory reads at 4.58 TB/s; may include instruction work, barriers, repeated L2 reads, memory ramp/tail, PDL waiting and overlap.

PDL example: decode-16 full O spans 109.95 µs but extends the preceding completion frontier by only 5.50 µs. Treating its full node span as exposed O work would misidentify the bottleneck.


Actual routed expert counts (important for interpreting the byte model):

| Shape | Distinct local experts | Local routes | Experts >16 tokens |
|---|---|---|---|
| d1S | 0 | 0 | 0 |
| d1F | 0 | 0 | 0 |
| d16S | 3 | 8 | 0 |
| d16F | 7 | 12 | 0 |
| d32S | 8 | 21 | 0 |
| d32F | 7 | 16 | 0 |
| v1S | 2 | 2 | 0 |
| v1F | 6 | 6 | 0 |
| v16S | 17 | 73 | 0 |
| v16F | 14 | 66 | 0 |
| v32S | 18 | 140 | 2 |
| v32F | 22 | 149 | 2 |

FA3 metadata/combine byte attribution was audited offline: metadata reads sequence lengths; combine reads split accumulators, whose exact count was not retained. The raw receipt preserves the lab’s original estimates. `nodes.csv` includes both archived and audited bytes. The low-50 proxy was recomputed; timestamps and stage byte totals did not change.


Every original kernel node is in [nodes.csv](nodes.csv), with CUPTI node ID, stream, start/end, signed next-start gap, compulsory logical bytes, attribution class and effective rate. Unknown internal reads are marked `unresolved*`, never inferred from durations.

Payload SHA-256: `ce30e3c3a3d8212cb070e7a6c47f48a278c464e7e575ae0119f58ae6bcae67de`. Normal and forced local screens each passed all 13 official cases and 23 static checks; [validation.json](validation.json) keeps the measured local observer ranges. The final standalone embedded payload also completed all 12 shapes locally. Selected-only setup matched full setup on 36 tensors and 36 three-seed fixtures in both normal and forced modes.
The unchanged prefill host scored 1.42974×; that is not a decode candidate gain.

Integration: diagnostics only; no candidate patch or promotion. Next measurement: use decode-calibrated routing, retain FA3 split metadata, and add data-ready/first-consumption stamps plus DRAM counters to identify actual PDL waits, dependent latency and HBM duty. Those quantities are not established by this trace.
