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

Corrected H200 check **`c6bd22e2084c4809`**, payload **`ce30e3c3`**, is accepted.
Both selected-only setup comparisons passed: 36 exact tensor checks and 36
whole-layer fixtures (three seeds × twelve shapes) in normal and forced modes.
Routes matched; maximum relative L2 was 0.000116 / 0.000074. The exact standalone
selected-only payload completed twelve shapes locally in 16.9 seconds. Awaiting
the H200 receipt; no candidate speedup is claimed.
