# Next H200 checks — 2026-10-24 09:00 KST

No checks/submissions were launched for this structural round. Reset is 2026-10-24
00:00 UTC. The coordinator owns execution and commits. `submission.py` remains v3:
best check 1.1428x (`2c112482e81745af`), scored 1.137375x (`f17fad6758ed46d0`).

Re-read `runs/ledger.jsonl` and decode profiles before using this order: the nine repaired
round-2 checks are still pending at this report's snapshot. adapt2/3/4 have now returned
and all fail the already-repaired tuple/device integration, without timing. Those are
older candidates, not the nine repaired ones. Use any new *whole-layer* H200 evidence to
choose dispatch; do not promote isolated or local wins as measured layer wins.

Read [STRUCTURAL.md](STRUCTURAL.md) for all local stage timings, comparisons against v3
and all nine round-2 candidates, exact hashes and validation. Every proposed candidate
has 13/13 local tests and 12/12 fresh-input graph cases on sm_120. These timings are
screening evidence, not H200 predictions. The new fusion currently loses locally.

| Order | Candidate | What this check answers | Local/static evidence |
|---:|---|---|---|
| 1 | [cand_layer_sweep.py](../cand_layer_sweep.py) | One diagnostic check for projection tiles/cuBLAS, unfused vs epilogue vs persistent prefix, v4 vs v6 experts, suffix fusion, and PDL/prefetch overlap. It prints `LAYER_SWEEP` lines from synthetic weight-shaped work in prepare. Its judged output uses the fixed chain-v6 implementation. | 13/13 + 12/12; preparation logs are under 16,000 characters locally. Cold-L2 graph timing, no workload inputs in prepare, no retained outputs. See [log](local_layer_sweep.log). This is an information check, not a proposed scored promotion. |
| 2 | [cand_epilogue.py](../cand_epilogue.py) | Conservative fusion: O epilogue publishes row statistics; pooler/router forms the residual and normalized BF16 input during loads, removing one boundary launch and hidden-buffer round trip. Tests this approach with measured-best v6 experts and a complete PDL chain. | 13/13 + 12/12; eight new epilogue specializations compile for sm_90a. Local overall **0.967x v3**, V1 **0.978x** on common inputs; least local regression among v6 structural alternatives in the initial comparison. |
| 3 | [cand_chain6.py](../cand_chain6.py) | Full persistent dense islands around v6 experts: O through top-8/pre-expert norm, and unpooler through final residual/norm. Tests whether Hopper's launch/tail savings repay internal barriers. | 13/13 + 12/12; static sm_90a compile, zero ptxas spills in the main dense templates. Local overall **0.952x v3**, V1 **0.925x**. Actual T16/T32 graph: six compute kernels + one fill, six programmatic edges; overlap verified on sm_120. |
| 4 | [cand_chain.py](../cand_chain.py) | Hold experts at v4 to isolate structural fusion from v6 scheduling/residency. Compare directly with the queued `cand_moe4_attn.py` result once it arrives. | 13/13 + 12/12; local **0.978x v3**, V1 **0.951x**. This is the local-best whole-island control, though standalone H200 evidence favors v6. |
| 5 | [cand_chain_occ2.py](../cand_chain_occ2.py) | Request two resident dense CTAs per SM only for NT1/T<=8. Retain one-CTA launch bounds at T16/32, and clip every grid to actual occupancy. Tests whether greater memory concurrency pays without the spills from forcing two CTAs everywhere. | 13/13 + 12/12 on its final exact source; sm_90a compile with no spills. NT1 uses 118/124 registers for suffix/prefix; NT2/4 retain the ordinary templates. See [compiler evidence](sm90a_occ2/dense.compile.log) and [stage comparison](stage_comparison_occ2.json). |
| 6 | [cand_chain_nopf.py](../cand_chain_nopf.py) | Disable the persistent islands' O/unpooler L2 prefetch, retaining all arithmetic and PDL. A control for contention/capacity versus useful latency hiding. | 13/13 + 12/12; local **0.949x v3**, V1 **0.930x**. Synthetic local attention+prefix pairs show only small prefetch effects; Hopper uses bulk prefetch instead of the local per-line implementation. |

The first diagnostic may make orders 4–6 unnecessary. In particular, prefer the ordinary
chain when extra residency or prefetch loses in complete stage pairs. The retired forced
`launch_bounds(256,2)` version is **not** a check candidate: it spilled at NT2/4 and lost
badly at T32. Its evidence is retained in `occ2_forced.py`, `local_occ2_forced.*` and
`sm90a_occ2_forced/`.

The prefetch experiment covers only immutable, known dense weights: O (16 MiB), and a
bounded unpooler prefix (8 MiB sliced or 16 MiB full). Hopper emits
`cp.async.bulk.prefetch.L2.global`; sm_120 emits `prefetch.global.L2`. The epilogue
pooler/router also prefetches its own known slices before waiting. Expert weight loads
remain downstream of current-call routing. Prefetching every dense bank at once was
rejected because later banks can evict the next-used O weights; no gain is claimed for it.

At T>32 these candidates retain the existing projection/boundary/FA3 structure. Their
large-verify improvement hypothesis comes from v6 experts, whose standalone H200 check
`eefecf1eeb724feb` is 1.4916x, not from a new all-token-count megakernel. This matters for
serving, which uses verify loads. A new H200 result must establish both full-layer speed
and correctness before a scored promotion or serving port.

Static compilation does not exercise Hopper FA3, actual cold-L2 behavior, or the judge's
router seed. Cold-cache judge compilation time is also unverified. Local cached prepare
is below the 300 s limit, including the diagnostic; inspect its runtime and compile logs
before expanding the sweep. Exact source hashes in the local JSON must match the files
sent for checking. Use `tools/okc_run.py` for future authorized checks; this document does
not schedule or launch them.
