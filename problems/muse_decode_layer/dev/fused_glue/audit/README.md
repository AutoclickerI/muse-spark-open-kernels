# Why the first native glue mapping loses

H200 job80b70f7f20fa41c6 rejects every arm (1.51–17.63x whole-layer).
The local SM120 reproducer isolates the same failure without MoE/attention.
These are work-deletion diagnostics, never candidates. H200 phase attribution
is inferred from this local causal test and the identical source mapping.

| M | Pool full -> no prologue us | Unpool full -> no boundary us | O full -> no boundary us |
|---|---:|---:|---:|
| 1 | 673.8 -> 120.8 | 63.5 -> 17.4 | 74.8 -> 19.5 |
| 8 | 1224.7 -> 149.5 | 368.6 -> 19.5 | 340.0 -> 21.5 |
| 16 | 2082.8 -> 172.0 | 460.8 -> 85.0 | 350.2 -> 22.5 |
| 32 | 3861.5 -> 199.7 | 466.9 -> 119.8 | 337.9 -> 36.9 |

The initial grid is not short: pool launches576/576/1152/2304 CTAs at M1/8/16/32.
It redundantly recomputes the entire8192-column boundary576 times per token:
72 output-row tiles x8 K splits. At M32 that is18,432 full-row scans.
One warp handles each token with256 serial column iterations; only one warp
does useful norm work at M1, and the prologue repeats before every matrix tile.
Deleting that prologue removes81–95% of its local total cost.

The opposite failure occurs after GEMM: only1/1/2/4 last CTAs survive for
M1/8/16/32, regardless of the original grid size. Each owns up to8 full rows.
At M>=8 four warps each process two token rows through repeated256-iteration
scalar scans; at M1 the special whole-CTA path still uses serial64-element
per-thread scans. Deleting the boundary removes74–95% of O/unpool time.
The pool routing epilogue itself costs roughly49–82us locally, rather than
the few-us parallel Triton route consumer it replaced.

PDL on/off changes these local stage times by at most about3%; no source spin
exists. Each losing CTA returns after its acq/rel arrival. The SM90a build has
zero register spills. Source review therefore excludes an unresident-grid
barrier and PDL launch ordering as the cause of the large penalty. The matrix
mainloop remains slower than the incumbent too (skinny8-token tiles, scalar
shared-fragment loads); deleting glue does not turn it into a measured win.

This is a performance mapping defect, not a numerical/launch correctness bug.
It closes this replicated-row prologue / single-CTA scalar epilogue mapping.
It does not establish a bound against fusion. A replacement must keep each
norm cooperative and avoid repeating a whole-row reduction for every output/K
tile; routing metadata should be produced once and consumed by the expert grid.
No replacement projection mapping or remote run was built from this diagnosis.

Reproduce locally with
`CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_decode_layer/dev/fused_glue/audit/diagnose.py`,
then run `python problems/muse_decode_layer/dev/fused_glue/audit/summarize.py`.
