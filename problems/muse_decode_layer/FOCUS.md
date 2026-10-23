# Decode layer focus — G3 first

Serves: G3, G2.

Goal: raise the overall-board score before the 2026-11-05 17:00 KST close.
Decode carries 51% of the fitted board time weight; serving uses its verify shapes.

Control: the latest H200 scored dispatch receipt is `85e67f5d6ef945c9`, 1.366737x (CI
[1.346236, 1.387237]); its exact source is `cand_r3_v9_dispatch.py`, SHA
`633c96ccffa40c6673c99b808c18b28685db5aafd4541667de68e1f0e6c41b84`.
Do not edit sent sources. The orchestrator owns checks and scored runs.

H200 profile bottleneck: verify16/32 MoE is 286/362/499/503 us (sliding/full),
49.8–76.9% of candidate time. Decode32 sliding MoE is 119 us (54.8%).
Full verify32 attention is next at 377.5 us; projections and boundaries are smaller.

Standalone H200 evidence: MoE V9 reaches 1.654x overall and beats V7 at every load. The scored V9 dispatch layer now reaches 1.366737x overall, but verify32 remains the largest excess: full attention is 377.5 us in the accepted profile and MoE dominates verify16/32.

The next candidate `cand_r3_v9_cost_autotune.py` keeps V7, V9, grouped wide tiles, FA3's 27 `(num_splits, pack_gqa)` settings, cuBLAS/cuBLASLt/custom projections, PDL on/off for both residual boundaries, and a fused O-projection/residual/pooler prefix path plus a persistent small-load island path. Its `prepare()` selects by shape on the judge, logs every timing, and rejects invalid/resource-wrong variants instead of aborting. Local sm_120 admission is 13/13 tests and 12/12 fresh-input graphs; Hopper FA3/cuBLASLt branches are forced through explicit stand-ins or skipped locally.

Structural fusion candidates reduced launch count but regressed complete local layers;
keep them diagnostic until a packed variant is neutral or better. Read [NEXT_CHECKS](dev/NEXT_CHECKS.md)
for frozen source hashes and order. No remote run or commit is part of this lane round.
