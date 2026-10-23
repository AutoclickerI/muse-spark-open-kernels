# Decode layer focus — G3 first

Serves: G3, G2.

Goal: raise the overall-board score before the 2026-11-05 17:00 KST close.
Decode carries 51% of the fitted board time weight; serving uses its verify shapes.

Control: the latest H200 scored dispatch receipt is `85e67f5d6ef945c9`, 1.366737x (CI
[1.346236, 1.387237]); its exact source is `cand_r3_v9_dispatch.py`, SHA
`633c96ccffa40c6673c99b808c18b28685db5aafd4541667de68e1f0e6c41b84`.
`cand_r3_v9_attention_portfolio.py` is queued from `c14edac`; do not resend it. The next
source is the merged portfolio `cand_r3_merge.py`.
Do not edit sent sources. The orchestrator owns checks and scored runs.

H200 profile bottleneck: full attention is the largest remaining decode cost at
verify32/decode32/verify16 = 373.8/227.9/192.6 us; projections and glue are the next
non-MoE costs. MoE V11/V17 row-tile work is owned by the MoE lane.

Standalone H200 evidence: pure V9 reaches 1.428x overall (rank 85), with 1.216/1.136x
at 256 tokens versus V7's 1.105/1.071x; synthetic grouped timing is misleading inside
the layer. The MoE lane owns future routing-aware choices; this merged layer fixes MoE to V9.

`cand_r3_merge.py` holds every non-MoE family: cost-v9 fused O-projection/pooler prefix,
persistent small-load island, the attention FA3/XQA/WGMMA/FP8-PV/q1-gather portfolio, and
the projection lane's skinny/mix/TMA/Lt variants. It tunes each stage per shape with MoE
fixed to V9, logs each timing and rejection, and enforces one 285-second prepare budget.
Local sm_120 admission is 13/13 tests and 12/12 fresh-input graphs; Hopper branches use
portable stand-ins locally and are measured only by the judge.

Structural fusion candidates reduced launch count but regressed complete local layers;
keep them diagnostic until a packed variant is neutral or better. Read [NEXT_CHECKS](dev/NEXT_CHECKS.md)
for frozen source hashes and order. No remote run or commit is part of this lane round.
