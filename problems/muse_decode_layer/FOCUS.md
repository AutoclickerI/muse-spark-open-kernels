# Decode layer focus — G3 first

Serves: G3, G2.

Goal: raise the overall-board score before the 2026-11-05 17:00 KST close.
Decode carries 51% of the fitted board time weight; serving uses its verify shapes.

Control: the latest H200 scored dispatch receipt is `85e67f5d6ef945c9`, 1.366737x (CI
[1.346236, 1.387237]); its exact source is `cand_r3_v9_dispatch.py`, SHA
`633c96ccffa40c6673c99b808c18b28685db5aafd4541667de68e1f0e6c41b84`.
`cand_r3_v9_cost_autotune.py` is already first in the queue from `dafac99`; do not resend it.
Do not edit sent sources. The orchestrator owns checks and scored runs.

H200 profile bottleneck: full attention is the largest remaining decode cost at
verify32/decode32/verify16 = 373.8/227.9/192.6 us; projections and glue are the next
non-MoE costs. MoE V11/V17 row-tile work is owned by the MoE lane.

Standalone H200 evidence: MoE V9 reaches 1.654x overall and beats V7 at every load. The scored V9 dispatch layer now reaches 1.366737x overall. The MoE lane owns the next row-tile work; this lane's largest remaining lever is full attention, led by verify32.

The next candidate `cand_r3_v9_attention_portfolio.py` keeps the cost candidate's full
portfolio and adds the attention lane's FA3 settings, XQA/WGMMA warp and grouped families,
FA3-structured WGMMA, FP8-PV, and the four-stage q_len=1 cp.async gather. It uses the
layer's raw qkv/gate/KV layout and selects per `(batch, q_len, window)` in `prepare()`;
each invalid, resource-failing, or tolerance-failing variant is logged and rejected.
Local sm_120 admission is 13/13 tests and 12/12 fresh-input graphs; Hopper attention
branches use the layer path as a portable stand-in locally and are measured only on the judge.

Structural fusion candidates reduced launch count but regressed complete local layers;
keep them diagnostic until a packed variant is neutral or better. Read [NEXT_CHECKS](dev/NEXT_CHECKS.md)
for frozen source hashes and order. No remote run or commit is part of this lane round.
