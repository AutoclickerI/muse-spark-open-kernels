# Decode layer focus — G3 first

Serves: G3, G2.

Goal: raise the overall-board score before the 2026-11-05 17:00 KST close.
Decode carries 51% of the fitted board time weight; serving uses its verify shapes.

Control: the latest H200 scored dispatch receipt is `85e67f5d6ef945c9`, 1.366737x (CI
[1.346236, 1.387237]); its exact source is `cand_r3_v9_dispatch.py`, SHA
`633c96ccffa40c6673c99b808c18b28685db5aafd4541667de68e1f0e6c41b84`.
`cand_r3_v9all.py` goes first at the 09:00 KST reset and `cand_r3_merge.py` goes second.
The prior persistent megakernel `cand_r3_merge_mega.py` is held: its qkv producer was
nondeterministic at M=32 and changed verify routing IDs. Both persistent megakernel sources are held: `cand_r3_merge_mega.py` has a qkv race, and
`cand_r3_merge_mega_fix.py` still rejects a persistent dense island at M=16 after regular qkv.
The attention v2 source is parked after its proxy gate rejected faster verify variants; the
attention lane owns the incoming vstream. This lane's glue fusion diagnostic `cand_r3_v9_glue.py` is parked after profiling found
four split-K weight rereads and a second consumer launch still dominate; the regular v9 path wins
every local shape. The overlap experiment remains parked because its verify race failed the
correctness guard.
`cand_r3_union.py` is the diagnostic portfolio; `cand_r3_best.py` is the next single-run
candidate. It repairs the undefined Hopper Lt/XQA branches, keeps V9 as the MoE fallback, and
selects per scored shape from the q_len=1 V9/FA3/decode-attention family, q_len=8
FA3/XQA/WGMMA/FP8-PV plus vstream/two-group attention, custom/cuBLAS/cuBLASLt projection
layouts, router-derived V9/moe_rt/Humming/adaptive choices, and v9all PDL boundaries. Attention choices use
the vstream FP32 self-reference gate at 0.8 of the decode bound; projection and MoE choices
use the proven V9 result within 1%. Its compact `UNION_SUMMARY` is printed last in prepare.
The selected combination also passes a small FP32 full-layer self-reference gate before the
state is returned; a failure resets every optional plan to V9/custom/layer defaults.
The union H200 receipt regressed 7--13 us on small shapes because cuBLAS/cuBLASLt pool-router
selection added a separate nvjet GEMV and split-K reduction. `cand_r3_best.py` preserves custom
pool/router GEMV at T≤32, adds the corrected cost/projection portfolio and embeds the standalone
XQA matrix/group/FP8-PV families. Prefix fusion stays off because its prior 128/256 tiles exceeded
the 232,448-byte Hopper shared-memory limit; attention and projection choices reject failures and
the V9 layer remains the fallback.
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
The held megakernel extension's persistent qkv and dense-island choices are disabled after
repeated shape-dependent agreement failures. The glue candidate leaves V9 MoE and attention intact, fuses O projection → residual/norm →
pooler/router with PDL, and feeds the existing PDL route/top-8/softmax/pre-expert-norm kernel.
Its sm_120 tuner rejects the fused path locally on timing at every shape; the source is parked.
There is no separate shared/dense branch after boundary1: its normalized
hidden and residual stream share the same statistics; pooler/router→experts, experts→unpooler,
router→attention output, and KV write→attention remain ordered dependencies. Local sm_120 admission of parked `cand_r3_v9_glue.py` is 13/13 tests and 12/12 fresh-input graphs;
prepare is 4.386 s after token tiling; fused glue still loses regular v9 at every shape, and Hopper branches use portable stand-ins.

Structural fusion candidates reduced launch count but regressed complete local layers;
keep them diagnostic until a packed variant is neutral or better. Read [NEXT_CHECKS](dev/NEXT_CHECKS.md)
for frozen source hashes and order. No remote run or commit is part of this lane round.
