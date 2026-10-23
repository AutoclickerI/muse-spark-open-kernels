# Dense tuning stopped; grouped MoE next (G1/G3)

Read [GOAL](../../docs/GOAL.md) and [CONSTRAINTS](../../docs/CONSTRAINTS.md).
Solo, GPU0 only. Jihoon/orchestrator sends and commits; this thread did neither.
Queued at priority2.0: **`../muse_prefill_layer/cand_v11_gemm.py`**, frozen **`1b30690ad9e4…`**.
Full SHA and commands: [NEXT_CHECKS](dev/NEXT_CHECKS.md). New owned candidate only;
V11 experts/attention/norms unchanged; no other lane source or queued source edited.

Single file embeds the dense block; V11's exact concatenated router +routing +pooler
is always the group incumbent. Real consumer graphs compare four hybrid/fused choices.
Dense controls are V11 matmuls; any failing variant logs/rejects, retaining V11.
Three fresh probes; cold128MiB;20-pair finalists; shared210s dense/225s group deadlines.
Optional early H200 profiler identifies the actual stock kernel per dense role.
Every compact selection prints last: local tail50 lines /4731 chars, all retained.
Local **sm_120**: **5/5 official tests,8/8 graph shapes,173.16s prepare**.
Strict Hopper stand-ins:136 QKV configs /7 families;12 group faults +wrapper fault rejected.
18 new swapped WGMMA specializations +12 packing layouts compile SM90; Lt builds SM90a.

New QKV: token chunks512/1024/2048/4096/8192 (direct output), NN/TN layouts;
zero-padded N2560/3072/4096, and operand-swapped Lt/persistent paths with timed packing.
M16384 focused20-pair sm_120 best: chunk4096 NN **2380.9->2286.1us** (local only).
Direct H200 dafbcad0 supersedes the44%/146ms qkv inference: library qkv766–806us,
pooler/unpooler1.34–1.47ms, o338–406us, router178–204us, all near80%peak.
Profile-name attribution was wrong; no cache-cause evidence remains. Stop dense tuning.
[Corrected evidence](../../docs/reference/prefill_gemm.md). This thread now owns a distinct
M128/256 persistent grouped TMA MoE design, separate from the MoE lane's V19.

Serving module `layers/muse_prefill_gemm.py`; copyable `dev/prefill_gemm_block.py` updated.
Jihoon applied triage991e838:14 singles withdrawn/deleted; reduce_hier and tile remain queued.
`cand_dflash_diag.py` remains frozen/queued (9b4fae2f); its receipt is pending from Jihoon.
