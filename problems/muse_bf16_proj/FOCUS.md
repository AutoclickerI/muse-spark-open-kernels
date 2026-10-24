# Prefill glue portfolio ready (G1)

Read GOAL/CONSTRAINTS; solo, GPU0 only. Orchestrator sends/commits; none launched here.
Ready `../muse_prefill_layer/cand_glue_prefill.py`, frozen2a2e3914ea26….
Full SHA/interface/evidence: `../muse_prefill_layer/dev/glue/README.md`.
Direct H200 inventory: `docs/reference/prefill_glue.md` (all three loads).
Board-best controls are a30fb411 /99820fa6; glue profile entries hidden below top12 stay N/A.
Boundary pair:851.7–855.2us at16K,671.1us byte floor;159ms per first turn /125ms floor.
Separate dequant633.6–651.3us is expert-stage work; no double-counting its bytes/time.

New choices: GEMM last-CTA residual/two-RMS epilogues; pooler RMSNorm epilogue with
normalized expert consumers; router hi+lo GEMM/top8 epilogue; vector boundary and topk.
The full original producer+consumer is always eligible. H200 dense/expert/attention
controls seed from a30fb411, leaving230s prepare budget for new glue trials.
Three fresh banks, cold128MiB, CUDA graphs,20-pair finalists, independent FP32 gate;
all optional exceptions reject. Compact114-row summary is6906 chars and survives tail.
Final sm120:5/5 official tests +8/8 graphs,48.111s prepare, all78 trials/36 finalists.
Direct M33/16384:60 configs /540 fresh checks /1200 timed pairs /18 counter resets.
Forced all-fusion integration5/5; forced Hopper prepare/library/router branches pass.
60 SM90 specializations compile (48 WGMMA); H200 execution/speed remains unverified.
Local only small vector/topk gains; full GEMM epilogues lose, pooler keeps stock.
O serving fusion cannot cross TP all-reduce; vector boundary remains usable after it.

Dense tuning stopped: direct H200 dafbcad0 refutes44%/146ms qkv inference.
`cand_v11_gemm.py` stays frozen1b30690a…, queued2.0. Do not edit/resend.
`cand_tma_grouped.py` stays frozenb1209b2a…, queued MoE reset priority.
`cand_dflash_diag.py` stays frozen9b4fae2f…, queued; receipt pending from Jihoon.
Projection triage applied:14 subsumed singles deleted; reduce_hier/tile stay queued.
