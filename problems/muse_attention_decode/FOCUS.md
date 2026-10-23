# Attention verify — Serves: G2, G3

Goal: beat FA3 on q_len=8 full B16/B32 first, then sliding; preserve q_len=1.
Submission: custom decode plus FA3 verify; 1.384x scored `526f31018e2c4449`,
1.388x check `4f8c41ef9e924ff4`. No new H200 promotion yet.

Next: `dev/NEXT_CHECKS.md` holds frozen hashes and queue order.
The old `cand_xqa_diag.py` stays unchanged; its sender is waiting locally,
not yet accepted by the judge. Read its `XQA_DIAG` when the receipt arrives.
`cand_xqa_k128.py` is the sent eight-warp packed diagnostic; do not edit it.
`cand_xqa_pipeline.py` is the private K128 specialization; it removes the
runtime RoPE branch on full attention, raises occupancy, and keeps the packed
prepare-time FA3 dispatch.

K128 design: one KV head / CTA, 64 queries on MMA N, keys on M; raw E4M3 QK;
FP32 online softmax; exact FP16 V or raw FP8 PV via ldmatrix/packed transpose;
3/4/5-stage page-row cp.async (32/48/64 KiB outstanding), SM-sized split-KV,
in-kernel last-CTA FP32 merge; separate causal K32 tile for the eight new keys.
Every key and exactly 2048 sliding positions are attended. No cached outputs.

The exact K128 source passes 16/16 tests and 12/12 graph shapes; both PV
formats pass fresh graph replays through 1M context. Clock64 profiling found
the largest per-tile costs are gather issue/address work (~2.1k cycles), QK
(~1.2k), softmax (~1.2k), widened-PV (~1.1k), and FP16 widening (~0.9k); the
cross-SM merge mark is not usable as a latency measure. ncu is blocked by
ERR_NVGPUCTRPERM. The private full-specialized K128 pipeline measures 346/676
us for verify B16/B32 on sm_120; these are not H200 claims.

Only the orchestrator sends runs. Keep old H200 receipts; infer no H200 wins
from local times or stand-ins.