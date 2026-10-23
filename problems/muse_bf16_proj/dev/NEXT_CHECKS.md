# Projection next checks

`cand_portfolio.py` is frozen at SHA256
`57086918cce13065f97757c2471b7b79072e2bc45e8c589899723bf1d70d748e`.
The orchestrator queued it as a **scored run, priority 0**; do not resend or edit it.
Its pending job/receipt will appear under `runs/pending/` / `runs/muse_bf16_proj/`.

## Ready for the orchestrator

1. `problems/muse_bf16_proj/cand_reduce.py` — passing, frozen fusion experiment.
   SHA256: `280822ad5f91aae8e20e9440032f3f3ce22043119179375c4087059ee0c887f6`.
   Router and qkv m128/256 always select a single kernel. Router hi+lo products update the
   same FP32 accumulators in each K tile; one activation tile serves both. Split-K merges
   through last-CTA or DSMEM, with no standalone reduce/add. Local sm_120: 9/9 tests,
   30/30 graph shapes, 770 parameter/replay cases, 1,075 forced Hopper glue cases;
   **1.093592x overall**, router 1.327x/1.161x and qkv 1.053x/0.979x at m128/m256.
   Kineto verifies baseline router 5 kernels -> 1, qkv 2 -> 1 at both m values.
   sm_90a build passes. H200 latency remains unverified.
   Screen: `runs/screen/proj_reduce/muse_bf16_proj.cand_reduce.json`.
   Details: `dev/reduce_*.log`, `dev/cand_reduce_kernel_counts_local.json`.

`cand_reduce_packed.py` is under validation: it packs the same math and reduction with
128-thread cooperative issue, a 160-thread producer-warp pipeline, and the original
256-thread producer-warpgroup pipeline. Do not send it until its frozen hash is recorded.
`cand_reduce_fast.py` is the cooperative diagnostic (local 1.078570x, 9/9 + 30/30);
it is folded into packed and is not a separate judge recommendation.

## Follow-through

- No okc_run launch or commit from this thread; the orchestrator sends passing sources.
- Checks are spent until 09:00 KST; the orchestrator chooses scored submission or waits for reset.
- Read the portfolio's selected configs and per-shape H200 table, prune losing variants into
  `cand_portfolio_v2.py`, then screen/freeze it. Do not claim synthetic timings as judge timings.
- Event-driven receipt reader: `python3 problems/muse_bf16_proj/dev/receipt_digest.py --watch`.
  It reads receipts and pending markers; it does not submit, collect or poll.
- Reproduce fusion screen:
  `CUDA_VISIBLE_DEVICES=0 tools/img python tools/local_eval.py problems/muse_bf16_proj/cand_reduce.py --json runs/screen/proj_reduce/muse_bf16_proj.cand_reduce.json`.
  Extra checks: `tools/img python problems/muse_bf16_proj/dev/validate_reduce.py kernels` / `glue`,
  and `tools/img python problems/muse_bf16_proj/dev/profile_reduce.py` (GPU 0).

## Receipts and inherited queue

Read all 11 completed projection receipts through `122997f357d04fa2`. The scored
`68e2f34b1d4744cb` stays at 1.142642x. GEMV checks `a3b29a26372c4b35` (1.112106x) and
`122997f357d04fa2` (1.114312x) lose overall; their pooler M32 cuBLAS branch beats the scored
skinny branch (0.991/0.996x versus 0.930x), now folded into the portfolio. Other same-algorithm
large-M differences and o M1's ~1% difference are inside noise, not per-shape discoveries.

At handoff, all 13 inherited dispatch waiters are alive; no new receipt has landed for them.
Do not resend or edit them: checks dsv3, dsv3_select, gemv3, gemv_static, m16, skinny, tc,
tc_k256, tc_k256_qkv; scored tile, tuned2, wg, wg2. The P8 hardware probe `849bd4b650e54c7f`
also has no completed receipt. Their identities/snapshots live in `runs/waiting/` and
`runs/pending/`; fold new per-shape winners into a new frozen portfolio revision.
The wg/wg2 files never set `wg_ok`, so their queued scores do not exercise WGMMA.

Projection docs: STATUS.md was folded into FINDINGS and a 24-line FOCUS.md.
The project-docs checker reports zero errors and one warning in the concurrently edited
decode lane FOCUS.md; the projection docs have no warnings. Cold-read agent test was not
run because this round is solo. Skill revision: `5205a93`.
