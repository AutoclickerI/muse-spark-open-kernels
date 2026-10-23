# Projection next checks

1. `problems/muse_bf16_proj/cand_portfolio.py` — **passing, frozen; send one check**.
   SHA256: `57086918cce13065f97757c2471b7b79072e2bc45e8c589899723bf1d70d748e`
   Note: "P7 packed m128/256 portfolio: cold-L2 synthetic graph tuning of TMA/WGMMA
   BT64/128/256, grouped K64/K128, 2–5 stages, split-K with last-CTA in-kernel merge,
   DSMEM clusters through 16, balanced Stream-K at one/two SM waves, and up to 32
   cuBLASLt heuristic algorithms. Fused permitted hi+lo router; retain measured small-M
   kernels and fold landed pooler M32 cuBLAS winner. Every path has local kernel/glue
   coverage; each tuning graph owns a native stream to avoid PyTorch #193402."

The orchestrator sends this file. No okc_run was launched and no commit was made in this round.
Do not promote over `submission.py` until the H200 receipt supports it.

## Local admission evidence (sm_120)

- Exact-source screen: 9/9 tests and 30/30 graph shapes, 20 replays × 3 fresh input sets;
  **1.078233x** locally, not an H200 measurement; prepare 0.78 s warm / 53.4 s cold.
- Screen: `runs/screen/proj_round/muse_bf16_proj.cand_portfolio.json`.
- Manifest: `runs/screen/proj_round/muse_bf16_proj.portfolio_validation.json`.
- TMA/MMA pipeline + last-CTA/DSMEM/Stream-K: 840 passing cases including outliers,
  probe rows, graph replays with changed inputs, and counters reset after each launch.
- Forced Hopper prepare/public dispatch: 1,091 passing cases. Oversized sm_90 shared-memory
  branches use a portable GEMM stand-in locally; cuBLASLt algorithms execute natively.
- sm_90a cross-build: 22 WGMMA specializations, zero local bytes/spills, 58–154 registers;
  SASS has HGMMA/TMA instructions. WGMMA cannot execute on the local GPU.
- Full validation and SASS/resource logs: `problems/muse_bf16_proj/dev/portfolio_*.log`.

Reproduce:
```sh
mkdir -p runs/screen/proj_round
CUDA_VISIBLE_DEVICES=0 tools/img python tools/local_eval.py problems/muse_bf16_proj/cand_portfolio.py --json runs/screen/proj_round/muse_bf16_proj.cand_portfolio.json
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_bf16_proj/dev/validate_portfolio.py kernels
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_bf16_proj/dev/validate_portfolio.py glue
```

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
