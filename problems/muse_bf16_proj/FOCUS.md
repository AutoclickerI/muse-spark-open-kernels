# Projections in decode and serving

Serves: G3, G2; read [GOAL](../../docs/GOAL.md) and [CONSTRAINTS](../../docs/CONSTRAINTS.md).
The fitted overall board takes max(decode, component composite): proj alone currently
cannot move our rank. Put its winners inside decode and serving; top 50 needs ~1.54x decode.
This round is solo, GPU0 only; the orchestrator sends and commits.

Ready: **`../muse_decode_layer/cand_r3_proj.py`**, based on frozen autotune-q1 `099ee216…`.
It tunes 24 projection keys at T=1/8/16/32/128/256: qkv_gate, o, pool_router and unpooler/slice.
Ported: skinny/GEMV winners, one-pass hi+lo router, mixed pooler/router, strided pooler+router,
TMA/WGMMA direct/last-CTA/DSMEM/Stream-K, and all checked Lt TN/NN/split-K algorithms.
Projection trials include the next consumer; every projection preserves `(partials, splits)`.
Two fresh inputs must agree within 1% with cuBLAS; failing trials are logged and dropped.
FA3's 27 settings and PDL remain tunable, with a shared 240 s trial deadline.

New complete-layer H200 evidence: pure V9 `0e94268cf5654012` **1.427904x**;
grouped dispatch `2ad75c19cb514f5c` **1.265668x**, losing heavily at verify32.
Use proven V9 throughout this projection experiment; synthetic MoE timing misranked that path.
Local **sm_120**: **13/13 tests, 12/12 graphs, 29.05 s prepare**, no trial rejections.
Additional admission: **2,742** native configurations ×3 fresh graph replays, **730** forced
Hopper ABI cases (including FA3 tuple returns/qkv tuples), **16** injected rejections, sm_90a build.
H200 correctness, selected projections, speedup and cold prepare time remain unverified.

Standalone evidence remains in [FINDINGS](../../docs/FINDINGS.md#projection-evidence-and-next-experiment-g3-g2):
board **1.181998x**, router m128/256 **10.133/11.792 us**; small-m router ~**6.1 us**.
Only `cand_portfolio_all.py` was sent; the other broad portfolios are subsets for this task.
Builders now reject coarse/finalist/layout failures and stop trials after 240 s; queued sources stay frozen.
[dev/NEXT_CHECKS.md](dev/NEXT_CHECKS.md) owns hashes, queue state, validation commands and the next send.
