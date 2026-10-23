# Old projection singles: queue triage

For Jihoon/orchestrator; no withdrawal or send was performed here.

cand_lt_layout.py: subsumed by cand_portfolio_all
cand_skinny.py: subsumed by cand_portfolio_all
cand_reduce_hier.py: keep: hierarchical split-K32/64 with BK128 and extra stage counts absent from the portfolio's three hierarchical templates
cand_wg.py: subsumed by cand_portfolio_all
cand_reduce_packed.py: subsumed by cand_portfolio_all
cand_tile.py: keep: staged cp.async mma.sync tile_kernel at M32/128; this kernel is absent from cand_portfolio_all
cand_fused_lean.py: subsumed by cand_portfolio_all
cand_reduce_fast.py: subsumed by cand_portfolio_all
cand_gemv_static.py: subsumed by cand_portfolio_all
cand_tc_k256.py: subsumed by cand_portfolio_all
cand_wg2.py: subsumed by cand_portfolio_all
cand_tc_k256_qkv.py: subsumed by cand_portfolio_all
cand_tuned2.py: subsumed by cand_portfolio_all
cand_fused.py: subsumed by cand_portfolio_all
cand_reduce.py: subsumed by cand_portfolio_all
cand_tc.py: subsumed by cand_portfolio_all

The portfolio includes skinny configurations from the original/tuned banks, all four GEMV bodies,
TC slab16/32 xK128/256, fused/last-CTA/DSMEM/Stream-K and Lt TN/NN.
Its hierarchical templates are only (64,64,4), (128,64,3), (256,64,2); reduce_hier additionally
tries valid BK128 and stage4/5 templates at split32/64, so it is not fully subsumed.
The wg/wg2 singles never set wg_ok, so their compiled WG paths do not execute; they add no evidence.
The tile single stages activations and weights for a distinct mma.sync ring, not the tc_slab body.

Coverage now exercises the actual current Hopper enumerator and dispatcher across all30 keys: 7581 configurations, 12 families, strict portable ABI stubs.
It reproduces the missing old_hopper error from archived commit1d6a356. The builder template
now mirrors the owner-fixed source and cannot reintroduce the removed old family. Queued source
was not edited. The original validator fabricated an old_hopper stub; that stub is now removed,
and its default entrypoint invokes strict enumeration first. Evidence: runs/screen/prefill_gemm/portfolio_hopper.json.
