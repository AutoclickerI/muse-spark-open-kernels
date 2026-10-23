# Prefill focus
Serves: G2, G1.

- Rank 6/P2; optimize H200 score and the serving prefill path at 753–16K tokens.
- H200 control: v11 check `6a9cd943fe404d74` 1.367883x; scored `965a9a816d4744ea` 1.366811x.
- Scored profile first-turn gaps: expert GEMM+dequant 917.137 ms, dense/router 781.297 ms,
  attention 264.493 ms; tensor floor 1,295.661 ms.
- New tuner routes random BF16 hidden rows through the prepared router hi+lo plus bias,
  chooses the busiest 32-expert range, and constructs fresh local slots/counts.
- It compares the old uniform fixture and router-derived fixture, but dispatches the realistic winner.
  Every graph, oracle, timing and finalist error is logged `SCORE_REJECT` and dropped.
- Three frozen routing-aware complete portfolios are ready; exact order and hashes are in
  `dev/NEXT_CHECKS.md`: fused expert, dense fusion, then score portfolio.
- `cand_fused_expert_portfolio_routing.py` has 45 expert choices; score has 32; dense fusion
  retains structural front/back, 45 experts and six attention paths.
- sm_120: all three pass 5/5 official tests and 8/8 graph shapes; prepare 4.2–4.4 s with
  Hopper tuning dormant. Local route diagnostic shows route-dependent choices and timings; close sm_120 choices can vary.
- H200 cold-L2 graph selection and receipts are unverified; no local timing is an H200 claim.
- Serving KV is 15,872 B/token/GPU: 1.270/5.079/20.316/40.632 GB at c1/4/16/32 and 80K;
  2.080/8.321/33.286/66.572 GB at the 131K cap. One full alternate bank is 54.358 GB/GPU.
- Serving port facts from the folded notes: preserve TP/EP collectives, BF16 norm boundaries,
  separate gate/up/down scales, capture padding and per-request KV mapping; agreement is open.
- Old `RESULTS.md`, `PORTING.md`, and `dev/LIVE_STATUS.md` facts are folded into `docs/FINDINGS.md`;
  this file is the lane entry point and stays under 30 lines.
- The orchestrator sends and commits; the obsolete promotion handler and judge process are gone.
