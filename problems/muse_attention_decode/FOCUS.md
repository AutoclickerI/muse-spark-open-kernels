# Attention verify
Serves: G2, G3.

Goal: beat FA3 on q_len=8 full B16/B32 first, then sliding; preserve q_len=1.
`submission.py` now matches check32c5b4b287c24c38 exactly:1.390910x overall,
SHA610a590120ae79c446248cce5b6a857e283b425290aba67af29e18283c96c810.
This meets the owner's promotion threshold, inside noise, with FA3 on all verify
shapes. Board stays1.384x (526f31018e2c4449) until orchestrator scores promotion.

K64 diagnostic32c5b4b287c24c38 and K128a0511c0001464dbe both lose all verify
shapes. K128 paired full B16/B32 custom617/1174us vsFA3207/391us.
Stages3–5 barely change full times; nominal KV throughput0.53–0.56TB/s vs
standalone gather2.558TB/s. Profiles show selected FA3, not custom stalls.
The synchronous recipe is closed; producer overlap/WGMMA remain open.

Next candidate:cand_xqa_next.py packs K128 swap-AB mma.sync and K64 WGMMA.
Both use128 producer threads, cp.async/full-empty mbarriers,32–64KiB in flight,
64 queries on N, exact causal/2048-window mask,FP32 softmax,last-CTA FP32 merge.
Hopper WGMMA QK:E4M3; PV:exact FP16 V. sm_120 runs matching MMA stand-ins.
prepare validates and times16 variants +FA3 per shape, then pairs the best.
All actual input work stays in kernel_fn; only shape choices survive prepare.

Admission, frozen hashes and09:00 order:dev/NEXT_CHECKS.md.
SM90a:zero spills; QGMMA/HGMMA/LDGSTS confirmed. CUTLASS layout/descriptor
checks pass. Actual WGMMA correctness/performance needs H200.
Pending FA3 tune:dd927c6c46264afe; pipeline remains in local sender queue.
Only orchestrator sends. Do not edit sent candidates; no commit this round.
