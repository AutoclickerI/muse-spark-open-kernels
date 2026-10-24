# Decode verify projections: CUTLASS split-K /Stream-K

Solo, GPU0 only, no commits. Send labs only; decode candidates go to Jihoon.
Never poll/wait for judge: inspect receipts once per turn; build, then end if pending.
M128/256 exact W[N,K]: qkv2304x8192, o8192x1024, unpool8192x4096,
pool_router4608x8192=[pool4096;router_hi256;router_lo256], FP32 output.
Union215cafaf raw graph us M128/M256: qkv24.32/27.33, pool32.32/37.12,
unpool27.39/32.03, o13.92/15.68. M>64 custom delegates to cuBLAS.

Library596b2a5d436e4fee: M128 qkv21.10->20.22, unpool24.00->22.96us;
other Lt choices near parity. Existing dev/verify/projection_dropin.py retains that shortlist.
Custom TMA v2 landed92e7802783a54212: all8 keys lose1.21–1.95x.
PDL chains also lose. This closes that implementation, not TMA/WGMMA generally.
V1 6b72087f64584258 remains invalid (live mbarrier reinit); never select it.

Sent priority0: prof/lab/verify_proj_cutlass.py, payloadSHAe43ed9b225d98cf1…;
hostSHA b81c9564402fa62c…; full hashes/status in dev/NEXT_CHECKS.md.
Image CUTLASS4.5 Hopper3.x cooperative collectives +Stream-K scheduler.
16 compiled specializations;144 recipes/key: splitK2/4/8/16, Stream-K/DP,
multicast clusters, swapped operands, K64/128,3/auto stages,132/98/66 SMs.
Required counter reset timed; cold1GiB, fresh3,20 paired finalists; exceptions reject.
sm120 host9/9+30/30 graphs+3 static checks; all1152 Hopper ABI recipes forced.
Drop-in dev/cutlass/projection_cutlass.py preserves project()->(partials,splits).
No unmeasured native default is installed. Full-layer H200 effect remains unknown.

#45015 port belongs to serving lane; no duplicate edits. Owned dense modules have no
prepared dequantized NVFP4 bank. Raw17 scales stay bytes; legacy V9 recoding noted in FINDINGS.
MoE grouped withdrawn/parked: c8c89f9f loses; synthetic Humming/V11 fail real-weight agreement.
Dense cand_v11_gemm is genuinely requeued; d318167e was expert_tma_cluster, not our dense run.
