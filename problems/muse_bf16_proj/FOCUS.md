# Decode verify projections: M128/256 (G3, serving)

Solo; GPU0 only; no commits. Jihoon authorizes proj/lab sends this round.
Decode candidates remain owned by that lane; only a local integration fixture was built.
Control union215cafaf8d674c7e, M128/M256 raw graph us:
qkv24.320/27.328; pool_router32.320/37.120; unpooler27.392/32.032; o13.920/15.680.
Union custom at M>64 is cuBLAS; FP32 pool_router excludes Lt; Lt lacks CUDAContext.h.
Do not subtract pure device floors from raw graph replay time.

Library596b2a5d436e4fee landed: M128 qkv21.10->20.22us/unpool24.00->22.96us.
Other Lt choices near parity; default7-key shortlist embedded. Native v2 queued priority0:
prof/lab/verify_proj_tma_v2.py, payloadb0cfbb95563cfec3…; NEXT_CHECKS has full hashes.
It combines K64/K128/K256 commit groups,2–6stage TMA ring,160/288threads,
activation multicast2, DSMEM in-kernel K merge,132SM persistent grids and PDL chains.
Native v1 accepted6b72087f64584258 is INVALID for selection: reinitialized live barriers.
No execution cancellation API. V2 invalidates before reinit and uses cluster acquire.
Native multicast is extremely slow on sm120 (NVIDIA documents other-target penalties).
Final local transport stand-in uses ordinary TMA; Hopper keeps real multicast/WGMMA.

V2 sm120:2148 fresh checks,342 configs/dtype;73 Hopper families forced/1247 calls.
PDL chains M128/256×bothflags×fresh3 pass; host9/9+30/30 graphs,2.091s prepare.
Full drop-in fixture13/13+12/12 graphs,3.264s prepare; all SM90a static checks pass.
Shortlist/dispatch32 trials+140 fresh cases; module/unknown-family/budget rejection passes.
Block dev/verify/projection_dropin.py, SHA97c874743a23…; exact ABI/H200 table in README.
All7 H200 layouts pass layer13/13+12/12 locally,4.877s; full H200 layer effect pending.

Dense prefill: cand_v11_gemm has no H200 result yet (d318167e was tma_cluster's run, misattributed); re-sent 10-24.
Glue parked; tile721290f2=.808x closed. MoE belongs to its lane until receipts.
Owner-fixed portfolio_all, dflash_diag and reduce_hier remain frozen/queued.
