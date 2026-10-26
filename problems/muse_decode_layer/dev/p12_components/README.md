# P12 component diagnostics: frozen S15

Purpose: rank the remaining dense, attention and expert gaps against component-board leaders. Overall score still comes from the decode layer. Base is exact S15 `60254883`; component adapters preserve the reviewed NF MIT notice and are generated under1MB. No active decode candidate is changed here.

MoE preserves the NT1 and heavy/light stream paths, EP8 local32 experts, routing and output ABI. Attention preserves S15's selected decode, FA3 and VQ8 producers and dynamic-head correctness path. Projection preserves the S15 producers; the current board adapter reduces/casts in one Triton kernel, including router hi+lo. Separate boards cannot measure the layer's pool/router fusion or gate/attention overlap.

## Measured ranking

Public `okc leaderboard <problem> --json` snapshots contain aggregate value and speedup only. Leader per-shape times are unavailable; the documented job endpoint is owner-only (`upstream/llms.txt:78`). `shape_times.csv` preserves all48 of our shape timings and explicitly marks leader shape times unpublished. These aggregate gaps rank opportunities; they are not whole-layer effect estimates.

| Rank | Stage | SGLang/ours,95% CI | Time/leader | Tests/shapes | Job |
|---:|---|---|---:|---|---|
| 1 | attention | 1.64698 [1.62228,1.67169] | 1.79873 | 16/12 | `4e108beb9a0a4cdb` |
| 2 | projection | 1.20236 [1.18432,1.22039] | 1.36125 | 9/30 | `08cc94050a4e4564` |
| 3 | moe | 2.02924 [1.99880,2.05968] | 1.25835 | 5/6 | `97f5e7f3c3fc4159` |

Component receipts use the official paired SGLang controls; independent A:A is **unavailable**, not zero. All correctness/static gates passed. Attention is the largest aggregate gap, followed by projection and MoE. The original projection adapter `d6ad694a27954c80` scored0.98059x [0.96588,0.99530]; its conversion chain overstated the layer's projection gap.

Paired adapter lab **4e87b1d44d334104** completed30/30 shapes×8 seeds. One-reducer B/A is0.5607–1.0132, A:A per-cell max deviation1.1–6.23%; all FP64 and one-hot gates pass. The corrected adapter was admitted on the component board as **08cc94050a4e4564**,1.20236x [1.18432,1.22039]. Producers are identical. This promotes only the board adapter; the layer already has fused consumers. `reduce_results.json` has every shape and producer-only attribution.

MoE us at T1/8/16/32/128/256:30.878/72.944/105.207/138.927/207.993/268.911. Attention d1/d16/d32/v1/v16/v32 us: sliding9.320/13.245/16.862/17.826/22.707/28.369; full19.439/104.390/195.318/29.597/139.703/235.585.

## Verification and reproducibility

`screen.py` runs tools/local_eval.py in the judge image on GPU0, normal and forced, three inputs per scored shape. MoE5tests/6shapes, attention16/12, projection9/30; pinned admissions include25static checks. Hopper-only execution uses explicit local arithmetic stand-ins while retaining real launch/allocation glue. Changed CUDA compiles sm90a; unchanged source+flags reuse the exact S15 compile proof. No stand-in is shipped and sm120 timing does not establish H200 gains.

`compare.py`, `analyse_reduce.py` and `findings.py` regenerate reports. `own_receipts.py` supplements ledger entries lost during shared-worktree updates only from a catalog with an exact source SHA and matching local receipt; it never edits the shared ledger. `receipt_catalog.json`, `admission.json`, `reduce_admission.json`, `leader_gap.json` and `shape_times.csv` pin the evidence. Generated candidates, wrappers and verbose logs remain ignored.

All three original reset component jobs are complete. Attention's scored-only waiter was replaced by an exact-source check when only check capacity remained. Original queue epoch1792972800/priority4 is historical; there are no pending component jobs. No new geometry sweeps are queued: attention's large gap has priority; the small pool winner is handed to P4 separately.
