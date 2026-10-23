# MoE checks for 09:00 KST (20-check round)


## Distinct grouped TMA design: ready for orchestrator

`cand_tma_grouped.py`, SHA256 **b1209b2ac260654999c11a915606dcfb784dc3373c6e3a26da001a3b2c9e163c**.
Single file, unchanged complete V9_phase prefix /incumbent; twelve grouped choices at
M128/256. One resident launch includes route planning, both GEMMs, SwiGLU and norm/sum.
Model-realistic router fixtures, independent FP321.2% global/6% row gate, any Exception
rejects,230s overall trial deadline;20 cold fresh paired finalist samples. Summary last.
Local sm120:5/5 tests +6/6 graphs,7.935s cached prepare.276 direct graph/math checks,
24 strict native ABI stand-ins, six rejection phases +wrapper/budget failures pass;
four SM90a specializations compile with all m64n8/16/32/64 WGMMA/TMA paths.
Direct real-router M128 nativeV9/grouped403.54/402.48us; M256700.45/580.64us.
M256 actual V9_phase/Humming incumbent563.30us is faster locally; no H200 gain claimed.
No other lane candidate edited; no send/commit by this lane. The existing V19 queue
remains owned by the MoE lane. [Full design/results](tma_grouped/README.md).

No remote job is launched from this lane. The orchestrator owns the queue and sends only
locally screened files. `cand_v8.py` and `cand_v9.py` are frozen/sent; this round preserves
both. Frozen hashes: V8 `7ab6818fcbb8f28b53686ed90362e4950fe4a642891836dfab9bcabb56c390be`,
V9 `a933dac686bbb25f0acc5f4675dd277b3a1362d041647e30c8263ba8845f89f5`, V11
`c01f691e4fd283a1a58a9ae7bc577838177cd76c85ee04fc12efb9104052a18a`, V12
`c455c2072cc72a33588dbc474aa4bfa8c9bc5ea731f15b5272926fc246f484ca`, V17
`967c926acf29b67175ae7b8c8c0847071b9d371ca52bf468d23cc00f47474a91`, V18
`3cb4f1054b2297978d2a09cddde0c832b4a02e747e275f8bc5faa078c59dbd78`, diagnostic
`cand_v9_phase.py` `84eb293228a3e3132a73cd5ee3f6ac1d252abe9b2384a70090e1540239a09e7e`, hybrid
`d0fd42118ab8fe2ae8a85429b0b67f68a7647c49780dd4f31fdc704b49bd6487`, and Humming6k
`f91fa07c8e65971c540ffa4c75f762816aada78e513f6045fc9c2b34b41f4db3`.

| Order | Source | Purpose |
|---:|---|---|
| 1 | `cand_v18.py` | V17 portfolio with the 1% gate and optional router-aware `moe_tune` hook; standalone screen passes. |
| 2 | `cand_v17.py` | Committed V17 graph/cold-L2 portfolio control. |
| 3 | `cand_v9_phase.py` | Packed H200 clock64 diagnostic with V9 phase spreads and variant timings. |
| 4 | `cand_v12.py` | V11 dispatch plus a three-stage V9 ring probe at M=128. |
| 5 | `cand_v11.py` | Frozen H200-best V9 + Humming dispatch, receipt `41bcf27cfc0e4eff` (1.664x). |
| 6 | `cand_v9.py` | Frozen scored V9 control, receipt `f6028a478d594b31` (1.654x). |
| 7 | `cand_hybrid.py` | Frozen H200 hybrid control. |
| 8 | `cand_humming6k.py` | Frozen Humming-wide control. |
| 9 | `cand_v8.py` | Frozen V8 control. |

Local evidence: V18 passes 5/5 correctness and 6/6 graphs, 1.264x sm120 GM, with an 8.3 s cached prepare; V17 remains 1.269x. The decode router bench records real/matched/uniform M256 timings in `problems/muse_decode_layer/dev/moe_route_bench.log`; `moe_tune.py` is the reusable router-aware builder. H200 claims include V11 `41bcf27cfc0e4eff` (1.664x), V9 `f6028a478d594b31` (1.654x), hybrid `318227bc382d465c`, Humming6k `45a82e56c24f49f7`, V7 `f95bdf5e394e474e`, and V6 `eefecf1eeb724feb`.
