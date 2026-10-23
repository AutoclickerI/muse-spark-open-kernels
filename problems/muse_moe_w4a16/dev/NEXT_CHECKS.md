# MoE checks for 09:00 KST (20-check round)

No remote job is launched from this lane. The orchestrator owns the queue and sends only
locally screened files. `cand_v8.py` and `cand_v9.py` are frozen/sent; this round preserves
both. Frozen hashes: V8 `7ab6818fcbb8f28b53686ed90362e4950fe4a642891836dfab9bcabb56c390be`,
V9 `a933dac686bbb25f0acc5f4675dd277b3a1362d041647e30c8263ba8845f89f5`, V11
`c01f691e4fd283a1a58a9ae7bc577838177cd76c85ee04fc12efb9104052a18a`, V12
`c455c2072cc72a33588dbc474aa4bfa8c9bc5ea731f15b5272926fc246f484ca`, hybrid
`d0fd42118ab8fe2ae8a85429b0b67f68a7647c49780dd4f31fdc704b49bd6487`, Humming6k
`f91fa07c8e65971c540ffa4c75f762816aada78e513f6045fc9c2b34b41f4db3`, diagnostic
`cand_v9_phase.py` `84eb293228a3e3132a73cd5ee3f6ac1d252abe9b2384a70090e1540239a09e7e`, and
portfolio `cand_v17.py` `9b682ae6b33f174e4a86869ebe35aa27a38365d1d60e409c7cd93bb2d727e86c`.

| Order | Source | Purpose |
|---:|---|---|
| 1 | `cand_v17.py` | Portfolio: graph/cold-L2 autotunes V9, V12, V16, Humming-wide, and adaptive 8/16/32/64-row expert-major reuse; all paths and layouts stay resident. |
| 2 | `cand_v9_phase.py` | Packed H200 clock64 diagnostic with V9 phase spreads and variant timings; V16 remains embedded, not standalone. |
| 3 | `cand_v12.py` | V11 dispatch plus a three-stage V9 ring probe at M=128; retain as a control. |
| 4 | `cand_v11.py` | Frozen H200-best V9 + Humming dispatch, receipt `41bcf27cfc0e4eff` (1.664x). |
| 5 | `cand_v9.py` | Frozen scored V9 control, receipt `f6028a478d594b31` (1.654x). |
| 6 | `cand_hybrid.py` | Frozen H200 hybrid control; compare only if portfolio choices regress. |
| 7 | `cand_humming6k.py` | Frozen Humming-wide control; retain only if grouped glue regresses. |
| 8 | `cand_v8.py` | Frozen V8 control; no edits. |

Local evidence: V17 passes 5/5 correctness and 6/6 graphs, 1.269x sm120 GM, with a 174.0 s prepare; it selects V9 at M=1/8/32, V12 at M=16/128, and adaptive at M=256 on sm120. V9-phase remains a passing 1.261x diagnostic. H200 claims include V11 `41bcf27cfc0e4eff` (1.664x), V9 `f6028a478d594b31` (1.654x), hybrid `318227bc382d465c`, Humming6k `45a82e56c24f49f7`, V7 `f95bdf5e394e474e`, and V6 `eefecf1eeb724feb`.
