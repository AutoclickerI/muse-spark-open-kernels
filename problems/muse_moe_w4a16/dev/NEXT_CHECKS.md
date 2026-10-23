# MoE checks for 09:00 KST (20-check round)

No remote job is launched from this lane. The orchestrator owns the queue and sends only
locally screened files. `cand_v8.py` and `cand_v9.py` are frozen/sent; this round preserves
both. Frozen hashes: V8 `7ab6818fcbb8f28b53686ed90362e4950fe4a642891836dfab9bcabb56c390be`,
V9 `a933dac686bbb25f0acc5f4675dd277b3a1362d041647e30c8263ba8845f89f5`, V11
`c01f691e4fd283a1a58a9ae7bc577838177cd76c85ee04fc12efb9104052a18a`, V12
`c455c2072cc72a33588dbc474aa4bfa8c9bc5ea731f15b5272926fc246f484ca`, and diagnostic
`cand_v9_phase.py` `84eb293228a3e3132a73cd5ee3f6ac1d252abe9b2384a70090e1540239a09e7e`.

| Order | Source | Purpose |
|---:|---|---|
| 1 | `cand_v9_phase.py` | Packed H200 diagnostic: plain V11 kernel plus clock64 V9 phase spread and V9/V12/V16/Humming timings at all six shapes; V16 is embedded, not a standalone check. |
| 2 | `cand_v12.py` | V11 dispatch plus a prepare-time three-stage V9 ring probe at M=128; Humming3 remains M=256+. Screened 5/5 and 6/6 graphs; promote only if the shape choice wins on H200. |
| 3 | `cand_v11.py` | Frozen V9 + Humming3 dispatch; current production candidate and control. |
| 4 | `cand_v9.py` | Frozen scored V9 control; use receipt to confirm native choices and nearby-shape behavior. |
| 5 | `cand_hybrid.py` | Frozen H200 hybrid control (V9 native through M192, Humming above); compare against V11's single-bank dispatch. |
| 6 | `cand_humming6k.py` | Frozen Humming wide control; retain only if V11's wide glue regresses. |
| 7 | `cand_v8.py` | Frozen V8 control; no edits and no promotion ahead of V9. |

Local evidence: the diagnostic passes 5/5 correctness and 6/6 graphs, 1.261x sm120 GM, with a 3.1 s cached prepare (the first uncached compile was 129.5 s); its V16 partial path is timed inside prepare. V12 5/5 correctness and 6/6 graphs pass, 1.275x sm120 GM; its M128 3-stage probe was 0.5659 ms vs V9 two-stage 0.5643 ms locally, so the selected plan stayed two-stage. H200
claims currently come only from receipts `f6028a478d594b31`, `318227bc382d465c`,
`45a82e56c24f49f7`, `f95bdf5e394e474e`, and `eefecf1eeb724feb`.
