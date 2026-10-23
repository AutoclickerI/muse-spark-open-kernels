# MoE checks for 09:00 KST (20-check round)

No remote job is launched from this lane. The orchestrator owns the queue and sends only
locally screened files. `cand_v8.py` and `cand_v9.py` are frozen/sent; this round preserves
both. Frozen hashes: V8 `7ab6818fcbb8f28b53686ed90362e4950fe4a642891836dfab9bcabb56c390be`,
V9 `a933dac686bbb25f0acc5f4675dd277b3a1362d041647e30c8263ba8845f89f5`, V11
`c01f691e4fd283a1a58a9ae7bc577838177cd76c85ee04fc12efb9104052a18a`.

| Order | Source | Purpose |
|---:|---|---|
| 1 | `cand_v11.py` | V9 for M=8/16/32/128; Humming wide for M=256 and serving M>=753; one canonical bank, prepare-time shape dispatch. |
| 2 | `cand_v9.py` | Frozen scored V9 control; use receipt to confirm V11 native choices and nearby-shape behavior. |
| 3 | `cand_hybrid.py` | Frozen H200 hybrid control (V9 native through M192, Humming above); compare against V11's single-bank dispatch. |
| 4 | `cand_humming6k.py` | Frozen Humming wide control; retain only if V11's wide glue regresses. |
| 5 | `cand_v8.py` | Frozen V8 control; no edits and no promotion ahead of V9. |

Local evidence: V11 5/5 correctness and 6/6 graphs pass, 1.262x sm120 GM. H200
claims currently come only from receipts `f6028a478d594b31`, `318227bc382d465c`,
`45a82e56c24f49f7`, `f95bdf5e394e474e`, and `eefecf1eeb724feb`.
