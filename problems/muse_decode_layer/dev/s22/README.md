# S22 integration

native G0/per-cell CIs pending; all hosts quota0 at send, awaiting reset; phase0, source4b8c803d, shipmentd1ccdc35; dropped=[], disabled hybrids=[]; sender3725147, payload403ff22fe051272d; finite follow-through worker3807459 selects resolved losers, confirms on new64seeds, scores only after passing G0.

Base is accepted S21 `eacc087b` / expanded `109c0135`; its scored sender3225144 is unchanged.
Exact12 gate at d1S, native-layout T128 banks, C3 full d16/d32 and v32F Rw4 remain.

| Load | Sliding attention | Full attention |
|---|---|---|
| d1 | S19 | S19 |
| d16 | S19 | C3 |
| d32 | direct Lee | C3 |
| v1 | direct OP | Lee |
| v16 | Lee | Lee |
| v32 | Lee | Rw4 |

The licensed hybrid patch is `9b043487`, receipt `cde88b9136de4093`; embedded MIT/Apache/NOTICE text and measured sources are preserved. The original d16S port is removed.

Each remaining ported cell has one whole-layer CUDA-graph arm replacing only that cell with exact S19 attention. Both direct-input hybrids also have an arm restoring the previous S21 port, so the d32S nomination whose CI crosses1 is isolated. All active arms get TASK checks on64freshseeds and16symmetric all-round timings; A:A and before/after-block clocks, temperature, power and enqueue traces remain. Per-cell R is time(leave-out)/time(S22): >1 favors retaining the port. `P5B_FEEDBACK.json` supplies the per-cell evidence after the receipt.

The prospectively declared drop rule is an upper95% CI below1. A selected composition uses new disjoint64seeds and new weights/router seeds before scoring. The finite worker stops on a failed/unmeasured final G0 or any execution error; it never silently scores incomplete cell evidence. Primary G0 remains the unchanged held-out six-load lower-bound gate through `tools/promotion.py`; prepare must fit300s. No new freeze precedes official TASK results.

Checks: **2024 contracts**, normal/forced **13 TASK + 12 graphs each**; exact packed **2816 TASK + 260 nearby**. Wire **11461 bytes**, full4000-character suffix and exact archive round trip pass.

Payload SHA256 `403ff22fe051272ddc3b2e154de6b7529998399a4ab284ae62cb47b1264caf9a`; wrapper `f923c406edb9ce95d7578ddab0486481aead7d00753ae904278f6f35883ef1c3`.

Current queue/state: [queue.json](queue.json), [state.json](state.json). Worker 3807459 waits on this one sender completion event. Each phase is archived under `stages/pN/`; canonical receipts remain in `runs/`.
