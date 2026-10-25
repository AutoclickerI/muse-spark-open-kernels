# S16 second-round integration

Base: exact S15 `60254883`. Reset queue is unchanged. S16 is an experimental next-round comparison;
no S16 source is promoted from a component/stage result. The S15 freeze gate in `dev/s15/report.py`
is mandatory and unchanged. If it fails, `decide.py` stops without emitting S16 or changing S14.

`python3 problems/muse_decode_layer/dev/s16/respond.py` reads landed reset receipts, applies the existing
S15 gate, emits S15 final and selected S16, runs serial combined normal/forced screens and full paired
wrapper screens, then queues **one check at priority -0.9**. No missing receipt or failed batch is treated
as a win. Independent completed batches may proceed; every unavailable/invalid result is recorded in
selection.json. The response records wall time in response.json; target is under30min after receipts.

## Reset decisions

| Waiting sender / source SHA | Prepared implementation | One-line decision rule |
|---|---|---|
| 3526702 /49ce378b (S15 LOO) | Existing exact S15 source and gate | Gate passes -> freeze exact60254883 and admit it as S16 base; otherwise stop and retain S14. |
| 2411468 /0ccc2548 (global attention) | global_C64/C128/W4/W8 | At each decode shape choose a complete16-seed win beyond A:A against OLD (S15 last sliding/direct-partials full), competing with early-history. |
| 3038721 /f1c270e6 (DSM verify sliding) | dsm_1/2/4/8/16 and wave policy | Choose the best native-gated8-seed gang at each sliding verify shape beyond A:A; require S16 whole-layer pairing against actual S15. |
| 3549805 /0cec6706 (early history) | early_E4/E2/E1/L4 | Choose a complete16-seed win vs OLD4 beyond A:A at decode1 sliding; it competes with the global recipe. |
| 3171016 /03086413 (softmax) | soft_mix25/mix50/gap/mixgap/tc25/ping/ping25 | Nominate only a three-seed native-gated range wholly below1-A:A; compare its N64/stage6 path with S15 N128/stage4 in S16. |
| 3360739 /917e79f3 (small MoE) | epi_VEC/PROD/NORM/PAIR | At each M choose a complete16-seed gain with upper95% bound below1-A:A; keep both sliding/full scope for a token policy. |
| 3408321 /f867a8b7 (heavy MoE) | heavy_DIRECT_CLUSTER1/NORMAL/SERIAL/DEEP_SERIAL | At each M choose a48-fixture TASK-passing arm below A:A, with every paired ratio below1; require full S16 pairing. |
| 3517006 /d29911c5 (PDL+boundary) | pdl_B/M/D; boundary_R/P/RP/C/S | Nominate a policy only if every affected measured row clears A:A; choose one global preload/boundary policy, then pair it in S16. |

These are experiment-selection rules, not promotion. S16's `report.py` requires official correctness,
all12 shapes x16 fresh seeds, every retained piece/load beating the full A:A envelope, all untouched
shape/load controls within A:A, and S15-back wins before writing an S16 final.

The8 existing reset senders are recorded byte-for-byte in reset_inventory.json. Their reset gates are
epoch1792972800 (00:00UTC /09:00KST), or1792972860 (00:01UTC /09:01KST) for the already-held P5/P3b runs.
No new H200 sender/job is created during this preparation. Every S16 payload depends on the S15 reset gate.

## Working set and verification

- `capture.py`: reconstructs production-capable cells from the exact queued wrappers, verifies their SHA,
  and emits cells.json. No dependency on an uncommitted builder edit or a removed historical path.
- `patches.py`:37 executable arm patches rebased onto S15. Only selected cells are emitted. Fixed-work
  ONCE, ready-input, CUPTI/occupancy and phase-clock controls remain diagnostics; they cannot be shipped.
- `screen_all.py`: serial GPU0 normal/forced local_eval for each implementation; admission.json pins
  exact source hashes and13 TASK/12 graph/native-static check counts. Logs and per-run outputs are local.
- `screen.py`: explicit Hopper launch stand-ins, branch call counts, recursive prepare tensor inventory.
  Actual early-history and small/heavy MoE arithmetic run where the local architecture supports them.
- `static_cache.py`: caches successful byte-identical CUDA/flags compiles in this exact judge image;
  Python name/signature checks and runtime tests always run. Changed CUDA compiles with concurrency2.
- `audit_native_hint.py`: independent sm90a compilation of the two dynamically generated NF hint producers.
  Both pass; only immutable weight lines are prefetched before the PDL wait, inside the timed call.
- `check_decision.py`: synthetic receipt-schema, neutral/incomplete rejection and policy-scope checks.
  `decision_validation.json` is explicitly synthetic and makes no H200 timing claim.
- `build_lab.py` / `lab.py.inc`: exact selected host plus each piece off, S15 back, A:A,16 fresh seeds,
  all12 shapes,16 rounds at T<=8 /8 elsewhere, full1GiB read flush. Compact archive preserves all controls.
- `respond.py`: select/build/screen/queue in one command after receipt collection. `--screen-only` omits
  the final send for a rehearsal. It never edits or requeues the existing reset batch.
- `pack.py`: lossless single-file shipment with a plaintext MIT notice. Expanded candidate is separately
  audited. `report.py` writes a frozen packed source only after the whole-layer promotion gate passes.

Rebase details that matter: S15's NF unpooler supersedes P7's balanced unpooler. P/RP therefore include
real native first-weight-line hints for its cp.async and RS-WGMMA producers; this adaptation is unmeasured
until S16. The old Triton hint is not counted as an active S15 change. New expert banks/counters install
before the incumbent's prepare-time FP32 gates; no new path falls through to old math during admission.
Prepared MoE data remain packed E2M1/E4M3; new NT banks only duplicate lossless packed layouts. Native
attention changes preserve all keys, KV/activation formats and cache stores.

Current screen progress is in admission.json and screen_all.log. Initial local orchestration/static
failures were repaired before any send; failed attempts remain in the local screen records. No reset
result exists yet and no S16 speedup is claimed. Next: finish all arm screens and compound-wrapper rehearsal,
then run respond.py when the reset receipts land.
