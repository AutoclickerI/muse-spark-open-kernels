# S20 integration

S20 expanded source is `9e6f525f`, shipment `f9bd44d4`. It is built on the exact S19 expansion `62ea5bf4` (shipment `8b415553`). S19's scored sender,
pid 4046452, is unchanged. Neither candidate is promoted or newly frozen before S19's official TASK result.

| Piece | Rows | Nomination and composition |
|---|---|---|
| P5b C3 | d16F, d32F | Fixed-ring source `519ea55f`; fixed receipt `d53966e4`, repeated against S19 in `deea291e`. Replace R8 in its existing compile. Latest CIs cross 1; `no_C3` measures the leave-out. |
| P10 exact12 fused gate | d1S | `e3643817`: R 1.004405 [1.003003, 1.005808] over S19, +9.192 s prepare. Exact measured patch; bank construction moves into the late-bound delta, before warm. |
| P5b FA3 M64 | v1F | New `deea291e` nominee: stage time ratio .909787 [.898514, .921060]. Exact split-64 metadata, qk_prep and norm/gate path; no extra native compile. |
| P3a PRIVATE | M1 (d1S/F) | `6f5afe0c`: stage ratio .997059 [.994298, .999820]. Exact PRIVATE change to S19's FAST+INDEX cell. Identical-control alias timing bias is recorded by the lane; do not read this as a proven whole-layer gain. |
| P3a AFFINE | M32 (d32S/F) | `6f5afe0c`: .997026 [.994998, .999054]. Separate M32 cell; retain S19 M16. |
| P7 H2 | v16S/F, v32S/F | `73d2d80d`: R 1.001504 [1.000124, 1.002887] over S18. Exact nominated P3B CUDA `dbef0a8e`; first W2 weight stage before its PDL wait. |
| P7 B1 | b1 at all loads | `73d2d80d`: R 1.001355 [1.000064, 1.002647] over S18. Exact nominated native glue `df9bf06e`; preserve the nominated b1 dispatch. S19's reverted `_boundary_late` needs an explicit b1-only port; its b2 path remains the S19 code. |

All seven pieces have leave-outs in the same import-time lab. Other weak point nominees whose CIs cross 1
(EXP, B4, E, older glue/tail choices) stay out. C4/T16 are alternatives to C3, not additive pieces. Earlier
pieces already selected or rejected by S19 stay as S19 chose them. Every lane's handoff and point file is hashed
in `handoff_review.json`. C3 CUDA and the P7 recipe inputs were snapshotted under `inputs/` so S20's composer
can be reproduced from this commit without committing another lane's files.

`compose.py` retains the S19 prepare definitions and delta ownership, adds banks through that delta, and moves
one final warm to the end. `contracts.py` extends S19's contracts: flags, binds, prepare chain, shared producers,
base plans, exact receipt picks and the S20 component/leave-out identities. It runs before GPU work.

Reproduce:

```
python3 problems/muse_decode_layer/dev/s20/compose.py
python3 problems/muse_decode_layer/dev/s20/contracts.py
python3 problems/muse_decode_layer/dev/s20/build_lab.py
python3 problems/muse_decode_layer/dev/s20/check_producer.py
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_decode_layer/dev/s20/screen.py problems/muse_decode_layer/cand_r3_best7_s20_final.py --out problems/muse_decode_layer/dev/s20/screen/normal
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_decode_layer/dev/s20/screen.py problems/muse_decode_layer/cand_r3_best7_s20_final.py --forced --out problems/muse_decode_layer/dev/s20/screen/forced
```

The lab uses 16 fresh input seeds `1729315067 + 104729*i`, new weights/router seeds `1729315001/1729315003`,
and records 960 scored-shape TASK checks plus 260 nearby checks. Restricted leave-outs take the independent A2
control's timing outside their changed rows. Native arm status, all paired seeds, checker/source identities and
A:A survive in the checksum-protected packet. Optional-arm failure stays attributed to that arm. The producer
fault-injection check verifies that primary evidence survives and official TASK is still required to promote.

`report.py` verifies the wrapper and invokes `tools/promotion.py.evaluate`. Its `lab_gate` permits queuing an
official scored run after native paired R's lower 95% bound exceeds 1 with complete primary TASK/nearby evidence;
`freeze_gate` also requires the matching official shipment's TASK pass. No S20 frozen file is written.

## Native verdict: do not queue

One prefill check, `0640cce4b12f42f9` ([receipt](../../../../runs/muse_prefill_layer/20261026T091016Z_check_0640cce4b12f42f9.json)),
completed the import experiment in **301.7 s**. Every arm is native and complete: **960/960 TASK + 260/260 nearby**,
no failures. Candidate prepare **157.650 s**, including GATE **9.050 s** and final warm **17.771 s**, below ~200 s.
The host's deliberate prepare stop is separate from this passing lab evidence.

`tools/promotion.py`: **R 1.002271 [0.996468, 1.008109]** over exact S19. **G0 fails**. No S20 scored run is queued;
S19's reset sender is unchanged, and no S20 frozen file exists. Official TASK remains unverified for S20.

| Leave-out | R = leave-out / S20 time, six-load geomean | 95% CI |
|---|---|---|
| no_C3 | 1.000968 | [1.000339, 1.001597] |
| no_GATE | 1.000289 | [0.999671, 1.000906] |
| no_M64 | 1.001167 | [1.000527, 1.001807] |
| no_PRIVATE1 | 0.999940 | [0.999382, 1.000498] |
| no_AFFINE32 | 1.000064 | [0.999385, 1.000744] |
| no_H2 | 1.000183 | [0.999655, 1.000712] |
| no_B1 | 0.997004 | [0.996359, 0.997649] |

Values above 1 favor retaining the piece. C3 and M64 are resolved gains in this composition; B1 is a resolved loss:
removing it is about **0.30% faster**. GATE, PRIVATE1, AFFINE32 and H2 remain unresolved. Per-piece diagnostics do
not replace the whole-composition gate.

The overall interval is dominated by one verify16 pair: final seed `1730886002` has INC/S20 **1.290288**, while
that load's A2/S20 is **1.001889**, no_H2/S20 **1.001387**, and no_B1/S20 **0.999722**. The aggregate shape summaries
locate the change in v16S. We retain this seed and the full CI. The packet lacks per-replay clocks or stage traces,
so timing interference versus a data-dependent runtime effect is **unverified**; no outlier is discarded.
A:A also shows small systematic drift (d1 1.003021, v1 1.001075; v32 .999068); it is reported, not subtracted.

Exploratory S20-no_B1/S19 from the same samples is R **1.005283 [0.999432, 1.011168]**, still not a G0 pass.
A leaner next composition should omit B1 and reconfirm on new disjoint seeds; choosing a leave-out using these
confirmation samples cannot promote it. No second native lab was sent.

The final shipment strips blank-comment whitespace from screened `f78f6ac8`; Python AST and the expanded source
are identical. Final packaging preflight passes; `validation.json` retains both identities.
