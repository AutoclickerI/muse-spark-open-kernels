# Point-estimate nominations to P4

All patches apply onto exact S15 `60254883`. No per-load or A:A clearance is required for nomination; P4 composes the shape choices and applies the held-out six-load R lower-confidence gate. TASK.md is the only correctness criterion.

Whole guarded S15n9cc90d2e is nominated now, patch792eacc3 in `muse_decode_layer/dev/nearby/native_validation/s15n_guarded.patch`. Its scored dispatch is identical; nearby/scored speedups improve. Native correctness sender1096763 remains a pending proof, not a performance veto.

Changes from the earlier handoff: O256 is included; small-M release cells at d16S/d32S/d32F are included despite tiny gains; pure VDSK qkv remains nominated at v16 despite its misleading source-wide aggregate. No unpool128/v1full/v32slide cell has a point improvement, so those exact verdicts remain losses. Both reset softmax batches have no changed arm below1 at either target shape.

| Piece | Shape | On/off time | Full/off load SE | Seeds |
|---|---|---:|---|---:|
| attention | 16x1s | 0.992497 | 0.000723/0.001295 |8|
| moe | 16x1s | 0.999470 | 0.000723/0.000552 |8|
| attention | 16x1f | 0.992745 | 0.000723/0.001295 |8|
| attention | 32x1s | 0.989617 | 0.000757/0.000446 |8|
| moe | 32x1s | 0.999029 | 0.000757/0.001114 |8|
| attention | 32x1f | 0.998857 | 0.000757/0.000446 |8|
| moe | 32x1f | 0.998030 | 0.000757/0.001114 |8|
| moe | 16x8s | 0.986953 | 0.001127/0.001294 |8|
| projections | 16x8s | 0.990892 | 0.001127/0.001403 |8|
| moe | 16x8f | 0.991732 | 0.001127/0.001294 |8|
| projections | 16x8f | 0.982348 | 0.001127/0.001403 |8|
| moe | 32x8s | 0.992067 | 0.000394/0.000502 |8|
| moe | 32x8f | 0.995440 | 0.000394/0.000502 |8|
| o | 16x8s | 0.994591 | 0.000907/0.000475 |8|
| pool128 | 16x8s | 0.997922 | 0.000907/0.000514 |8|
| o | 16x8f | 0.996424 | 0.000907/0.000475 |8|
| pool128 | 16x8f | 0.998523 | 0.000907/0.000514 |8|
| o | 32x8s | 0.998844 | 0.002977/0.002994 |8|
| o | 32x8f | 0.998190 | 0.002977/0.002994 |8|

Per-shape SE and paired piece SE were not retained by the frozen receipts. The displayed SE belongs to the full/off load mean, not the cell ratio. `P4_HANDOFF.json` preserves that distinction, receipt IDs, exact patch hashes and candidate hashes. Do not infer a promotion confidence bound from these summaries.
