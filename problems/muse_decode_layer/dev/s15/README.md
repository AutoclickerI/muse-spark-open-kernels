# S15 incumbent

Frozen source `602548834f7481747701469b2a85154dde214bea28076c10fefce2c46b5240cb`:
[`cand_r3_best7_s15_final.py`](../../cand_r3_best7_s15_final.py). S14 is retired; S15c is dropped.
S15 combines the selected NF projections, last-CTA decode-1 sliding and N128 register-fold full verify.

The scored receipt `9c5fc18d302c4d40` is 1.800345x versus S14's `9b94c126d450448a` at 1.742759x.
Paired receipt `281b1383359c4fe2` gives R of load means 1.025676. It retained only summary statistics,
so an exact paired-seed CI cannot be reconstructed. The approximate interval in
`281b1383_report.json` is descriptive. Its old host-equivalence failures are not TASK failures.

`report.py` audits that receipt and uses the shared score gate for newly built confirmations.
`ab_build.py`/`ab_lab.py.inc` now embed the unchanged public TASK checker on every arm/seed and retain
weighted seed-level load ratios. Host-equivalence comparisons are diagnostics. The incumbent does not
wait on a new admission gate; this is Jihoon's explicit P4 decision.

`build.py`, `proj/`, `last_decode1.patch` and `regmax_full_verify.patch` preserve source provenance.
The captured `s15ab_payload.py`, `ab_manifest.json` and `queue.json` identify the historical receipt;
rebuilding a lab does not alter an already-sent snapshot. `local_standins.py` exercises Hopper glue
on the local architecture. New integration, selection and held-out confirmation: [S16](../s16/README.md).
