# Held-out S15 release and library confirmation

Base S15 `60254883`. Candidate C `2f93745e`; `C.patch` SHA256
`71b83bcc2a21c9781c8435090f65393299f9d2a9996c41032e5119de79623da7`.
This is a nomination, not a promoted candidate.

Reset PDL receipt `f88fda4aac144374` passes native parity/routing gates. Full E
new/old times at d1/d16/d32/v1/v16/v32 are
1.018308/.988901/.991881/1.001564/.977655/.992563; A:A
1.009834/.997998/.998476/.997318/.998725/.999697. Its no-glue executable
is identical to E and agrees with the useful d16/d32/v16/v32 signals.
The d1 control has upward bias; it does not support a d1 gain claim.

Library receipt `f22020d25d234be6` passes all768 native TASK gates. Full L
ratios are .999802/.999176/1.001312/1.006730/.994152/1.005131; A:A
1.013075/.998606/.999222/.998548/.999198/.999509. L/no-O is .995180
at v16; L/no-pool128 is .998126. V1 full and v32 sliding native attention
replacements lose1.007972/1.006421, unpool128 loses1.003428. O256 is unresolved.

The selected PDL pieces are q1 attention consumer enable+wait, large-M N_full
producer release/waits, and the independent VDSK qkv release. The small-M and
generic V9 release edits are excluded. The library pieces are O128 and pool128
only. No changed arithmetic in the PDL pieces; all other scored dispatch stays
S15. The new combined VDSK has one entry release, then a CHAIN-only wait before
its first activation/TMA read. Both CHAIN specializations are compiled for Hopper
without launching; the two changed CUDA bodies are identical to the measured E.

`../build_confirm.py` generates fifteen arms: A,A2,P, attention-only, large-only,
VDSK-only, P minus each of those three, O128-only, pool128-only, O+pool, C=P+O+pool,
C without O and C without pool. All12 shapes/six loads, eight held-out seeds,
eight cold rotating graph replays;1,440 actual TASK checks. Per-seed weighted
load times are retained at .001us precision, sufficient for the six-load
geomean R and its held-out confidence interval. No per-load win requirement.

Decision: every affected shape point below1 is nominated. A preselected composition
requires its held-out six-load R lower confidence bound above1 and TASK checks
for promotion; A:A is reported as a diagnostic, without per-load veto. The
frozen reset summaries omitted seed covariance and cannot establish that bound.

Local admission and queue identity are recorded after completion. An initial
forced wrapper preparation and one official normal candidate fixture hit GPU0
capacity while concurrent local screens held the memory. No kernel/checker was
changed: rerun the identical screens with fewer owned concurrent allocations.
See local_resource_failure.json and candidate_resource_failure.json.

Control qualification after R-20261026-1: d1 has directional A:A drift in both reset audits. Retain its point estimates with that uncertainty; this is not a per-load nomination veto. P4 confirms the paired held-out six-load R. The frozen summaries cannot supply its confidence bound.

Final admission: C passes143/143 scored/nearby/official fixtures eagerly and after two graph replays in both modes,25/25 full-source static checks each. Exact wrapper7127261f passes1,440 TASK gates,13 official host tests and2 static checks per mode. Native combined VDSK has one release before all loads, a dependent-only wait,94/90 registers at qkv/pool and zero local memory. **Queued sender760641**, decode check priority5.5, not-before1792972800. `admission.json`, `native_compile.json`, `queue.json` retain the evidence.

All seven handoff patches pass git apply --check at repo root and round-trip from exact S15 to the recorded candidate hash in isolated temporary trees. Headers use the real S15 repository path; source/math and queued payload bytes are unchanged. `patch_validation.json` records hashes.

## Held-out native result

Job **60bad72bcf964453** passes all1,440TASK checks,8held-out seeds×8cold
rotated replays,57.1s lab plus280.1s total preparation. No errors/failures.
A:A six-load R=.998390,95%CI[.996468,1.000315]. Per-load A:A time ratios
d1/d16/d32/v1/v16/v32 are1.007519/1.001212/1.002077/.998741/1.000699/.999572;
d32 has a small directional seed range, retained as uncertainty.

| Arm | Six-load R |95%CI| G0 against S15 |
|---|---:|---|---|
|C|1.006161|[1.004298,1.008028]|pass|
|C_no_o|1.005248|[1.003740,1.006759]|pass|
|C_no_pool|1.004273|[1.002476,1.006074]|pass|
|P|1.004848|[1.003157,1.006541]|pass|
|no_vdsk|1.004958|[1.003487,1.006432]|pass|
|no_large|1.002892|[1.001349,1.004438]|pass|
|no_attention|1.001725|[1.000143,1.003310]|pass|
|attention|1.001721|[0.999537,1.003911]|not cleared|
|large|0.999270|[0.997186,1.001358]|not cleared|
|vdsk|0.999469|[0.998596,1.000343]|not cleared|
|O|0.999525|[0.997991,1.001062]|not cleared|
|pool|0.999399|[0.998044,1.000755]|not cleared|
|OP|0.996698|[0.994523,0.998878]|not cleared|

C is the preselected complete composition, not a post hoc shape combination: early
q1 attention, large-M MoE and VDSK release plus native O128/pool128. Hand P4
`C.patch` with source/patch hashes and exact8seed R samples in `P4_HANDOFF.json`;
C now passes G0 against S15. P4 confirms any composition with newer winners.
Single-piece noisy aggregates do not veto their affected shape point gains.
No new confirmation run is needed for the already tested exact C versus S15.
