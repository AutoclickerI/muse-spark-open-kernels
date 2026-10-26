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

Decision: select the fastest supported arm/composition by the six-load geomean,
then require a held-out R lower confidence bound above1 and valid A:A/TASK gates
before promotion. Failed controls or unproven execution are unmeasured. The
frozen reset summaries omitted seed covariance and cannot establish that bound.

Local admission and queue identity are recorded after completion. An initial
forced wrapper preparation and one official normal candidate fixture hit GPU0
capacity while concurrent local screens held the memory. No kernel/checker was
changed: rerun the identical screens with fewer owned concurrent allocations.
See local_resource_failure.json and candidate_resource_failure.json.

Control qualification: d1 is **unmeasured** in both reset audit recipes because every A:A seed ratio exceeds1. Its point estimates do not establish a d1 gain or loss. The fresh confirmation retests d1; no whole-composition promotion is claimed from the frozen reset summaries.

Final admission: C passes143/143 scored/nearby/official fixtures eagerly and after two graph replays in both modes,25/25 full-source static checks each. Exact wrapper7127261f passes1,440 TASK gates,13 official host tests and2 static checks per mode. Native combined VDSK has one release before all loads, a dependent-only wait,94/90 registers at qkv/pool and zero local memory. **Queued sender760641**, decode check priority5.5, not-before1792972800. `admission.json`, `native_compile.json`, `queue.json` retain the evidence.

All seven handoff patches pass git apply --check at repo root and round-trip from exact S15 to the recorded candidate hash in isolated temporary trees. Headers use the real S15 repository path; source/math and queued payload bytes are unchanged. `patch_validation.json` records hashes.
