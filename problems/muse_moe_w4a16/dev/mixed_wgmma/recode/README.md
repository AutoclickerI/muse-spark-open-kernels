# M256 lossless scale recode

H200 `9e8a6a15f95748db` completed: C1g2112 **365.2us mean /346.4 median**, against
same-run raw386.1, masked Humming448.2 and NT1-recoded500.4. All gates pass across12
public weight/router seeds,3fresh fixtures and8rotated/reversed cold128MiB graph timings.
No per-seed winner selection. The recode gains2.4–6.2% across grids.

The exact C1g2112 is integrated in `problems/muse_decode_layer/cand_r3_best7_scale_recoded_m256.py`,
SHA256 `cadef19a6eced1ae1fa0f3deaffa9ee21e083aa5ba6fe46784707bb8df52a7c7`.
Decode `dev/s3/` pins accepted source and complete-layer local agreement. Whole-layer
H200 composition and cold H200 prepare remain unmeasured; owner controls candidate sends.

`rc_repack` folds each E4M3 group's sign into packed E2M1 signs, clears FP4 codes for zero
scales, and stores a uint8 exponent/mantissa. Timed conversion constructs BF16 multipliers
with PRMT/shift/add. This is exact joint repacking; no expanded weight or scale bank is
prepared. FP32 expert global scales remain in the epilogues. `recoded.py` and `recode.cu`
retain N16/32,K128,S4,grid528/1056/2112 and the distributed W2 finalizer.

Local proof:65024 encoding products/fields and111 graph/extreme-scale checks pass,
maxL2/row0.004132/0.004714; independent gate/up scales, allocator churn, empty routes,
nonzero local expert offset and absolute Hopper addresses. All11 native SM90a entries
have zero local-memory operations; WGMMA retains wait1/wait1/finalwait0. Host5tests/6static
and payload3static pass. Fresh cache-disabled build3.5s/full lab15.6s within78s. Deadline
guards cover compilation, seeds, JIT/capture, repetitions and phases; partial n is explicit.

Regenerate with `build_native.py`, `build_lab.py`; verify via `check.py`, `dump.py`,
`tools/local_eval.py` in `tools/img`. Current proof/hashes: [admission.json](admission.json).
Payload `prof/lab/moe_recoded_m256.py`, SHA256
`d30f4a862fa0b882529ba3b9ea9c37013f305559c707e27db62e10b4ff1a2bce`;
send `fdac893179c9f8ad`. The new W13-only issue experiment is in `../w13_issue/`.
