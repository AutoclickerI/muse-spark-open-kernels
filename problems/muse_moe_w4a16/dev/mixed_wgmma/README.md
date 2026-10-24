# Packed WGMMA MoE producer

Free lane owns M256 inside the decode layer. Serving is parked. Owner sends layer candidates
and commits; this lane sends labs. Current measured results and causes live in
[docs/FINDINGS.md](../../../../docs/FINDINGS.md#grouped-nvfp4-rs-wgmma-free-lane).

| H200 fixed M256 path | Seed mean, us | Receipt |
|---|---:|---|
| Raw CTA1g2112 |385.2|97be11d4631b44d6|
| Lossless recoded C1g2112 |365.2|9e8a6a15f95748db|
| Masked Humming, same respective runs |447.6 / 448.2|same|

Raw is integrated in decode S2 (`cand_r3_best7_scale_w2fix.py`, SHA256
`b080544708544fd67ee23b82aca111bf0b9bb82c30a5eb86315044cd9bf92337`);
recoded in S3 (`cand_r3_best7_scale_recoded_m256.py`, SHA256
`cadef19a6eced1ae1fa0f3deaffa9ee21e083aa5ba6fe46784707bb8df52a7c7`).
Whole-layer H200 results remain pending. Exact sources/screens are in decode `dev/s2/` and `dev/s3/`.

`w2fix.{cu,py}` is the shared producer: timed route, W13+SwiGLU, W2+weighted Y/tile sums,
then a vector CTA per token for norm/combine. This replaces the serialized last-warp tail;
W2/finalizer have no completion atomics or publication fences. Raw and lossless recoded
weight/scale formats stay packed until in-kernel conversion; FP32 global scales stay in epilogues.

The old vector mismatch was a B128 base-phase error, fixed by1024-byte alignment and absolute
Hopper-address emulation. Native register-A lifetime audits retain wait1 overlap without spills.
`native_abi.py` supplies lean pointer-only bindings; plans retain scratch owners and obtain
current streams at launch. `check_tail.py`, `check_w2fix.py` and `dump_w2fix.py` reproduce these checks.

Current work: [w13_issue/README.md](w13_issue/README.md), instruction/scale-copy cuts and
N16-cold/N32-hot scheduling at fixed K128/S4/grid2112. New MoE-host lab8483c0c6783a4df1
(send22da326e27940230) is accepted; performance pending. No new cell is promoted.
Lossless scale format/provenance: [recode/README.md](recode/README.md).
Older CUTLASS numeric GEMM stays quarantined and out of candidates.

## M128 K64 non-finite gates: shared register-bank repair

MoE lane owns this repair; free lane owns the W2/finalizer work above. The shared
`fused.cu` now alternates A-register banks across paired K64 stages. Its split-K/W2
epilogues are byte-identical to the old measured source; K128/K256 scheduling is
unchanged. `m128/k64_bank_rotation.patch` is the exact scoped diff, and the M128
source generator follows that shared register index.

H20066133279's cfg3/cfg5 converted into af[0] again after wait1 left its WGMMA group
live, violating the [PTX input-register lifetime requirement](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-wait-group).
Actual native SASS confirms the reuse. All10 K64 math paths go from63 premature
register overwrites per64-stage loop to zero; new banks are disjoint.393,216 Hopper
B128 addresses match the writer and remain inside the tile. The immediate local
MMA stand-in hid asynchronous lifetime errors. This is a confirmed PTX violation
matching the failed configurations; H200 removal of the non-finite gates is pending.

After the fix:234 M128+210 shared FP32 checks pass; native12+16 specializations have
zero spills/C7513. The pointer-only repair payload (reusing `native_abi.py`) passes
seven cfg/grid choices on12 public-router seeds plus R0/HOT128, with poisoned
ACT/Y/SS/INV fully written on active rows. Worst M128 L2<=0.004132,row<=0.004948.
Its actual sm90 extension builds/loads locally in12.5s, without native math, and
checks its deadline before compilation/every seed. W2 timing and H200 numeric
confirmation remain open. No decode candidate was promoted.

Send key `0ea82c48f2255889` on MoE, priority0.8; immediate detached log says waiting
for a judge slot without rejection. Full ownership/cause/reproduction and frozen
hashes: `m128/REPAIR.md`, `m128/repair_admission.json`.

Current lossless scale-bank experiment: [recode/README.md](recode/README.md).
Paired raw/recoded WGMMA, NT1-recoded and masked Humming: job9e8a6a15f95748db,
send keyfdac893179c9f8ad; H200 winner unverified. Both joint encodings are lossless.
