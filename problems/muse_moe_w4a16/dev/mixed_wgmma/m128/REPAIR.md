# K64 register-A lifetime repair

H200 status: confirmed numerically in combined receipt97be11d4631b44d6; every
cfg3/cfg5 grid and cfg4 control passes12 seeds plus R0/HOT128, maxL2/row
0.00413/0.00471 and no non-finite active ACT/Y/INV/OUT. The old finalizer is
slow; current work replaces it in `one_read/` (judge640d67e7e56d42e5).

MoE lane owns cfg3/cfg5's non-finite gates from H20066133279; free lane owns the
slow W2/finalizer. The shared fix is `../fused.cu`, with its exact diff in
`k64_bank_rotation.patch`. The split-K and W2 epilogue suffix is byte-identical to
the measured source (`repair_scope.json`). The M128 generator follows the shared
register index. No decode candidate was promoted.

K64 has one WGMMA group per stage. The old loop converted af[0] again after wait1
left that group's A fragments live. Native SASS used registers88–103 for every
stage; this is undefined under the [PTX wait-group rule](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-wait-group).
It matches the cfg3/cfg5 failures, and the combined H200 receipt confirms that
the repair removes their non-finite gates. The immediate local MMA reader cannot
reproduce asynchronous undefined behavior.

The fix unrolls pairs of K64 stages and alternates af[0]/af[1], keeping the previous
bank live through the following wait1. K128/K256 keep their within-stage rotation.
`check_async.py` reads actual old/new native SASS: all10 K64 width/phase paths
change from63 premature A-register overwrites per64-stage loop to zero. The new
paths use disjoint16-register banks, two wait1 sites and a final wait0. All393,216
canonical Hopper B128 addresses match the writer and stay within the staged tile.

After the fix,234 refreshed M128 graph checks and210 shared-producer comparisons
pass independent FP32 gates. Max M128 L2=0.004132, worst row=0.004948. All12 M128 and
16 shared native specializations have zero spills/C7513; existing shared K128/256
kernels retain91–112 registers. The actual pointer-only lab passes seven cfg/grid
choices on all12 public router weight seeds, R0 and HOT128 locally. Poisoned scratch
has zero non-finite entries in active ACT/Y/INV and OUT after the producers finish.
These checks use SM120 immediate MMA with Hopper-address emulation; H200 numeric repair now passes all12 seeds plus R0/HOT128 in97be11d4631b44d6;
performance of the serialized combine is rejected.

The lab uses the free lane's `../native_abi.py` to omit PyTorch headers. Its actual
sm90 extension builds/loads locally in12.5s, without running Hopper math. The full
local lab completes in7.2s with warm caches; its first pointer build took6.5s. The
three-way old/rotated/wait0 header prototype cost77.4s to build and was not sent.
The final payload checks for40s before native compilation and checks deadlines
before every seed. It retains all12 seeds, R0 and HOT128 within a78s local share.

Original MoE-host send key `0ea82c48f2255889`, priority0.8, was carried in the
completed combined receipt `97be11d4631b44d6` with the M256 W2 lab. Payload:
`prof/lab/moe_m128_k64_repair.py`, SHA256
`0927bfa53e38d7b7a8172f8a188dc7004854612c6d2ca6fd977fbbc9ef86cd15`.
Current frozen hashes and checks: `repair_admission.json`.
