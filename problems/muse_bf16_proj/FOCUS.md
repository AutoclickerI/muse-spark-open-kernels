# Projection lane focus (P7 / G1)
Serves G1/G2: the overall decode-layer board score.

P7 bulk-ring batch: `dev/bulk_ring/`, real proj-host BF16/FP32 weights.
Held send ef2048bf49b3fdf4, PID4164652; gate09:02KST, behind MoE09:01.
16 role/M pairs,21 configs,4 O/boundary/pool segments, unchanged O/config6.
Cold256MiB, AB/BA, three input banks, producer+consumer AA, segment AA.
Pool+router M32 is the largest3.6TB/s gap:12.72us/layer,788.5us x62.
QKV leads at M1/M8; pool+router at M16/M32. FINDINGS owns all16 rows.
Native persistent equal-work grid,1/2CTA perSM,4/8-stage bulk-copy ring.
Lossless row-swizzled BF16 tiles; explicit splits1/2/4/8; fused consumer reduction.
Hopper residency capped by smem; each CTA differs by at most one K tile.
756 checks/252 graphs,9 host tests,33 spill-free sm90a kernels; forced smem stand-ins.
Local O/OP routes flip at M8/16/32; pool-only exact. H200 timing unmeasured.
`dev/bulk_ring/manifest.json` binds source; README owns reproduction.

95ab383d:15/16 unsplit register-MMA finalists lose; mainloops dominate.
Native entry-first47-101cycles, PDLwait10-11; fewer CTAs alone is insufficient.
M1 O/config6: producer+consumer15.39->15.04us, segment41.95->41.29us.
6 live full/sliding cases pass; exact routes; max hidden drift0.1833%.
Existing cell `dev/gemv_fixed/projection.py`, Best7ProjectionCell;
SHA256 c42e9fcc02a6ce57652507e763e1c5dbd67fc459b32a14684f379ac2ffe9d9fe
Decode handoff `dev/gemv_fixed/handoff.json`; whole-layer H200 gain unmeasured.

P10 compression4233660b granted; zero/pattern BW2.03x/1.70x, real weights~1x.
No compression drop-in. `dev/memory_compression/h200_result.json` owns evidence.
Oracle09d19813 caps later-stage prefetch~2-4us; chain87dc944b loses38.3->57.7us.
PDL7426e17c measured one M1 case, noise; remaining cases failed0.3% gate.
