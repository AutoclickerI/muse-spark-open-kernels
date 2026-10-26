# P7 cross-stage stream continuity

The source base is frozen S17 `fea717603233` (`cand_r3_best7_s17_final.py`, shipped `219fac280951`).
The interrupted S15+C build was replaced; no result from that build is promoted.

[BOUND.md](BOUND.md) derives the per-load scenario ceiling from S15 receipt `7eaa262a30ec483e`.
[bound.json](bound.json) preserves every critical interval, byte model, owner and exact idle gap.
The trace does not contain DRAM counters: HBM-idle time is unidentifiable. One fitted2.36us ramp
per weight/KV boundary plus actual GPU idle gives R1.08557 (32.8% of the initial GOAL score snapshot
gap,1.83789 to2.3601). The one-store/read interstage materialization scenario adds R1.01602 at
4.58TB/s; this is not measured HBM miss savings and is not additive to the ramp ceiling. P11's
L2 issue and repeated activation/page reads remain open.

| Arm | Change from S17 |
|---|---|
| A / A2 | Identical source, functions, states and captured launch/resource/DAG fingerprints |
| E | Selected NORM/PAIR small-M producer releases dependents immediately after its last TMA weight load is issued |
| P | E plus4MiB unpooler-bank prefetch from the producer's final weight-load path |
| W | O1/O8 NF consumers issue both initial weight-pipeline stages before grid dependency wait |
| F | Separate PDL intermediary issues4MiB reads for O/pool/unpool, releases its dependent at entry and waits before exit |
| B | Same intermediary, L2 hints instead of reads |
| L | Same intermediary, bulk L2 prefetch with16MiB budget |
| X | Frozen preselected combination: P+W+B (small-M tail uses P rather than a separate prefetch node) |

Prefetch descriptors follow the actual consumer's NF/coded/LT/raw allocation, including S17's
`p10_pool_raw`; original BF16 weight banks are not used when the consumer reads compressed NF.
All prefetch and dependent work stays inside the timed call; there is no access-policy window.
The intermediary's final wait preserves the transitive dependency on its own producer.

The first SASS audit found ptxas had sunk O's intended loads back after the wait. W now puts the
existing NF order-point branch before the wait. Exact sm90a SASS has all6 initial128-bit weight
vectors before ACQBULK at both O1/O8; activation loads remain after it.9 changed native sources
compile without spills. See [native_validation.json](native_validation.json).

The batch freezes16 new seeds,8 rotated forward/reverse cold rounds,12 shapes and9 arms.
Each seed's load time is46*tS+16*tF. R is the six-load geomean of incumbent/candidate times;
the CI is Student t95 on log R across16 paired seeds (df15). No A:A correction or per-load veto.
Only the unchanged public TASK checker is used, with1728 arm/shape/seed checks. A:A is diagnostic.

Normal and forced sm120 screens each passed1728 TASK checks; the host passed13 TASK tests and
12 CUDA graph shapes in both modes. Hopper-only attention launches use the established S17
stand-ins; changed MoE and NF weight-load code also compiles for sm90a. Local timing is not an
H200 result. The final source adds the order-point scheduling change to the normal screen;
the forced screen covers identical Python dispatch/ABI without that final native instruction move.

H200 check `a405c8b463054e6c` uses the prefill host because decode has spent20/20 daily checks.
This is one whole-decode paired lab on the H200, using synthetic decode weights and the decode
TASK checker, as a check only. The host's prefill board speedup is unrelated to the paired decode R.
All arms and seeds travel in this one batch. [P4_HANDOFF.json](P4_HANDOFF.json) records the verdict
and exact source hashes; [result.json](result.json) contains the paired decode verdict.

Reproduce:

```bash
python3 problems/muse_bf16_proj/dev/stream_continuity/bound.py
python3 problems/muse_bf16_proj/dev/stream_continuity/build_lab.py
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_bf16_proj/dev/stream_continuity/audit_native.py
python3 problems/muse_bf16_proj/dev/stream_continuity/report.py runs/muse_prefill_layer/20261026T054214Z_check_a405c8b463054e6c.json
```

`build.py` reconstructs every exact arm from the pinned packed S17 file. Generated sources,
CUDA objects, SASS and local logs stay ignored. The receipt preserves paired seed samples with
nanosecond quantization (round-trip checked within0.5ns); no six-load covariance is discarded.

The receipt clears G0 only for W: **R1.003215 [1.001964,1.004468]**,1728/1728 TASK pass.
A2 .999820 [.998789,1.000852]; E .994282 [.988018,1.000585]; P .989306 [.987589,.991027];
F .382012 [.380140,.383893]; B .914259 [.909898,.918640]; L .615109 [.613363,.616860];
X .914508 [.912608,.916412]. E is inconclusive; the other changed recipes lose.
The full W arm is promoted on S17 even though d1F/v1 pieces lose; no per-load veto applies.

[W_integration.py.inc](W_integration.py.inc) and [W.patch](W.patch) reproduce only the measured
O1/O8 function swaps after unmodified base preparation. Appending generic W_CODE by itself
would also change pool/router preparation and is not the measured recipe. The packed
[W_candidate.py](W_candidate.py) passes13 TASK tests,12 graph shapes and2 static checks in each
normal/forced production screen ([winner_validation.json](winner_validation.json)). Its timed
base Python/CUDA code is unchanged; only prepared NF O function pointers differ. P4 must confirm
its current-S18 composition with held-out six-load G0. The d1S-only W alternative and E d1S/d32S/F
point winners are selection nominations; numerical wins at unchanged shapes are not recipes.

`python3 problems/muse_bf16_proj/dev/stream_continuity/build_winner.py` regenerates the production
candidate and patch. Exact native sources, arm hashes, seeds, CI method and receipt source hashes
are preserved in the JSON manifests and [result.json](result.json).
