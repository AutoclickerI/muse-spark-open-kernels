# P3a small-M MoE

Current experiments use exact S15 `60254883`. Today's earlier measurements use S14 `f1c77ce6`, whose
small-M CUDA/math matches S15. Flags95 at M1 and69 at M8/16/32. No new whole-layer kernel is frozen here.

**Nomination:** hand P4 every valid per-shape point estimate below1, with ratio,SE,seeds and uncertainty.
A:A describes noise; it does not veto a shape. **Promotion:** P4 composes, then requires the held-out
per-seed six-load geomean R lower confidence bound above1. Correctness is TASK.md's checker only.
Ratios below are candidate/control time; smaller wins.

## Today's measured results and changed verdicts

`point_nominations.json` is the complete P4 handoff, including ratio,SE,seeds,affected shapes and exact
patches in `review/point_*.patch`. `review/rejudge.py` reproduces it from the audited receipts.

| Run | Result at M8 / 16 / 32 | Current decision |
|---|---|---|
| Epilogue `d650c04ead114119`,16seeds | PAIR/S14 .97493/.97262/.97932. PROD .98056/.97831/.98517; NORM .98142/.98340/.99000. | Hand all point winners to P4. Also M1 PAIR .99890 SE.00232 and NORM .99839 SE.00213 now nominate; their earlier A:A rejection is removed. EPBASE control-alias point wins are exported for deduplication. |
| Epilogue-next `29ad388de976432c`,32seeds | PAIR/S14 .97830/.97462/.97778; TREE .97888/.97404/.97746; LOCAL .99212/.98615/.98488. TREE/PAIR .99904/.99811/1.00007. | TREE's M8/M16 direct point wins now nominate; TREE and LOCAL versus S14 nominate at all3 M. LOCAL remains slower than PAIR, so it is an alternative. FOUR/FOUR_LOCAL remain slower. |
| Pipe `16c180784f414576`,16routed seeds | Deliberately wrong-math PDIA/S14 .90404/.89884/.82615. Independent-chain microprobe suggests1–2% headroom. | Diagnostic only; PDIA cannot nominate. Full correct rebuild measurement follows. |
| Rebuild `dd539517bc8344dd`,32seeds | ILP/S14 **.99055/.99041/.99046**, SE **.00181/.00132/.00143**. FLIGHT1.07662/1.05285/1.03119; K16 1.20394/1.21153/1.23270; LUT1.59959/1.62315/1.67379. | **ILP now nominates M8 as well as M16/M32.** Its earlier M8 uncertainty veto is removed. Other3 exact recipes remain losses, leaving other flight/occupancy/decoder methods open. |
| Direct PAIR `1b135b70835f4634`,32fresh seeds | ILP/PAIR **1.01094/1.02063/1.01190**, SE .00178/.00242/.00220. S14/PAIR1.02243/1.02739/1.02071. | PAIR is faster directly at every M. ILP remains an alternative versus S14; its composition with PAIR and every whole-layer gain remain unmeasured. |

All correct arms above pass their native TASK gates and official hosts, with complete seeds and output.
A:A mean absolute us, ordered M8/16/32: first epilogue .415/.235/.225; second .442/.363/.322;
pipe .443/.250/.336; rebuild .487/.316/.314; direct PAIR .488/.377/.364 (max1.424/.928/1.648).
First-run M1 A:A .243us. Errors stay below .00251 total/row versus .015/.075 MoE TASK bounds.
Old upper95%+A:A fields remain descriptive evidence in the JSONs, never nomination gates.
The original ONCE/VEC losses close only those recipes; [ISSUE.md](ISSUE.md) and [PIPE.md](PIPE.md)
retain their instruction/latency evidence. The old58.7us math clock belongs to ONCE, not S14.

## Exact P4 patches

- `point_nominations.json`: every eligible point winner, per-shape source,ratio,SE,seeds.
- PAIR: existing admitted S16 `epi_PAIR`, S15 candidate `af86073f`; S14 patch
  `epilogue_s14_pair.patch` `786a5ecb`. Other epilogue patches are in `review/point_*.patch`.
- ILP: `pipe_s14_ilp.patch`, full SHA256
  `b55c2970a7737c3708b0e5f251ab24b31ef1888bdae55f812c8e155d9eac230d`, onto S14 `f1c77ce6`;
  candidate `f71a0206`, cell `809bef8e`. `p4_ilp_handoff.json` now includes M8/16/32 sliding/full.
  Original M1 remains the ILP control path; no M1 ILP measurement was made.
- `patch_s14.py --kind <arm> -o <candidate.py>` reproduces earlier cells/patches/flags.
- No per-load confidence margin, strict routing-identity gate or component-equivalence threshold is added.
  P4's whole-layer TASK checks and held-out R confirmation decide the final composition.

## New review batch

[review/README.md](review/README.md), `review/claims.json` and `review/manifest.json` pin the actual S15
premises, independent candidates and exact patches:

| Item | Status |
|---|---|
| BLOCK: W2 waits on own `done[e,nb]` | Confirmed and built,512B done region; fence/publish/acquire ordering retained; M16/M32 only. |
| CANCEL: remove W2 global scale through RMSNorm | Confirmed and built; eps/a²,sign,zero and extreme handling. W13/SwiGLU untouched. |
| SHARD2 / SHARD4 / SHARD8 | Confirmed and built as3 separate changes to PAIR's norm slots; direct PAIR control retained. |
| TWO_BASE sign/exponent decoder | Exact finite identity confirmed, **dropped before H200**: integer instructions37→406,registers30→31 for this implementation. |
| PACK5 exact scale packets with raw fallback | Complete official layer seed5353 tensors measured first; final metadata-inclusive cut **4.058%**. Built and full-bank bit proof passes, plus actual raw-fallback MoE launches. |

The packed payload pairs all surviving arms and PAIR with S15 at16fresh seeds/M, M1/8/16/32, A:A and
native TASK gates. `review/lab_local_validation.json`: complete,6,399characters,52.4s locally.
All14 normal/forced exact-hash whole-layer screens pass13 tests,12 shapes and26 static checks each
(`review/screens.json`). Sent on MoE as sender**1191843**,wrapper**`a2344c10505b0e44`**,
priority1.2,import budget240s,full9000-character allocation (`review/queue.json`).
No H200 speedup is claimed for this batch before its receipt. Prefill is excluded for this task.

## Evidence

`read_reset.py`, `read_rebuild.py`, `review/read_pair.py` reproduce the earlier H200 audits.
`ilp_vs_pair_results.json` and its queue record pin the direct result; `followup_queue.json` pins
`dd539517bc8344dd`. Complete pipe probes decode with `read_pipe.py`.
All earlier local source/hash/normal/forced receipts remain in the stage's validation JSONs; GPU0 sm120
uses explicit stand-ins only for Hopper-only glue. Local timings are never H200 evidence.
