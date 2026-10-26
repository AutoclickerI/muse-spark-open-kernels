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

## S15 review: measured, then composed

Original `1da2153a0f5543b9` is **unmeasured**: its host passed, but the wrapper dropped GATE and
its timing archive. This closes no recipe. The receipt codec is now bounded below9,000characters
independent of timing entropy; all seed ratios and independent A:A pairs survive.

Retry check **`b57d691d689046a8`**,sender1365722,wrapper`3457baa10244c861`, completes16seeds/M,
all official/native TASK gates pass,worst total/row .002401/.002545. `review/results.json` retains
ratio,SE,seed indices and A:A; `point_nominations.json` has every point winner and exact S15 patch.
Candidate/control ratios below; M1/8/16/32 order.

| Recipe | /S15 | Nomination / closure |
|---|---|---|
| BLOCK | 0.99371 / 1.00218 / 0.99237 / 0.99406 | M16/M32 nominate. M1/M8 execute the identical S15 control; M1 noise is marked as an alias. |
| CANCEL | 1.00167 / 1.02266 / 1.01870 / 1.01407 | No point win; close this eps/metadata/finalizer implementation, leaving a cheaper identity realization open. |
| PAIR | 1.00192 / 0.98090 / 0.97231 / 0.97999 | M8/M16/M32 nominate; earlier positive M1 evidence remains an alternative. |
| SHARD2 | 0.99646 / 0.98603 / 0.97600 / 0.98517 | All4 M nominate versus S15; M1 versus PAIR .99179. M8+ loses directly to PAIR. |
| SHARD4 | 1.00410 / 0.98385 / 0.97798 / 0.98561 | M8/M16/M32 versus S15 and M1 versus PAIR .99911 nominate. M8+ loses directly to PAIR. |
| SHARD8 | 0.99876 / 0.98759 / 0.98019 / 0.98910 | All4 M versus S15 and M1 versus PAIR .99646 nominate. M8+ loses directly to PAIR. |
| PACK5 | 1.28466 / 1.39238 / 1.40819 / 1.45243 | Net4.058% byte cut loses after timed decoding. Close this exact packet/decoder; other encodings remain open. |

A:A mean absolute us **.200/.541/.409/.391**,max.736/1.280/1.312/.896; descriptive,not a veto.
Two-base assembly was dropped before H200:37→406 integer instructions,30→31registers; other packed
assembly remains open. [review/README.md](review/README.md) retains all code-premise and exact-format proofs.

The next single composition arm uses **SHARD2 M1, PAIR M8, PAIR+BLOCK M16/M32**. CANCEL and PACK5
are excluded by their measured losses. It compares with S15,PAIR and per-shape best independent on
16 fresh seeds (weights275353+7919*s,routes279000+M+97*s),with A:A and TASK before/after.
`review/combo/manifest.json` and `combo.patch` pin the exact submitted candidate. No additive gain
or whole-layer promotion is claimed. P4's held-out six-load R confirmation remains the only promotion gate.

## Evidence

`read_reset.py`, `read_rebuild.py`, `review/read_pair.py` reproduce the earlier H200 audits.
`ilp_vs_pair_results.json` and its queue record pin the direct result; `followup_queue.json` pins
`dd539517bc8344dd`. Complete pipe probes decode with `read_pipe.py`.
All earlier local source/hash/normal/forced receipts remain in the stage's validation JSONs; GPU0 sm120
uses explicit stand-ins only for Hopper-only glue. Local timings are never H200 evidence.


Follow-up accepted as MoE check**`c49dbdffe92d491a`**: sender1548101,wrapper`da4bfcfab9f0b0db`; `review/combo/queue.json`.
Normal/forced each pass13TASK/12shapes/28static;270 component/replay checks; exact packed16seed/64AA lab passes4371characters. H200 composition receipt c49dbdffe92d491a is complete:all16seeds/TASK gates pass.

## c49 composition and next instruction cuts

`c49dbdffe92d491a` completes16 fresh seeds/M,all TASK gates pass. COMBO/S15 at M1/8/16/32:
**.997261/.988757/.970993/.979561**,SE .001758/.002226/.002883/.003839.
Direct COMBO/BEST at M16/M32 **.995504/.993125**,SE .002804/.002482. A:A mean absolute
.225/.480/.309/.362us,max.688/1.552/1.136/.912. Every point winner,including fresh PAIR/SHARD2
comparisons,is in `point_nominations.json`;P4 receives exact combined `review/combo/combo.patch`6b3990b2.
M1/M8 are the same CUDA as independent SHARD2/PAIR;their direct control differences are noise.
P4 confirms the whole-layer held-out R; these stage wins do not promote a whole-layer candidate.

[review/next_issue/README.md](review/next_issue/README.md) records SASS-led follow-up:
FAST replaces CANCEL's duplicated FP64/sign/range paths with bounded FP32 norm metadata;
INDEX saves18 integer/address/control instructions per B-fill segment (1.125/L per16 weights);
FLUSH1 removes the second consecutive zero norm flush. All are separate changes on the measured c49 parent.
All9 variants pass810 native component/replay checks; FAST adds224 extreme helper and16 full-MoE scale checks.
Normal/forced screens,packed-lab admission and queue state live under `review/next_issue/`.

Sent check**`d7fb2d8bddae4b8c`**,sender1971066,wrapper`76a7f192a080d0b6`,on MoE. All6 normal/forced13TASK/12shape/28static screens pass; exact packed16seed/64AA local lab passes5919characters/94.7s. `review/next_issue/admission.json` and `queue.json` pin the send. H200 result pending.
