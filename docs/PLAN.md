# Plan

## Status (2026-10-26 20:41 KST, day 5 of 14; the kernel track closes 2026-11-05 16:59 KST)
- Board (SOTA "Leaderboard", 17:24): S18 `28b5c189` 1.8484x, 100th of 719; leader 2.3797x, 10th 2.1761x, 50th
  1.9561x. Gap and target: GOAL "Target".
- Confirmed incumbent: S18 `95df42fe` (`cand_r3_best7_s18_final.py`, `dev/s18/frozen.json`), scored `28b5c189`
  1.8484x, R 1.004321 [1.002314, 1.006332] over S17 `fea71760` (`5d0282bb`, import-time lab on a lab-only prefill
  host). S18 = S17 + P3a small-M cells (SHARD2+INDEX M1, PAIR M8, PAIR+BLOCK M16 / 32), PDL metadata before wait +
  d1F glue, full-attention V-wait removal at v16F / v32F, P5 D8 / D4 at d16S / d32S, pool128. S17 holds S16,
  guarded S15n, P5 v1 DSM, P7's P shell and v1F tail, CHUNK4 at M128 / 256, the fused pool_router and composition C.
- With P4: S19 `62ea5bf4` (`cand_r3_best7_s19_final.py`, `dev/s19/frozen.json`) lab-confirmed at R 1.014463
  [1.013044, 1.015885] over S18 (`6ac9317e`, 16 fresh seeds, import-time lab on the prefill host); its scored
  decode run waits for the 10-27 09:01 KST reset, and promotion completes when it passes TASK.
- P4 S20 `9e6f525f` (shipment `f9bd44d4`, [dev/s20](../problems/muse_decode_layer/dev/s20/README.md))
  **fails G0**: R 1.002271 [0.996468, 1.008109] over S19, one native prefill check `0640cce4b12f42f9`.
  All 960 TASK + 260 nearby checks pass; prepare 157.65 s. Seven leave-outs resolve C3/M64 gains and B1 loss
  (no_B1/S20 .997004 [.996359, .997649]); other pieces unresolved. One v16S/verify16 seed inflates the CI;
  it is retained. No scored S20 queued and neither candidate newly frozen. S19 sender 4046452 is unchanged.
- P4 S21 `81404eb2` (shipment `d5c6dd24`, [dev/s21](../problems/muse_decode_layer/dev/s21/README.md)):
  S19 + C3, exact12 gate, P5b FA3 schedule/combine (R64 subsumes M64), v16F/v32F VQ Rw4, P10 T128 native-layout
  banks; B1/PRIVATE1/AFFINE32/H2 dropped. 1340 contracts, both 13 TASK + 12 graph screens, changed sm90a bridge/
  reducer/VQ compiles, packed 2880 TASK + 260 nearby and five old-fixture checks pass locally; wire 11,463 bytes.
  Point projection 1.008688 (sensitivity 1.008412): prefill-check sender 2627539 replaces 1990194, held to 10-27
  09:00 KST; wrapper `627b0d37b019`, payload `bb4e7ec97dc6`. Add qualified P3a/P5 nominees before ~08:30;
  later ones go to S22, release early above ~1%. P5 consumer-deletion oracle has no candidate nomination.
  Keep 64 fixed fresh seeds x16 symmetric all-round cohorts, unchanged G0 gate, six leave-outs, telemetry and
  old-fixture replay. Composed native R/prepare remain unmeasured; target prepare ~200 s. S19 sender unchanged.
- P12 component gaps (ours / leader's time): attention 1.80x, projections 1.36x, MoE 1.26x (`4e108beb`,
  `08cc9405`, `97f5e7f3`); matching all three is ~1.30x: today's leader, ~76% of the gap to ~2.6x. Projections sit at our fitted roof yet
  1.36x behind, so our dense roof is not the leader's limit.
- Where the time goes (S15 timeline `7eaa262a`, `dev/s15_timeline/H200.md`): critical time above the stage roof
  as a share of wall at d1 / d16 / d32 / v1 / v16 / v32: MoE 18.7 / 18.2 / 21.4 / 14.8 / 17.3 / 24.2%; attention
  7.6 / 8.5 / 9.2 / 13.0 / 9.2 / 8.6%; projections 1.9 / 2.6 / 3.9 / 1.9 / 4.1 / 7.4%; glue 3.6 / 0.5 / 0.5 / 0.7
  / 1.3 / 1.5%; GPU idle <= 1.7%. Ceilings on S11: every stage at its roof (2.36 us + bytes at 4.58 TB/s each)
  2.79x; one continuous stream 3.08x; the 4.8 TB/s floor 3.31x.

## Critical path
Loop: a lane edits a candidate or lab payload → static preflight, changed-unit compile/check, one transport
smoke (`tools/local_eval.py`; sm_120 screens rank loosely; full confirmations only on H200) → H200 via `okc_run.py send` / `h200_lab.py send` (queue ~6.6 min per position ahead, 1-5 min per run)
→ read the receipt → integration pairs winners in the whole layer → scored run.
- **Bottleneck: H200 runs.** Each day gives 150 (5 problems x 10 scored + 20 checks). On 10-25 the prepare hosts
  were empty by ~17:00 KST and decode by ~21:00, and seven lanes then waited 12 hours for the reset.
- Fixes in place: batched payloads per run; every problem hosts labs (attention at import time, R-20261025-3,
  `--import-budget` default 150 s, proven by `33d8bb68`); labs queue with `--not-before` for the reset; integration's
  A/B uses `--lane P4 --decision "..."` at PLAN rank 1 on any host.
- Budget to the close (10.2 days): ~1,500 H200 runs (150 a day; 10-25 spent all of them by 21:00), ~300 of them
  decode runs for compositions. A structural rebuild takes a lane about half a day to a receipt (estimate), so
  each lane has ~20 structural verdicts left: enough for the levers above 30% of the gap only if runs stop going to
  sub-1% pieces.
- Fix in place (10-26): whole-layer paired labs run at import time on any host with checks left (`dev/s18/remote_lab.py`).
  S18 vs S17 with five leave-outs, 16 seeds and 1536 + 260 TASK checks took 398.6 s at import and 434.5 s of the
  900 s run (`5d0282bb`, lab-only prefill host whose prepare stops the run). The host's 300 s prepare cap does not
  bind an import lab; only the 15-minute run does. Decode checks are no longer the bottleneck for compositions.
- `prepare` binds: S18 took 241.6 s of 300 s and S19 round 1 254.7 s (`7c0fd3cc`; its first lab died at 239 s
  inside a prepare-time oracle, `3597359b`). Fix in place (P4, `dev/s19` bake): the receipts' picks are constants,
  the private FP32 oracles are gone, serial compiles run in the pool and unread ones are not built; S19 round 2
  prepares in 147.7 s (`6ac9317e`).
- Loop transport: declare payload compile/run/output/context risk for batching; use `--lab-only` for
  self-preparing import labs and `--lane <active P-id> --decision "..."` for PLAN scheduling on every host.
  Packets reserve output before logs; screens queue each pass immediately. Usage/migration: RESOURCES
  "Experiment loop (P1)"; completed loop work and live evidence: FINDINGS "Judge, quotas and loop".

## Levers
Ceiling: the share of the GOAL gap (ln 1.284 = 0.250) the lever removes if it works perfectly at every load it
touches, from the S15 timeline (`7eaa262a`) unless named; every stage at its roof together is 164%. A stage's
share bounds only a lever that removes that stage's time; the leader's component times (P12) show what is
reachable, not a ceiling. One verdict is one H200 run on the stage's host (1-5 min plus queue, 20 checks a day per
host, all arms batched) or, for compositions, a decode check.

| Lever | Stage | Ceiling (basis) | Cost per verdict | Standing |
|---|---|---|---|---|
| MoE M <= 32 at its stream (P3a) | data movement | 54%: 1.144x at d1 / d16 / d32 / v1; leader's MoE component 1.26x ours | MoE-host run | deciding: every reuse rebuild lost 1.12-1.78 |
| Cross-stage stream continuity (P7) | data movement | S15 scenario R1.08557 (32.8% of initial gap), not a physical HBM ceiling. Best S18 tested raw gain H2 +0.15% =1.83% of that log bound; G duplicate also +0.13%, so causal gain unresolved | whole-decode paired lab | parked: measured recovery ~0% of the model bound (`73d2d80d`, `a405c8b4`) |
| Attention at its KV roof (P5 sliding, P5b full) | algorithm | 39%: sliding 20%, full 19%; leader's attention component 1.80x ours (35%) | attention-host run | deciding: native kernels have lost to FA3 at verify sliding; untested: a coordinate-vs-transport oracle (page-coordinate gather issue dominates decode) and fixed-reference softmax with the denominator inside PV |
| MoE M = 128 / 256 at its stream (P3b) | data movement | 31%: 1.081x at v16 / v32 | MoE-host run | parked: mechanism oracle `3330edc2` found no build above 1% of the score |
| Projections below our roof (P10) | data movement | 15% to our roof; 30% at the leader's component time; exact 12-bit coding cuts 25% of dense bytes | proj-host run | gate1/sliding G0 +0.44% is with P4. Large-shape oracle9ab81a3f separates activation/tensor/epilogue only in an explicit raw schedule; residual includes timing/prologue. K-major banks improve24matched producer controls by1.5–11.4%, but native G0.999817[.999058,1.000576] is neutral; SS/packed rebuild G0.971503/.977755 reject. Private1.36x remains unidentified; re-rank from native verdict, next isolate producer issue/wait+split consumer with S19 geometry |
| Operands retained across stages (P11) | fusion | ~13%: retention 6% (HBM-equivalent 0.15 / 2.10 / 3.30 / 1.08 / 8.19 / 13.80 us per layer at d1 / d16 / d32 / v1 / v16 / v32, P7's `BOUND.md`; L2 and reread effects unmeasured) plus synchronization removal 7% (+1.74%) | days of build | parked: size, below every active lever |
| Glue: b1, route, b2 | fusion | 5.5%: 3.6% of wall at d1, <= 1.5% elsewhere | decode check | parked: size; measured pieces go to P4 |

## Ranked work
Promotion (GOAL G0) has one score gate: per seed pair R = exp(mean over the six loads of log(T_old / T_new)),
each load's T = 46 t_sliding + 16 t_full; promote when R's lower confidence bound across held-out confirmation
seeds is above 1. Correctness is TASK.md's checker on every arm and seed, nothing stricter. A win at every load,
stage or row is not required; shape-dispatchable pieces are chosen per shape, then the composition is confirmed by
R; alternatives are ranked by R. A loss closes only the recipe tested; a result from a failed control or an
unproven implementation is unmeasured. Receipts and mechanisms: FINDINGS.

| Rank | Id | Serves | Work | Done when | Who | State |
|---|---|---|---|---|---|---|
| 1 | P4 | G1 | Integration/scored runs. Board incumbent S18 `28b5c189` 1.848375x. S19 `62ea5bf4` (shipment `8b415553`): paired `6ac9317e` R 1.014463 [1.013044,1.015885], prepare 147.7 s; scored sender 4046452 unchanged for 10-27 09:01 KST. S20 failed G0 `0640cce4` R 1.002271 [.996468,1.008109]; retained INC/seed1730886002 verify16 ratio1.290288, A:A1.001889; individual round/incident telemetry absent, no similar earlier spike. Old order has unequal exposure positions; four global A:A CIs cover1 in2/4, insufficient to certify a robust replacement. S21 `81404eb2` (shipment `d5c6dd24`, [dev/s21](../problems/muse_decode_layer/dev/s21/README.md)) = S19+C3+exact12 gate+FA3 shape schedule/parallel combine (R64 subsumes M64)+v16F/v32F VQ Rw4+P10 T128 native-layout banks; drop B1/PRIVATE1/AFFINE32/H2. 1340 contracts, normal/forced 13TASK+12graphs each, changed sm90a bridge/reducer/VQ compile, packed2880TASK+260nearby+5old-fixture checks and11463-byte full-tail transport pass locally. Prospective64freshseeds x16 symmetric all-round cohorts, unchanged promotion.py Student-t gate, six leave-outs, telemetry/enqueue traces; repeat exact old fixture after primary. Point projection1.008688 (sensitivity1.008412): prefill-check sender2627539 replaces1990194, held09:00 reset; wrapper627b0d37b019/payloadbb4e7ec97dc6. Add qualified P3a/P5 nominees by~08:30, laterS22, release early above~1pct. P5 new deletion oracle has no candidate. Native composed R/prepare pending; ~200s prepare target. No S21 scored run or new freeze. | A scored run above the leader's and held-out six-load score above it; final candidate correct and competitive nearby | integration lane | active |
| 2 | P10 | G1 | large_split completed: oracle9ab81a3f eightT128/256sites,8seeds8cold; operational activation/tensor/epilogue CIs in oracle_split.md, opaque-Lt physical split unmeasured. SS24arms/site and exact GPU K-major42arms/site pass original projection/router checks; all24 matched layout comparisons win. Held-out S19 G0 7f110a37:16freshseeds960layerTASK+48strictchecks; SS R.971503[.970251,.972755], packed.977755[.976731,.978780], native-layout-only.999817[.999058,1.000576], no promotion. Native verify16 +.41% alone is not G0; optional np_patch.py/P4_HANDOFF.json for a planned composition batch only, addedprepare.316s/GPU.185ms. Existing dense_all gate1/sliding R1.004405[1.003003,1.005808] stays with P4. Next native issue/wait+split-consumer term under one timing boundary; private1.36x residual still unknown. | Every dense stage at its stream roof, or below it through coding | public-proj lane | active |
| 3 | P3a | G1, G2 | Small-M MoE M1-32:54% theoretical gap lever,now parked by next-step size/odds. S19 dependency554c392d:16seeds48oracle48AA240correctpairs,all controls/resources pass;wait-only5.19/4.54/6.83us gives modeled six-loadR1.01667 and~2.127us common added-cost budget for>1%. DecompositionSIZING.md separates historical stream49.44/50.84/53.22,actual loads-only47.586/53.740/69.284 from S19;exclusive exposed compute/per-phase fill-drain unmeasured. Coarse S256/S512 count108regs/no spills;ownerDSM97168/109456B models2CTA but cluster occupancy/U overhead unverified. Full output partial>=204800B fails2CTA. C16/C8 count-scaled costproxy givesR1.00206/1.00929 before sync/underfill/finish;C8 favorable halfproxy+.5us R1.01140 remains unmeasured,halfproxy+1us R1.00984. Finish-only fusion cannot rescue slower main;best measuredA128D1.2979/1.2529/1.2322. Reopen on calibrated coarseDSM/fusedfinish below~2.13us common extra cost or a measured new compute/load-issue saving;leader per-shape/public mechanism also reopens. Leader1.25835x is aggregateS15,not a small-M S19 result. No newarm/job/P4 nominee;prior handoffs/G0 unchanged. Artifacts:dev/nt1_timeline/structural/dependency/SIZING.md,sizing.json. | Each load's MoE within ~1.3x of its stream | small-M lane | active: not parked (orchestrator): dependency/DSM forms closed, but the 17-45 us residual is unassigned (excess over stream ~R1.12); now one matched S19 ladder (loads-only, +decode, +MMA, +epilogue, BASE, math-only, stream, per-CTA histogram) at M1-32, then build on the largest term |
| 4 | P5b | G1, G2 | S19 all-FA3/parallel merge9dc1159a076b4e6b,928nativeTASK/8freshseeds8cold. Preferred points: v16S R4w4 **.890066[.878445,.901687]**, v1F R64 **.888090[.875082,.901098]**, v16F/v32F Rw4 .992111[.986609,.997613]/.997763[.994638,1.000888]. Actual FA3 calls5->3 launches, metadata-only5->4. S19 v1S is native1launch, not FA3; explicit FA3 replacements all lose, no P5 native/query/RAW2 change. R/M incremental .933739/.982057; full F/G DSM forms lose1.005–1.034, fixed geometry already1.37–1.72x S19. A:A v1S-1.79pct drift disclosed, real FA3 controls cover1; no per-load veto. Two earlier pre-timing correctness attempts unmeasured, fixed slots/workspace/layout with all-rank local proof. All nine changed point estimates and independent S19 patches to P4; combined5f18b0a5/source dbeeb847, FA3-only ef9e9be4, native-row41e683b8, metadata ce4a7609. Next largest residual: FA3 main/parallel-merge tail and D main softmax/packing, not fixed-cluster fusion; true FA3 main epilogue fusion open. P4 checks held-out G0 and added full-prepare compilation. [Evidence](../problems/muse_attention_decode/dev/full_attention/chain/README.md) | Full attention within ~1.3x KV floor; composed held-out six-load G0 improves | full-attention lane | active |
| 5 | P5 | G1, G2 | Consumer finish-deletion oracle exact S19 62ea5bf4 measured ca022370c1f34adc (prefill import,8freshseeds/all12S/Fshapes,6 complete palindromic cold rounds,192BASE/AA+96ready-fixture TASK, all rounds+384telemetry reads retained). DEL six-load R1.015900[1.010938,1.020887], A:A1.000336[.996087,1.004602]; sliding-only1.011513[1.006474,1.016578], full-only1.004287[1.000021,1.008570]. LB1.010938< prescribed1.015: park consumer finish family, no fusion build/nominee. Native paths already fuse head finish; v1S is R2 DSM QT4/C16, FA3 onlyv1F/v16S; DEL removes native prep, full sumcast and FA3 prep+metadata. Initial510d394f unmeasured schema/live-history faults repaired; six source proofs+sm90a compiles+exact-wrapper admission pass. Earlier four-RAW2-cell parking rationale withdrawn. Other sliding/producer mechanisms remain open; prior d32 page-map point stays P4. [Oracle/handoff](../problems/muse_attention_decode/dev/consumer_finish/README.md) | Composed held-out six-load R improves G0;1.3KV targets2.280/4.559us remain open | attention lane | parked: consumer finish family measured below this decision's1.015 lower-bound threshold; broader attention not closed |
| 6 | P7 | G1, G2 | S18 full boundary cohort `73d2d80d` completes16 fresh paired seeds x8 cold,18 arms/12 shapes/3456 TASK all pass; A:A exact fingerprint. Q .666395, O .970939, pool .613680, unpool .710344, NT1 W13 .987878, W2 .999361, heavy W13 1.000721, b2 .997603, KV .995181; ALL .382568[.379952,.385202], EDITABLE .991852[.990393,.993313] lose. **H2 1.001504[1.000124,1.002887], B1 1.001355[1.000064,1.002647] raw G0 pass**; unchanged G1.001344[1.000094,1.002596], both nominee/G CIs span1, retained as caveat. 39sm90a CUDA/25sm90 Triton builds,146 SASS witnesses; QKV3 smem restores79regs; native OL ABI stand-in proves/fixes GNU-unique smem guard collision plus missing PDL launch. Model R1.08557 remains scenario only: ALL recovers0%, best raw H2 1.83% of log bound. [Handoff/results](../problems/muse_bf16_proj/dev/stream_continuity/extend/README.md): H2/B1 nominees plus readable patches/control evidence to P4 for fresh S19 composition (old W already dropped there). No broader continuity/P11 closure | Every requested boundary ranked and all-boundary recipe judged by held-out G0; P4 owns S19 integration | glue lane | parked: pre-wait loads recovered 0% of the modeled bound across 12 boundaries (`73d2d80d`) and every L2 prefetch form lost (`a405c8b4`); heavy W2 and b1 are marginal P4 nominees |
| 7 | P3b | G1, G2 | Mechanism decided by cheap S18 oracle `3330edc2`,16seeds x3fixtures,all nativeTASK/AA pass. PREPACK/PACK .9191/.9388(gather~15us);EARLY3/PACK3 1.0014/1.0036(same bytes/footprint,no gain);DUPCTA/DUPSYNC .8840/.8166(sync costs~26/56us);MC2/DUPSYNC .9998/.9985(half requests,no resolved net gain). Literal MC2-local is unsafe;own-copy controls isolate sync correctly. Old +39/+100us term was mixed reader granularity/issue/cache attribution,not bandwidth alone. Every timed-gather arm losesBASE. Free-gatherPACK .99567/.99169 gives at most~0.21% optimistic six-load transfer even at100% MoE layer share;no supported >1% next build. No P4 candidates;earlierB4 stays P4 decision. Propose parking by odds/size,retain31% theoretical ceiling,reopen on a new measured mechanism after fresh re-rank. Evidence branches/mechanism/. | A credible >1% six-load case before another structural build | MoE lane | parked: no build above 1% of the score after the mechanism oracle `3330edc2` (gather ~15 us, cluster sync 26 / 56 us; activation bytes are not the cost) |
| 8 | P11 | G1 | Operands retained across stages, up to a whole-layer persistent kernel at small M: synchronization removal alone is +1.74% estimated (+2.22% if the whole synchronization tax went); a resident kernel needs competitive inner loops, precise readiness and operands kept across stages, not just fewer launches (`docs/reference/research_megakernel.md`); our persistent dense chain lost 1.5-6.5x (`87dc944b`, mechanism unknown) | Reopen when P7's stream-continuity bound or stage receipts lift the estimate above the rows ranked above it | none | parked: +1.74% estimated; waits for P7's bound |

## Questions for Jihoon
None open.
