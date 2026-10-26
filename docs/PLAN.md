# Plan

## Status (2026-10-26 22:09 KST, day 5 of 14; the kernel track closes 2026-11-05 16:59 KST)
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
- P4 S21 `109c0135` (shipment `eacc087b`, [dev/s21](../problems/muse_decode_layer/dev/s21/README.md)):
  S19+C3+exact12 gate+v32F Rw4+NP T128, Lee at d16S/d32S/v1F/v16S/v16F/v32S and operatorstack v1S;
  MIT/Apache/NOTICE shipped. Native G0 **passes**, R **1.002638 [1.001195,1.004084]** over S19 on 64 fresh seeds
  x16 symmetric rounds (`5864563117fd44b6`, projection checks), 3136 TASK+260 nearby and five fixture checks pass;
  full prepare **145.86 s**. Point projection 1.019738 did not transfer: decode16 R .958360; grouped port leave-out
  .998709 [.997343,1.000077] has no established net port gain. A:A R1.000991 (+0.10% drift), supplementary INC/A2
  1.001646 [1.000178,1.003116]; primary gate unchanged. 1167 contracts, both13TASK+12graphs, native compiles,
  exact packed rehearsal and10820-byte tail transport pass;34 tool tests. Payload `7a5b62175cd8`, wrapper `702486c83f56`.
  Scored sender **3225144**, 10-27 **09:02 KST**, after unchanged S19 sender4046452 at09:01. No new freeze.
  Next batch: restore S19 at d16S (~+.71% six-load point from production leave-out), with new disjoint G0 seeds;
  no separate small confirmation. Exact old S20 fixture did not recur; its original round/cause stay unproven.
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
S19 `9c16c850` (12 S/F shapes,3fresh seeds x3rounds,72TASK). D=ln(2.6/1.875109)=.326845;
S19 score is lab-estimated, unscored. Ceiling=%D to4.8TB/s or989TFLOP/s roof (parentheses:1.3roof).
Order=ceiling*p/C; p=judgment prior, C=forecast hours/verdict including a batched lab. P4 executes first.
Nonadditive. Bases, scenarios, odds and receipts: FINDINGS "S19 timeline and lever re-rank";
[evidence](../problems/muse_decode_layer/dev/s19_timeline/README.md).

| Lever | Ceiling (% gap; basis) | p / C hours | Next decision / standing |
|---|---|---|---|
| P5b FA3/full +v16S | 16.8 (10.4); KV roof | .80 /1 | 26d5054f ~.76 calls; P4 G0 pending |
| P5 native sliding | 13.1 (12.7); KV roof | .60 /1 | 26d5054f .75-.78 calls; ca022370 consumer family stays parked |
| P3a MoE M<=32 | 41.1 (21.4); unique stream | .25 /4 | da1fb2ee decode R1.00718; wait-only5.1%D, not family cap |
| P10 projections/coding | 37.1 (20.9) ideal exact12; current25.7 (11.1) | .25 /4 | 7f110a37/c87c2d8a no winner; e4030204 split-only0.75%D; producer/coding next |
| Unassigned boundary compute/route | 14.1 (13.0); byte roof | .25 /2 | Uncovered6.79-7.26us above1.3roof; calibrated deletion/compute oracle |
| P3b MoE M128/256 | 27.0 (14.3); unique stream | .05 /6 | 550f0aa8/17b91ef9 public ports and actual hybrids lose; matched W13 point unresolved; tested recipe parked |
| P7 stream continuity | 25.1; S19 ramp+idle scenario | .02 /6 | 73d2d80d/a405c8b4 no large causal gain; parked |
| P11 retained operands | 5.9; S19 materialization+idle scenario | .05 /24 | Bytes alone4.5%D; issue/reuse unmeasured; parked |

## Ranked work
Promotion (GOAL G0) has one score gate: per seed pair R = exp(mean over the six loads of log(T_old / T_new)),
each load's T = 46 t_sliding + 16 t_full; promote when R's lower confidence bound across held-out confirmation
seeds is above 1. Correctness is TASK.md's checker on every arm and seed, nothing stricter. A win at every load,
stage or row is not required; shape-dispatchable pieces are chosen per shape, then the composition is confirmed by
R; alternatives are ranked by R. A loss closes only the recipe tested; a result from a failed control or an
unproven implementation is unmeasured. Receipts and mechanisms: FINDINGS.

| Rank | Id | Serves | Work | Done when | Who | State |
|---|---|---|---|---|---|---|
| 1 | P4 | G1 | Integration/scored runs. Board incumbent S18 `28b5c189` 1.848375x. S19 `62ea5bf4` (shipment `8b415553`): paired `6ac9317e` R 1.014463 [1.013044,1.015885], prepare 147.7 s; scored sender 4046452 unchanged for 10-27 09:01 KST. S20 failed G0 `0640cce4` R 1.002271 [.996468,1.008109]; retained INC/seed1730886002 verify16 ratio1.290288, A:A1.001889; individual round/incident telemetry absent, no similar earlier spike. Old order has unequal exposure positions; four global A:A CIs cover1 in2/4, insufficient to certify a robust replacement. S21 `109c0135` (shipment `eacc087b`, [dev/s21](../problems/muse_decode_layer/dev/s21/README.md)) = S19+C3+exact12 gate+v32F Rw4+NP T128+licensed public attention: Lee d16S/d32S/v1F/v16S/v16F/v32S, operatorstack v1S; replaces FA3/R64/VQ16 where ports win. Native G0 passes5864563117fd44b6 on projection checks:64freshseeds x16symmetric rounds,R1.002638[1.001195,1.004084],prepare145.86s,3136TASK+260nearby+5fixture TASK pass. Five leave-outs: no_PUBLIC.998709[.997343,1.000077] does not establish net port gain; decode16R.958360 versus point projection1.019738 whole. A:A+0.10pct drift; supplementalINC/A2R1.001646[1.000178,1.003116],primary gate unchanged. 1167contracts,both13TASK+12graphs,nativecompile,exactpacked/10820-byte transport and34tests pass; lossless evidence codec preserves all samples/rounds. Payload7a5b62175cd8,wrapper702486c83f56. Prefill waiters cancelled after quota exhaustion; lab sender3156569 completed. Official scored sender3225144 at09:02 after S19; no new freeze until official TASK. Next batch restoreS19 d16S (~+.71pct transfer),freshG0,add late P5b direct-input v1S/d32S points; no standalone small confirmation. | A scored run above the leader's and held-out six-load score above it; final candidate correct and competitive nearby | integration lane | active |
| 2 | P5b | G1, G2 | Post-port hybrids cde88b9136de4093:780nativeTASK/8freshseeds/4752retained cold times/2N cyclic+reverse cohorts. v1S directOP F **.839438[.819888,.858987]**, d32S directLee F **.985855[.960184,1.011527]**; both2->1launch. Other cells retain picks: generic scalar FP32 qread loses2.805/3.059 atd16S/v16S; producerV/earlyQ/poly25/H and fullK-major lose in tested forms. Vectorized plane-outer reader and d1S gather/fill-drain open. Rank to1.3KV: d1S Rcap1.0121, v1S1.0095, d16S1.0073, d1F1.0068; d32F last1.0005 (scenario). Combined licensed patch9b043487/sourcefbaf2793 on current P4S21eacc087b; strong v1S-only80e45c72, marginal d32S-onlya59cb4b2. Selected families pruned, native/AST/SASS-equivalence proof and28eager+graph cases normal/forced pass; P4 confirms full prepare/held-outG0. v1F AA−.665pct disclosed, no nominee there. First1c5f6391 unmeasured oversized packet; exact timestamp codec repair loses no samples. No run queued. [Evidence](../problems/muse_attention_decode/dev/public_hybrid/README.md) | Full attention within ~1.3x KV floor; composed held-out six-load G0 improves | full-attention lane | active |
| 3 | P5 | G1, G2 | Consumer finish-deletion oracle exact S19 62ea5bf4 measured ca022370c1f34adc (prefill import,8freshseeds/all12S/Fshapes,6 complete palindromic cold rounds,192BASE/AA+96ready-fixture TASK, all rounds+384telemetry reads retained). DEL six-load R1.015900[1.010938,1.020887], A:A1.000336[.996087,1.004602]; sliding-only1.011513[1.006474,1.016578], full-only1.004287[1.000021,1.008570]. LB1.010938< prescribed1.015: park consumer finish family, no fusion build/nominee. Native paths already fuse head finish; v1S is R2 DSM QT4/C16, FA3 onlyv1F/v16S; DEL removes native prep, full sumcast and FA3 prep+metadata. Initial510d394f unmeasured schema/live-history faults repaired; six source proofs+sm90a compiles+exact-wrapper admission pass. Earlier four-RAW2-cell parking rationale withdrawn. Other sliding/producer mechanisms remain open; prior d32 page-map point stays P4. [Oracle/handoff](../problems/muse_attention_decode/dev/consumer_finish/README.md) | Composed held-out six-load R improves G0;1.3KV targets2.280/4.559us remain open | attention lane | parked: consumer finish family measured below this decision's1.015 lower-bound threshold; broader attention not closed |
| 4 | P3a | G1, G2 | Small-M MoE M1-32:54% theoretical gap lever,now parked by next-step size/odds. S19 dependency554c392d:16seeds48oracle48AA240correctpairs,all controls/resources pass;wait-only5.19/4.54/6.83us gives modeled six-loadR1.01667 and~2.127us common added-cost budget for>1%. DecompositionSIZING.md separates historical stream49.44/50.84/53.22,actual loads-only47.586/53.740/69.284 from S19;exclusive exposed compute/per-phase fill-drain unmeasured. Coarse S256/S512 count108regs/no spills;ownerDSM97168/109456B models2CTA but cluster occupancy/U overhead unverified. Full output partial>=204800B fails2CTA. C16/C8 count-scaled costproxy givesR1.00206/1.00929 before sync/underfill/finish;C8 favorable halfproxy+.5us R1.01140 remains unmeasured,halfproxy+1us R1.00984. Finish-only fusion cannot rescue slower main;best measuredA128D1.2979/1.2529/1.2322. Reopen on calibrated coarseDSM/fusedfinish below~2.13us common extra cost or a measured new compute/load-issue saving;leader per-shape/public mechanism also reopens. Leader1.25835x is aggregateS15,not a small-M S19 result. No newarm/job/P4 nominee;prior handoffs/G0 unchanged. Artifacts:dev/nt1_timeline/structural/dependency/SIZING.md,sizing.json. | Each load's MoE within ~1.3x of its stream | small-M lane | active: not parked (orchestrator): dependency/DSM forms closed, but the 17-45 us residual is unassigned (excess over stream ~R1.12); now one matched S19 ladder (loads-only, +decode, +MMA, +epilogue, BASE, math-only, stream, per-CTA histogram) at M1-32, then build on the largest term |
| 5 | P13 | G1 | S19 oracle611341aa route/bound/both R1.02034/1.02980/1.05126 (LB1.01871/1.02833/1.04975). Nativefec0b672 route fusion R.95713[.95530,.95897],552TASK pass; no nominee. [Evidence](../problems/muse_decode_layer/dev/p13_boundary/README.md). | Exact six-load G0 patch to P4 | boundary lane | route recipes rejected; b1/b2 fusion open |
| 5 | P10 | G1 | native_boundary complete: S19 native8freshseeds16symmetric rounds retained, metadata-only CUPTI e70024b1. Small-M actual coded-byte stream target gives2.910/13.278/8.393/2.927us at d1/d16/d32/v1, modeled six-loadR1.032884[1.030488,1.035286]; large upper26.517/52.635us at v16/v32,R1.038411[1.037944,1.038878]. Tested larger first: e4030204 native one-boundary deletion,320TASK; split-only3.175us v16/neutral v32,R1.002444[1.001132,1.003757], producer exclusive issue/wait still unmeasured. Built24dispatch plans with unchanged S19 WGMMA body/no new bank: c87c2d8a448layer+480strictTASK, all point means lose in changed-load whole layer, closest O1281.001106[.999258,1.002958]. No new G0 nominee; native_boundary/P4_HANDOFF.json negatives, previous optional NP/gate unchanged. Next small-M coded pool/router d16 surplus dominates; same full-graph/PDL boundary, test weight issue before X-ready. Old9ab physical split remains unmeasured, not evidence small terms. | Held-out six-load G0 gain from the largest measured dense term | public-proj lane | active |
| 6 | P3b | G1, G2 | Public ports decided on exact S19: 550f0aa8 eight fresh seeds x4 symmetric cold rounds,598TASK/all6AA CIs cover1; operator and Lee v43d variants lose all30whole cells. Native O W2/Lee finish advantages omitted S19 work; matched17b91ef9 eight new seeds x4/132TASK/bothAA cover1: O W13 M128 .981412 [.904507,1.058317] unresolved, O W2 1.025524 [.943486,1.107562], Lee finish1.350141 [1.307908,1.392374]. All8actual hybrids lose; metadata+lost branch overlap consume W13 point. Retain S19 all M,no winner patch/G0 nominee. M1-32 phase times,licenses,source/SASS mechanisms and negative P4 handoff in dev/moe_public_ports/. Prior3330edc2 activation-reuse recipe remains a scoped loss; structural ceiling stays open. | A credible >1% six-load case or a new matched sub-kernel speedup before another structural build | MoE public-port lane | parked: tested public ports/hybrids lose; no resolved matched speedup mechanism, M128 W13 point unresolved |
| 7 | P7 | G1, G2 | S18 full boundary cohort `73d2d80d` completes16 fresh paired seeds x8 cold,18 arms/12 shapes/3456 TASK all pass; A:A exact fingerprint. Q .666395, O .970939, pool .613680, unpool .710344, NT1 W13 .987878, W2 .999361, heavy W13 1.000721, b2 .997603, KV .995181; ALL .382568[.379952,.385202], EDITABLE .991852[.990393,.993313] lose. **H2 1.001504[1.000124,1.002887], B1 1.001355[1.000064,1.002647] raw G0 pass**; unchanged G1.001344[1.000094,1.002596], both nominee/G CIs span1, retained as caveat. 39sm90a CUDA/25sm90 Triton builds,146 SASS witnesses; QKV3 smem restores79regs; native OL ABI stand-in proves/fixes GNU-unique smem guard collision plus missing PDL launch. Model R1.08557 remains scenario only: ALL recovers0%, best raw H2 1.83% of log bound. [Handoff/results](../problems/muse_bf16_proj/dev/stream_continuity/extend/README.md): H2/B1 nominees plus readable patches/control evidence to P4 for fresh S19 composition (old W already dropped there). No broader continuity/P11 closure | Every requested boundary ranked and all-boundary recipe judged by held-out G0; P4 owns S19 integration | glue lane | parked: pre-wait loads recovered 0% of the modeled bound across 12 boundaries (`73d2d80d`) and every L2 prefetch form lost (`a405c8b4`); heavy W2 and b1 are marginal P4 nominees |
| 8 | P11 | G1 | Operands retained across stages, up to a whole-layer persistent kernel at small M: synchronization removal alone is +1.74% estimated (+2.22% if the whole synchronization tax went); a resident kernel needs competitive inner loops, precise readiness and operands kept across stages, not just fewer launches (`docs/reference/research_megakernel.md`); our persistent dense chain lost 1.5-6.5x (`87dc944b`, mechanism unknown) | Reopen when P7's stream-continuity bound or stage receipts lift the estimate above the rows ranked above it | none | parked: +1.74% estimated; waits for P7's bound |

## Questions for Jihoon
None open.
