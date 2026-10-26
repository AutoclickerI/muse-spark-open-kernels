# Plan

## Status (2026-10-26 23:10 KST, day 5 of 14; the kernel track closes 2026-11-05 16:59 KST)
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
- P4 S22 `4b8c803d` (shipment `d1ccdc35`, [dev/s22](../problems/muse_decode_layer/dev/s22/README.md)): native G0/per-cell CIs pending; all hosts quota0 at send, awaiting reset; phase0, source4b8c803d, shipmentd1ccdc35; dropped=[], disabled hybrids=[]; sender3725147, payload403ff22fe051272d; finite follow-through worker3807459 selects resolved losers, confirms on new64seeds, scores only after passing G0.
  S21+d16S restore+direct-input v1S/d32S hybrids. Six remaining ported cells have S19 in-graph leave-outs;
  both hybrids also have prior-port arms. 64freshseeds x16symmetric rounds, all rounds/telemetry retained.
  CI upper<1 drops a cell/hybrid; each selected composition requires fresh disjoint G0 before scoring.
  S21 scored sender3225144 at09:02 and S19 sender4046452 at09:01 stay; no new freeze.
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
| P3a MoE M<=32 | 41.1 (21.4); unique stream | .25 /4 | 5ee35b94 residual ceilingR1.04014;operand/recurrence/padding batch e1710fb3 queued,production S21 next |
| P10 projections/coding | 37.1 (20.9) ideal exact12; current25.7 (11.1) | .25 /4 | 7f110a37/c87c2d8a no winner; e4030204 split-only0.75%D; producer/coding next |
| Unassigned boundary compute/route | 14.1 (13.0); byte roof | .25 /2 | Uncovered6.79-7.26us above1.3roof; calibrated deletion/compute oracle |
| P3b MoE M128/256 | 27.0 (14.3); unique stream | .05 /6 | Public ports closed; exact S19 matched excess ladder queued bad55569 for10-27 09:01 KST quota reset; largest term/build unassigned |
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
| 1 | P4 | G1 | Integration/scoring: board S18 28b5c189 1.8484x; S19 62ea5bf4 G0 1.014463, scored sender4046452 at09:01; accepted S21 109c0135/eacc087b G0 1.002638[1.001195,1.004084],prepare145.86s, scored sender3225144 at09:02 unchanged. S22 native G0/per-cell CIs pending; all hosts quota0 at send, awaiting reset; phase0, source4b8c803d, shipmentd1ccdc35; dropped=[], disabled hybrids=[]; sender3725147, payload403ff22fe051272d; finite follow-through worker3807459 selects resolved losers, confirms on new64seeds, scores only after passing G0. S19 d16S restored; direct-input OP v1S and Lee d32S; six S19 per-cell leave-outs plus incremental hybrid arms. Same64x16symmetric telemetry; upperCI<1 selects losses, fresh confirmation required after any selection. [S22](../problems/muse_decode_layer/dev/s22/README.md). No new freeze. | A scored run above the leader's and held-out six-load score above it; final candidate correct and competitive nearby | integration lane | active |
| 2 | P5b | G1, G2 | Production-graph transfer supersedes call-point nominations: S21 58645631 no_PUBLIC/S21 .998709[.997343,1.000077]; d16S port/S19 whole-load1.043149[1.042282,1.044018]. Source audit finds ordinary pack/public main launches, losing independent gate-split overlap atd16S; timing attribution pending. Preserve accepted9b043487 onS21eacc087b. Native batch 3831549 queued attention/check09:00KST10-27, payloadb4a5ec1b:7cell S19 restores +D(PDL)/V(16B direct, CTA-sharedQ/KV)/F, d1S/F native vectors;8freshseeds8symmetric rounds432TASK/full kernel_fn graphs/CUPTI neighbours. Both108localTASK,28eager+graph each,5sm90a units,8bit-exact readers;15 balanced prologue cases99crossings/1idle-peer; ready6a605fa2 unmeasured. [Audit](../problems/muse_attention_decode/dev/public_vector/README.md). P4 confirms selected whole-load points with disjoint G0 seeds. | Whole-load held-out six-load G0 improves; roof is a target | full-attention lane | active |
| 3 | P5 | G1, G2 | Consumer finish-deletion oracle on S19 `ca022370`: deleting all setup between QKV and attention gives six-load R1.0159 [1.0109,1.0209] (sliding 1.0115, full 1.0043), below the 1.015 build bar; the consumer-finish fusion family is parked. P5b ports consume FP32 partials directly where they win. Other sliding mechanisms stay open. [Oracle/handoff](../problems/muse_attention_decode/dev/consumer_finish/README.md) | Composed held-out six-load R improves G0 | attention lane | parked: consumer finish family measured below the 1.015 lower-bound bar; broader attention not closed |
| 4 | P3a | G1, G2 | Matched S19 operand oracles queued for10-27 09:00KST: sender3637718/wrappere1710fb3. Split fragments/Bf lifetime,accumulator recurrence,N-padding and epilogue;56sm90a layouts2CTA/zero spills,448native+162nearby TASK pass. H200 timing pending(all hosts exhausted). Largest term then correct build in S21 production graphs;whole six-load G0,not per-call transfer. dev/nt1_timeline/structural/operand/REPORT.md. | Each load's MoE within ~1.3x of its stream | small-M lane | active: await oracle receipt,then rank build by production whole-load R |
| 5 | P13 | G1 | S22 distributed b1 row-scale/stats, then per-row route. Both 534 local checks+sm90a pass; boundary queued09:01 reset, route after its receipt. [Evidence](../problems/muse_decode_layer/dev/p13_boundary/distributed/README.md). | Whole-load promotion.py G0; exact winner to P4 | boundary lane | active: native quota wait; no new R yet |
| 5 | P10 | G1 | c74d2a26 ports:36cells/4408TASK; stage P32 .950465[.944382,.956588],15optional points with P4. Native stream scenario d16 carriers pool11.580/U.918/O.478/Q.302us; d32 Q3.000/P2.848/U2.536 (pool32 rawBF16S6),Rmodel1.032884. Corrected queued3347492 to3966799 before quota: fixed activation-TMA hoist, same arms/seeds/protocol,848+640localTASK; no native result. Night dispatcher4088520 waits exact parent-completion event, reads whole_read.py/drops losers/hands P4, then h200_lab --lane P10 sends S21+native portfolio with seven DSM/code/TMA carrier groups,leaveouts,and codedP32 alternative.512+208localTASK twice,normal/forced13TASK12graphs,9sm90NF64–96regs/no spills,TMA W-before/X-after-wait proof. Exact code/queue/evidence dev/public_ports/night/. No new G0 claim; old physical split unmeasured. | Held-out six-load G0 gain from the largest measured dense term | public-proj lane | queued: parent3966799 at reset; dependent dispatcher4088520 |
| 6 | P3b | G1, G2 | Assign exact S19 W13/W2 M128/256 excess: 12-arm matched L0/L1/L2/L3/BASE/MATH/INIT/STREAM plus AA/allocation/clone/observer controls; 8fresh seeds x4palindromic all-round timings, telemetry and CTA/SM work/finish histograms. Native64cut SASS/resource audit,384local TASK and lossless11154-byte transport pass; instrumented spills disclosed. Queue bad55569 (sender3645567) waits10-27 09:01 KST/all5hosts0+0; no native ladder/CI yet. Read3330edc2; no repeated gather/cluster-sync. After term assignment, size six-load ceiling, build against largest term and fresh promotion.py gate; G0 patch to P4. Public-port recipe550f0aa8/17b91ef9 closed. [Evidence](../problems/muse_decode_layer/dev/moe_excess/README.md). | Term-by-term ladder assigns the excess and a measured build is judged by six-load G0; park only with that evidence | large-M MoE lane | active: queued native ladder; terms/build unassigned until quota reset |
| 7 | P7 | G1, G2 | S18 boundary cohort `73d2d80d` (16 seeds, 18 arms, 3456 TASK): ALL/EDITABLE pre-wait loads lose; H2 1.0015 and B1 1.0014 are marginal P4 nominees; the R1.0856 continuity model stays a scenario (best arm recovers 1.8% of it). [Handoff](../problems/muse_bf16_proj/dev/stream_continuity/extend/README.md) | Every boundary ranked; all-boundary recipe judged by held-out G0 | glue lane | parked: pre-wait loads recovered 0% of the modeled bound (`73d2d80d`) and every L2 prefetch form lost (`a405c8b4`) |
| 8 | P11 | G1 | Operands retained across stages, up to a whole-layer persistent kernel at small M: synchronization removal alone is +1.74% estimated (+2.22% if the whole synchronization tax went); a resident kernel needs competitive inner loops, precise readiness and operands kept across stages, not just fewer launches (`docs/reference/research_megakernel.md`); our persistent dense chain lost 1.5-6.5x (`87dc944b`, mechanism unknown) | Reopen when P7's stream-continuity bound or stage receipts lift the estimate above the rows ranked above it | none | parked: +1.74% estimated; waits for P7's bound |

## Questions for Jihoon
None open.
