# Plan

## Status (2026-10-27 09:22 KST, day 6 of 14; the kernel track closes 2026-11-05 16:59 KST)
- Board (SOTA "Leaderboard", 09:20): S21 `eacc087b` 1.8781x, 105th of 755; leader 2.3797x, 10th 2.2210x, 50th
  2.0182x. Gap and target: GOAL "Target".
- Confirmed incumbent: S21 `109c0135` (`cand_r3_best7_s21_final.py`, `dev/s21/frozen.json`), scored `ec8cc95c`
  1.8781x with the official TASK passing (S19 scored 1.8621x, `fd4d153f`). S21 = S19 + C3 + exact12 gate + Rw4 +
  NP T128 + public attention ports (Lee, operatorstack; R-20261026-1); native G0 R 1.002638 [1.001195, 1.004084]
  over S19 (`58645631`). Its 1.0197 per-call projection did not transfer (decode16 .958: the d16S port blocks the
  gate-GEMM overlap).
- S22 `4b8c803d` ([dev/s22](../problems/muse_decode_layer/dev/s22/README.md)): S21 + S19 at d16S + direct-input
  v1S / d32S hybrids, with in-graph leave-outs per ported cell; lab 3725147 in flight, worker 3807459 prunes,
  confirms and scores.
- Rule from S21: judge every candidate by whole-load time in the production graph; per-call ratios do not transfer.
- Where the time goes: S19 timeline `9c16c850` ([H200.md](../problems/muse_decode_layer/dev/s19_timeline/result/H200.md));
  walls 81 / 190 / 246 / 130 / 321 / 426 us per layer at d1 / d16 / d32 / v1 / v16 / v32. Deletion oracles: setup
  R1.0159 (`ca022370`), boundary b1+b2 1.0298 and route 1.0203, both 1.0513 (`611341aa`).

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
| P3a MoE M<=32 | 41.1 (21.4); unique stream | .25 /4 | 5ee35b94 ceilingR1.04014;operand e1710fb3 then five correct S22 production arms ab36c11e,queued |
| P10 projections/coding | 37.1 (20.9) ideal exact12; current25.7 (11.1) | .25 /4 | 7f110a37/c87c2d8a no winner; e4030204 split-only0.75%D; producer/coding next |
| Unassigned boundary compute/route | 14.1 (13.0); byte roof | .25 /2 | Uncovered6.79-7.26us above1.3roof; calibrated deletion/compute oracle |
| P3b MoE M128/256 | 27.0 (14.3); unique stream | .05 /6 | Public ports closed; S19 ladder bad55569 followed by six S22 production outcome builds aa449a80; both queued/reset09:01 |
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
| 2 | P5b | G1, G2 | S22 4b8c803d production-graph outcome portfolio: CHAIN(PDL), DIRECT(vector partials), STAGE(shared QKV prologue), LANE1(two-CTA target), SOFTMAX(half FMA exp), OCHAIN(native O128/256). Six exact candidates normal/forced:156TASK+144graphs; final 64seed/16round rehearsal3712TASK+260nearby,3sm90a units pass; native gains unmeasured. Diagnostic3831549/check09:00KST10-27 remains unchanged. Sealed dependency worker691620 waits for its receipt, prunes only universally resolved inherited losses, sends ONE paired attention check at rank2 with ec754e5/A:A/all-round telemetry. No call-time promotion. [Builds/decision](../problems/muse_attention_decode/dev/public_vector/night/README.md). | Held-out whole-load six-load G0 improves; selected composition needs disjoint confirmation | full-attention lane | active: diagnostic then built paired cohort; native results pending |
| 3 | P5 | G1, G2 | Consumer finish-deletion oracle on S19 `ca022370`: deleting all setup between QKV and attention gives six-load R1.0159 [1.0109,1.0209] (sliding 1.0115, full 1.0043), below the 1.015 build bar; the consumer-finish fusion family is parked. P5b ports consume FP32 partials directly where they win. Other sliding mechanisms stay open. [Oracle/handoff](../problems/muse_attention_decode/dev/consumer_finish/README.md) | Composed held-out six-load R improves G0 | attention lane | parked: consumer finish family measured below the 1.015 lower-bound bar; broader attention not closed |
| 4 | P3a | G1, G2 | Operand diagnostic3637718/e1710fb3 then S22 4b8c803d whole-graph CACHE1/LAZY/LAZY2/N1SIMT/EPI4,paired sender984504/ab36c11e at reset09:00. 1350leaf TASK+all normal/forced130TASK/120graphs pass;15sm90a units2CTA/zero spills. Finite receipt worker prunes only refuted still-local arms,then hands exact patches/point wins to P4.64fresh seeds×16symmetric all-rounds,A:A,all12graphs;whole-load promotion.py G0. [Builds](../problems/muse_decode_layer/dev/nt1_timeline/structural/builds22/REPORT.md). | Whole-load held-out G0;P4 confirms any selected composition | small-M lane | active: diagnostic then built paired cohort;native results pending |
| 5 | P13 | G1 | S22 distributed b1 row-scale/stats, then per-row route. Both 534 local checks+sm90a pass; boundary queued09:01 reset, route after its receipt. [Evidence](../problems/muse_decode_layer/dev/p13_boundary/distributed/README.md). | Whole-load promotion.py G0; exact winner to P4 | boundary lane | active: native quota wait; no new R yet |
| 5 | P10 | G1 | c74d2a26 ports:36cells/4408TASK; stage P32 .950465[.944382,.956588],15optional points with P4. Native stream scenario d16 carriers pool11.580/U.918/O.478/Q.302us; d32 Q3.000/P2.848/U2.536 (pool32 rawBF16S6),Rmodel1.032884. Corrected queued3347492 to3966799 before quota: fixed activation-TMA hoist, same arms/seeds/protocol,848+640localTASK; no native result. Night dispatcher4088520 waits exact parent-completion event, reads whole_read.py/drops losers/hands P4, then h200_lab --lane P10 sends S21+native portfolio with seven DSM/code/TMA carrier groups,leaveouts,and codedP32 alternative.512+208localTASK twice,normal/forced13TASK12graphs,9sm90NF64–96regs/no spills,TMA W-before/X-after-wait proof. Exact code/queue/evidence dev/public_ports/night/. No new G0 claim; old physical split unmeasured. | Held-out six-load G0 gain from the largest measured dense term | public-proj lane | active: parent 3966799 queued for the reset; dispatcher 4088520 sends the dependent lab |
| 6 | P3b | G1, G2 | S19 ladder bad55569/sender3645567 then S22 4b8c803d production batch aa449a80/sender384352 at rank6/reset09:01. Six outcome arms REG/DECODE/PAD/OCC/BALANCE/EPI target loads,residency,decode,MMA padding,tail,readiness. 72component TASK+22sm90a audits; exact normal/forced64x16 rehearsals each3840TASK+1040nearby pass. First local700 unassigned/not reproduced. Worker408337 reads diagnostic, prunes only qualified refuted arms while local, then reads whole-load R. [Evidence](../problems/muse_decode_layer/dev/moe_excess/builds/README.md). Public ports remain closed. | Native term evidence and production whole-load R; selected composition needs disjoint G0 | large-M MoE lane | active: both cohorts queued; no H200 gain yet |
| 7 | P7 | G1, G2 | S18 boundary cohort `73d2d80d` (16 seeds, 18 arms, 3456 TASK): ALL/EDITABLE pre-wait loads lose; H2 1.0015 and B1 1.0014 are marginal P4 nominees; the R1.0856 continuity model stays a scenario (best arm recovers 1.8% of it). [Handoff](../problems/muse_bf16_proj/dev/stream_continuity/extend/README.md) | Every boundary ranked; all-boundary recipe judged by held-out G0 | glue lane | parked: pre-wait loads recovered 0% of the modeled bound (`73d2d80d`) and every L2 prefetch form lost (`a405c8b4`) |
| 8 | P11 | G1 | Operands retained across stages, up to a whole-layer persistent kernel at small M: synchronization removal alone is +1.74% estimated (+2.22% if the whole synchronization tax went); a resident kernel needs competitive inner loops, precise readiness and operands kept across stages, not just fewer launches (`docs/reference/research_megakernel.md`); our persistent dense chain lost 1.5-6.5x (`87dc944b`, mechanism unknown) | Reopen when P7's stream-continuity bound or stage receipts lift the estimate above the rows ranked above it | none | parked: +1.74% estimated; waits for P7's bound |

## Questions for Jihoon
None open.
