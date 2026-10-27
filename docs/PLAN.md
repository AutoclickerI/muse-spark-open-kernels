# Plan

## Status (2026-10-27 10:55 KST, day 6 of 14; the kernel track closes 2026-11-05 16:59 KST)
- Board (SOTA "Leaderboard", 10:51): S22 `22874960` 1.8892x, 101st of 757; leader 2.3797x, 10th 2.2210x, 50th
  2.0242x. Gap and target: GOAL "Target".
- Confirmed incumbent: S22 `fd5e0ca2` (shipment `22874960`, [dev/s22](../problems/muse_decode_layer/dev/s22/README.md)),
  official 1.8892x (`210d40e0`, TASK pass): S21 + S19 kernel at d16S + direct-input v1S/d32S hybrids, v16F port
  dropped by its leave-out; native G0 R 1.009997 [1.009669, 1.010326] over S21 (`0e1f9bc8`), prepare 179.87 s.
  Scored chain: S19 1.8621x (`fd4d153f`) -> S21 1.8781x (`ec8cc95c`; public attention ports, R-20261026-1) -> S22.
- S23 (S22 + P10 projection ports, in-graph R 1.0071 [1.0052, 1.0091] on S21): paired lab `0d7f2e48` in flight.
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
| P5b FA3/full +v16S | 16.8 (10.4); KV roof | .80 /1 | six S22 arms unmeasured: 10c8f076 hit the 900 s cap with no log; split resend 5caaec87/d97ca0e9 |
| P5 native sliding | 13.1 (12.7); KV roof | .60 /1 | 26d5054f .75-.78 calls; ca022370 consumer family stays parked |
| P3a MoE M<=32 | 41.1 (21.4); unique stream | .25 /4 | Operand 33f5c4c1: MMA+activation R1.040 modeled, other terms <=1.016; five S22 builds lose (f6970526); re-rank |
| P10 projections/coding | 37.1 (20.9) ideal exact12; current25.7 (11.1) | .25 /4 | 72f273a8: ports+P16 vs S21 R1.0071 [1.0052,1.0091]; S22 handoff with P4; T32 carriers lose |
| Unassigned boundary compute/route | 14.1 (13.0); byte roof | .25 /2 | Distributed BOUND (7342f368, R.978/.988) and per-row route (9e79dda5, T1/8/16 .893, T1 1.0008) lose; deletion oracles 1.0298/1.0203 unrecovered |
| P3b MoE M128/256 | 27.0 (14.3); unique stream | .05 /6 | Split c22acbbf/c67e0a1e: ladder load term was sink tax; branch serialization largest (R1.067), true loads R1.013; re-rank due |
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
| 1 | P4 | G1 | Integration/scoring: S21 eacc087b / expanded 109c0135, frozen at official 1.8781x (ec8cc95c). S22 fd5e0ca2/22874960: phase0 R 1.007814 [1.007209,1.008420] (`6a1303c1`); selected drop=['v16F'], disabled hybrids=[]; phase1 native G0 passes: R 1.009997 [1.009669,1.010326], prepare 179.87s (`0e1f9bc8`); official 1.8892x (`210d40e0`), TASK pass, frozen. Per-cell S19 leave-outs and hybrid arms on 64 fresh seeds x16 symmetric rounds; upper CI<1 selects a loss, each selection needs a fresh disjoint G0 before scoring. [S22](../problems/muse_decode_layer/dev/s22/README.md). | A scored run above the leader's and held-out six-load score above it; final candidate correct and competitive nearby | integration lane | active |
| 2 | P5b | G1, G2 | S22 4b8c803d outcome portfolio CHAIN/DIRECT/STAGE/LANE1/SOFTMAX/OCHAIN (local 156 TASK+144 graphs and 64x16 rehearsal pass). Paired check 10c8f076 (all six) hit the judge's 900 s cap with an empty log, so nothing is measured; a progressing run needs ~400-550 s, so a setup or native stall is likelier than the budget. Resent split with per-phase progress and a stall watchdog: DIRECT/CHAIN/OCHAIN 5caaec87, STAGE/LANE1/SOFTMAX d97ca0e9 (ec754e5 protocol). [Builds/decision](../problems/muse_attention_decode/dev/public_vector/night/README.md). | Held-out whole-load six-load G0 improves; selected composition needs disjoint confirmation | full-attention lane | active: split resend 5caaec87 / d97ca0e9 in flight |
| 3 | P5 | G1, G2 | Consumer finish-deletion oracle on S19 `ca022370`: deleting all setup between QKV and attention gives six-load R1.0159 [1.0109,1.0209] (sliding 1.0115, full 1.0043), below the 1.015 build bar; the consumer-finish fusion family is parked. P5b ports consume FP32 partials directly where they win. Other sliding mechanisms stay open. [Oracle/handoff](../problems/muse_attention_decode/dev/consumer_finish/README.md) | Composed held-out six-load R improves G0 | attention lane | parked: consumer finish family measured below the 1.015 lower-bound bar; broader attention not closed |
| 4 | P3a | G1, G2 | Operand oracle 33f5c4c1 (S19): MMA+activation 8.6-19.3 us at M8-32 (modeled R1.040), recurrence/epilogue ~1.016, other terms <=1.011, overlapping. S22 builds f6970526 all lose whole-load: CACHE1 .9613, LAZY .9046, LAZY2 .9039, N1SIMT .8118, EPI4 .9983 (its d1S point is A/A2 spread); none to P4. [Builds](../problems/muse_decode_layer/dev/nt1_timeline/structural/builds22/README.md). | Whole-load held-out G0;P4 confirms any selected composition | small-M lane | parked: five correct builds lose (f6970526); no buildable operand term above ~1.6% modeled; reopen with a new small-M kernel design after the Levers re-rank |
| 5 | P13 | G1 | On S22 4b8c803d, distributed BOUND 7342f368 loses at every load (A .9778 [.9767,.9789], B .9875 [.9855,.9895]); per-row parallel route 9e79dda5 loses at T8/T16 (A .8928: ~5.4 us per extra row, no weight reuse) and T1-only B is 1.0008 [.9990,1.0025] (d1 +0.2-0.35 us). None to P4; oracles BOUND 1.0298 / ROUTE 1.0203 stay unrecovered. [Evidence](../problems/muse_decode_layer/dev/p13_boundary/distributed/README.md). | Whole-load promotion.py G0; exact winner to P4 | boundary lane | parked: both S22 recipes lose (7342f368, 9e79dda5); reopen with a boundary/route design that reads each weight once per call |
| 5 | P10 | G1 | In-graph ports f66e20e7 (S21): no single site above 1 at six-load lower bound; selected six points + CW8_S2 pool T16. Held-out 72f273a8: that parent vs S21 1.0071 [1.0052,1.0091] (d1 1.014, d16 1.021, v32 1.009); P16 carrier beats CW8_S2 (1.0021 [1.0002,1.0040]); T32 carriers and U16/Q16 lose. Handoff: S22 + PORT2/4/6/11/14 + P16, local normal/forced 13 TASK/12 graphs pass ([P4_HANDOFF](../problems/muse_bf16_proj/dev/public_ports/P4_HANDOFF.json)). | Held-out six-load G0 gain from the largest measured dense term | public-proj lane | active: S22 composition handed to P4; T32 projection sites still open |
| 6 | P3b | G1, G2 | Split J `c22acbbf` / I `c67e0a1e` on S22 (FINDINGS "P3b S22 large-M split"): the ladder's load term was mostly L0's sink tax; true loads-only term 6.6/24.2 us at M128/256 (R1.013: rereads 1.007, activation 1.004, issue 1.002; every residency change and L2 prefetch costs). Largest: branch serialization 53/80 us (R1.067): heavy at 2 CTAs/SM fills the register file and small fills shared memory, so no branch co-resides; heavy alone runs 3.5x its tensor roof. Builds 054e2884 all lose. [Evidence](../problems/muse_decode_layer/dev/moe_excess/split/README.md). | Production whole-load R with A:A; G0 winner to P4 | large-M MoE lane | active: PF loses (R.9857, `8e23096f`, as J predicted); RESH1/CO/XW4 cohort and I2 residency oracle in flight; next a co-resident heavy+small design if I2 shows co-residency works |
| 7 | P7 | G1, G2 | S18 boundary cohort `73d2d80d` (16 seeds, 18 arms, 3456 TASK): ALL/EDITABLE pre-wait loads lose; H2 1.0015 and B1 1.0014 are marginal P4 nominees; the R1.0856 continuity model stays a scenario (best arm recovers 1.8% of it). [Handoff](../problems/muse_bf16_proj/dev/stream_continuity/extend/README.md) | Every boundary ranked; all-boundary recipe judged by held-out G0 | glue lane | parked: pre-wait loads recovered 0% of the modeled bound (`73d2d80d`) and every L2 prefetch form lost (`a405c8b4`) |
| 8 | P11 | G1 | Operands retained across stages, up to a whole-layer persistent kernel at small M: synchronization removal alone is +1.74% estimated (+2.22% if the whole synchronization tax went); a resident kernel needs competitive inner loops, precise readiness and operands kept across stages, not just fewer launches (`docs/reference/research_megakernel.md`); our persistent dense chain lost 1.5-6.5x (`87dc944b`, mechanism unknown) | Reopen when P7's stream-continuity bound or stage receipts lift the estimate above the rows ranked above it | none | parked: +1.74% estimated; waits for P7's bound |

## Questions for Jihoon
None open.
