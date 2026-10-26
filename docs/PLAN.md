# Plan

## Status (2026-10-26 14:30 KST, day 5 of 14; the kernel track closes 2026-11-05 16:59 KST)
- Board (`runs/boards/20261026T033522Z`, 12:35): first S17 `5f35323a` 1.8379x, 90th of 707; leader 2.3601x, 10th
  2.1665x, 50th 1.9518x. Our best scored run is now S18 `28b5c189` 1.8484x (rank not re-fetched). Gap and target:
  GOAL "Target".
- Confirmed incumbent: S18 `95df42fe` (`cand_r3_best7_s18_final.py`, `dev/s18/frozen.json`), scored `28b5c189`
  1.8484x, R 1.004321 [1.002314, 1.006332] over S17 `fea71760` (`5d0282bb`, import-time lab on a lab-only prefill
  host). S18 = S17 + P3a small-M cells (SHARD2+INDEX M1, PAIR M8, PAIR+BLOCK M16 / 32), PDL metadata before wait +
  d1F glue, full-attention V-wait removal at v16F / v32F, P5 D8 / D4 at d16S / d32S, pool128. S17 holds S16,
  guarded S15n, P5 v1 DSM, P7's P shell and v1F tail, CHUNK4 at M128 / 256, the fused pool_router and composition C.
- With P4: S19 on S18 in build (`dev/s19`); P4 row "Next".
- P12 component gaps (ours / leader's time): attention 1.80x, projections 1.36x, MoE 1.26x (`4e108beb`,
  `08cc9405`, `97f5e7f3`); matching all three is ~1.30x: today's leader, ~76% of the gap to ~2.6x. Projections sit at our fitted roof yet
  1.36x behind, so our dense roof is not the leader's limit.
- Where the time goes (S15 timeline `7eaa262a`, `dev/s15_timeline/H200.md`): critical time above the stage roof
  as a share of wall at d1 / d16 / d32 / v1 / v16 / v32: MoE 18.7 / 18.2 / 21.4 / 14.8 / 17.3 / 24.2%; attention
  7.6 / 8.5 / 9.2 / 13.0 / 9.2 / 8.6%; projections 1.9 / 2.6 / 3.9 / 1.9 / 4.1 / 7.4%; glue 3.6 / 0.5 / 0.5 / 0.7
  / 1.3 / 1.5%; GPU idle <= 1.7%. Ceilings on S11: every stage at its roof (2.36 us + bytes at 4.58 TB/s each)
  2.79x; one continuous stream 3.08x; the 4.8 TB/s floor 3.31x.

## Critical path
Loop: a lane edits a candidate or lab payload → local screen (`tools/local_eval.py`, 1-5 min, sm_120, ranks
loosely) → H200 via `okc_run.py send` / `h200_lab.py send` (queue ~6.6 min per position ahead, 1-5 min per run)
→ read the receipt → integration pairs winners in the whole layer → scored run.
- **Bottleneck: H200 runs.** Each day gives 150 (5 problems x 10 scored + 20 checks). On 10-25 the prepare hosts
  were empty by ~17:00 KST and decode by ~21:00, and seven lanes then waited 12 hours for the reset.
- Fixes in place: batched payloads per run; every problem hosts labs (attention at import time, R-20261025-3,
  `--import-budget` default 150 s, proven by `33d8bb68`); labs queue with `--not-before` for the reset; integration's
  A/B gets priority -1 on decode.
- Budget to the close (10.2 days): ~1,500 H200 runs (150 a day; 10-25 spent all of them by 21:00), ~300 of them
  decode runs for compositions. A structural rebuild takes a lane about half a day to a receipt (estimate), so
  each lane has ~20 structural verdicts left: enough for the levers above 30% of the gap only if runs stop going to
  sub-1% pieces.
- Fix in place (10-26): whole-layer paired labs run at import time on any host with checks left (`dev/s18/remote_lab.py`).
  S18 vs S17 with five leave-outs, 16 seeds and 1536 + 260 TASK checks took 398.6 s at import and 434.5 s of the
  900 s run (`5d0282bb`, lab-only prefill host whose prepare stops the run). The host's 300 s prepare cap does not
  bind an import lab; only the 15-minute run does. Decode checks are no longer the bottleneck for compositions.
- Next fix: pack each run fuller (more arms and seeds per payload, all of a lane's variants in one run), spend
  decode checks only on whole-layer pairs, and send each lane's reset batch within the first hour so the day's
  second round fits.

## Levers
Ceiling: the share of the GOAL gap (ln 1.284 = 0.250) the lever removes if it works perfectly at every load it
touches, from the S15 timeline (`7eaa262a`) unless named; every stage at its roof together is 164%. A stage's
share bounds only a lever that removes that stage's time; the leader's component times (P12) show what is
reachable, not a ceiling. One verdict is one H200 run on the stage's host (1-5 min plus queue, 20 checks a day per
host, all arms batched) or, for compositions, a decode check.

| Lever | Stage | Ceiling (basis) | Cost per verdict | Standing |
|---|---|---|---|---|
| MoE M <= 32 at its stream (P3a) | data movement | 54%: 1.144x at d1 / d16 / d32 / v1; leader's MoE component 1.26x ours | MoE-host run | deciding: every reuse rebuild lost 1.12-1.78 |
| Cross-stage stream continuity (P7) | data movement | S15 model32.8%: R1.08557 after excluding glue ramps; original S11 model40% (3.08x vs2.79x). The2.36us intercept includes launch/flush/ramp/tail; HBM idle is unidentifiable from activity | whole-decode paired lab | measured recipes: W pre-wait O1/O8 +0.32%, G0 pass (`a405c8b4`); actual-bank P/F/B/L/X prefetch loses, E release inconclusive. Broader continuity and P11 L2 issue remain open; larger open levers first |
| Attention at its KV roof (P5 sliding, P5b full) | algorithm | 39%: sliding 20%, full 19%; leader's attention component 1.80x ours (35%) | attention-host run | deciding: native kernels have lost to FA3 at verify sliding; untested: a coordinate-vs-transport oracle (page-coordinate gather issue dominates decode) and fixed-reference softmax with the denominator inside PV |
| MoE M = 128 / 256 at its stream (P3b) | data movement | 31%: 1.081x at v16 / v32 | MoE-host run | deciding: multiplicity rebuilds lost 1.13-1.44 |
| Projections below our roof (P10) | data movement | 15% to our roof; 30% at the leader's component time; exact 12-bit coding cuts 25% of dense bytes (FINDINGS "Weight encoding") | proj-host run | deciding: how the leader is 1.36x faster; odds medium-high (the public port's mapping already wins shapes) |
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
| 1 | P4 | G1 | Integration/scored runs. Incumbent S18 `95df42fe` (`cand_r3_best7_s18_final.py`, [dev/s18](../problems/muse_decode_layer/dev/s18/README.md) `frozen.json`): scored `28b5c189` 1.848375x [1.820649, 1.876101]; paired `5d0282bb` (import-time lab on a lab-only prefill host; the decode check was withdrawn unsent) R 1.004321 [1.002314, 1.006332] over S17 `fea71760`, 16 held-out seeds, 1536 TASK + 260 nearby checks. S18 = S17 + P3a SHARD2+INDEX M1, PAIR M8, PAIR+BLOCK M16/32; PDL metadata before wait (M) + d1F glue (D); V-wait removal v16F/v32F; P5 D8 d16S / D4 d32S; S16 pool128. Loads: d1 1.0114, d16 1.0097, d32 1.0012, v1 1.0025, v16 1.0024, v32 0.9989 [.9981, .9997]: S18 loses v32S (S17 at .9948 of its time) and no leave-out recovers it. Leave-outs favour each piece where it acts (BLOCK 1.0021 and pool128 1.0013 with CI > 1; D8 at d16 1.0040, D4 at d32 1.0013, Vwait32 at v32 1.0027). Next: S19 on S18 (`dev/s19`): one v32S arm set (leave-outs of the source-wide pieces there, P5 RAW2 .975938 `47e1f9d8`, PIPE2 alternative), P3a FAST/INDEX M1 and SPLIT2 M32 (alternative arm), P5b d16F K-tile score reuse (`31730357`), gate-audit shape nominees; paired vs S18 on fresh seeds off the decode host, then scored on decode. | A scored run above the leader's, and our six-load score on many seeds above it; the final candidate correct and not much slower at nearby shapes (other batch sizes and context lengths; the final review reruns them, llms.txt:112) | integration lane | active |
| 2 | P3a | G1, G2 | Small-M MoE M1-32,54% of GOAL gap. Earlier c49 PAIR/BLOCK,FAST/INDEX M1 andSPLIT2 M32 alternatives to P4 S18. After depth3/4 lost with residency2→1 and64KiB decoder lost, next structural test now fits3full36864B slots at2CTAs: SINGLE3/DOUBLE3 alias dead route scratch,pack exact8bit reverse/16bit expert offsets and rebuild inverse route after math;HALF3 uses paired4warp partials in2KiB plus routing-byte alias. Exactsm90a counts121–125regs/zero spills;maxshared115680B,CUDAoccupancy calculator includes1024Bdriver reservation and predicts2 atM8/16/32. Four raw slots exceed useful115712B budget before scratch. All6selectors348TASK/capture pass,packed16seed624pair48AA lab passes8071chars;normal/forced screens underway. Send every fitting layout atdepth2/3 versusBASE/S15 in one MoE check;actual runtime residency andall16seed ratios decide,point winners to P4,R confirms. Artifacts:dev/nt1_timeline/structural/occupancy/ | Each load's MoE within ~1.3x of its stream | small-M lane | active |
| 3 | P7 | G1, G2 | S15 `7eaa262a` activity + unique bytes: continuity ceiling13.45/13.59/12.89/12.15/17.25/17.66us per weighted layer at d1/d16/d32/v1/v16/v32, R1.08557 (32.8% of initial GOAL gap); actual GPU idle0/.13/.48/.35/5.45/5.86us. No HBM counter; glue intercepts excluded and concurrent MoE combined. P11 one-store/read materialization0.15/2.10/3.30/1.08/8.19/13.80us HBM-equivalent (joint1.0160); L2 issue and repeated reads remain open. S17 whole-decode `a405c8b4` (prefill-host check because decode quota exhausted),16 fresh seeds x8 cold/1728TASK pass: **W O1/O8 pre-wait weights R1.003215 [1.001964,1.004468] clears G0**; E.994282[.988018,1.000585] fails gate; P/F/B/L/X lose. All6 initial NF vectors precede ACQBULK in SASS; actual consumer banks and S17 NORM/PAIR/CHUNK4 retained. W patch + d1S-only alternative and E d1S/d32S/F points go to P4; unchanged-shape fluctuations are not nominees. [Bound/result/handoff](../problems/muse_bf16_proj/dev/stream_continuity/README.md). These recipes recover0.32% vs8.56% modeled; no standalone sub1% rerun. Larger MoE/attention/coding levers rank ahead of further prefetch work | Reproducible bound and held-out six-load recipe verdict; current-S18 composition remains P4 | glue lane | batch complete; W to P4 |
| 4 | P10 | G1 | Projection gap decomposed in dev/stream_gap/:30shape receipt2015fc90 gives coded1.50–1.57 vs raw2B/W,4.614TB/s marginal stream+2.876us intercept,.519us/launch; component casts cost1.22–1.79us and mostly vanish at layer consumers. Leader private per-shape/code remains unknown. O1 last-CTA fusion loses1.60–2.95;20 large-M reuse/split arms lose1.10–3.73 with real neighbours (646822ab). New exact12 S4/WK4/ST1 QKV wins: M8full .905404[.897972,.912897],M16full .931355[.922068,.940737],M16QKV3 .917619[.910580,.924713],16seeds/24arms,TASK+AA,2b45988c. Patch+all point alternatives in P4_HANDOFF.json; only3 selected builds/two banks,FP32 consumers/gate concurrency preserved. Applied to S18e04635ec asf9e4661d; G0 held-out layer R belongs to P4 and is unmeasured. Physical HBM counters are unmeasured; rates use logical banks. | Every dense stage at its stream roof, or below it through coding | public-proj lane | active |
| 5 | P5 | G1, G2 | Sliding/decode-1 attention (46 of62 layers). Exact-S15 FA3 verify diagnostic45ab5a9b: five launches, actual default splits4/2 at v16/v32, main14.1/19.4us plus combine4.1/4.4us; 1.3x KV targets2.280/4.559us, gather-vs-math not isolated. Rebuilt one launch with two independent KV-owning FP8 WGs/CTA and DSM combine. Paired af563b6b:24 seeds,384 TASK, v32 RAW2 **0.982930 SE0.001788**, earlyS2 **0.997895 SE0.002301**, A:A1.000291; both handed P4 S18, RAW2 preferred. No v16 point winner. D16 D8/D32 D4 already delivered/composed by P4; shared-full/VQ8-sliding claim dropped against exact code. Follow-up47e1f9d8 (18 new seeds,144TASK) repeats v32 RAW2.975938 SE.002329; PIPE2.982128 SE.002075 also handed P4 but loses toRAW2 by1.006358, math/publish grow, regs172→255. No P5 receipt pending. Next: coordinate-vs-transport attribution and a footprint that can overlap the QKV producer; target remains unmet. [Evidence/handoff](../problems/muse_attention_decode/dev/window_rebuild/README.md) | Composed held-out six-load R improves G0; ~1.3x KV floor is a stage target | attention lane | active |
| 6 | P5b | G1, G2 | Full attention (Levers19% of gap). 4df30d04 passes491TASK/8freshseeds: R8 async page-coordinate ring winsd16F.933568[.930596,.936540],d32F.923836[.919381,.928291],3CTA/SM/splits unchanged,pageissue halves; D tensor-core denominator winsv16F.978271[.973818,.982723],v32F.982280[.979030,.985531],softmax1600->1500cyc butMMAissue305->450. Both to P4 S18 through dev/full_attention/page_ring/P4_HANDOFF.json (ontoS15 plus readable CUDA hunks preserving Vwait/PDL); v16 A:A.996655 drift retained. Earlier97d6fa2b: F whole-tile1.0049,G ownership1.16–1.18,nativev1 P1.40,H2 reaches2CTA but loses1.21–1.30. d1 fixed prologue/merge and v1 FA3 fixed work remain; d16/32 now expose transport, verify max/exp/packing remain largest. Next: P4 held-out G0 composition, then fixed-reference/denominator-in-PV oracle by size; no standalone sub1% sends. | Full attention within ~1.3x of its KV floor at every full shape | full-attention lane | active |
| 7 | P3b | G1, G2 | MoE M128/256 (31% of gap). Matched oracle `b5c4d99e`: unique stream114.4/131.8us; CHUNK4 full173.2/224.2. Extra weight requests .46/3.54us, valid-X request increment16.3/43.8us, split-stream increment3.05/−3.43us, full service/shell39.0/48.5us (conditional interventions, not isolated stalls). Count cutoffs8/4/2/0 and one N32/64 path tested in one16seed batch: all-light transfer loses21.7/14.7%, one path~25%; only B4/M128 point .99849 [.98683,1.01015] versus CH4 goes to P4, no standalone small-arm run. CH4 repeats /S15 .97505/.91685. Next: retain2CTA/SM while reusing each X stage across more output tiles; separate service costs before assuming overlap. Exact patches/SE/CIs in branches/balance/P4_HANDOFF.json. | 132.6 / 158.5 us; compare byte floors on matched fixtures | MoE lane | active |
| 8 | P11 | G1 | Operands retained across stages, up to a whole-layer persistent kernel at small M: synchronization removal alone is +1.74% estimated (+2.22% if the whole synchronization tax went); a resident kernel needs competitive inner loops, precise readiness and operands kept across stages, not just fewer launches (`docs/reference/research_megakernel.md`); our persistent dense chain lost 1.5-6.5x (`87dc944b`, mechanism unknown) | Reopen when P7's stream-continuity bound or stage receipts lift the estimate above the rows ranked above it | none | parked: +1.74% estimated; waits for P7's bound |

## Questions for Jihoon
None open.
