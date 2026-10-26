# Plan

## Status (2026-10-26 14:30 KST, day 5 of 14; the kernel track closes 2026-11-05 16:59 KST)
- Board (`runs/boards/20261026T033522Z`, 12:35): first S17 `5f35323a` 1.8379x, 90th of 707; leader 2.3601x, 10th
  2.1665x, 50th 1.9518x. Our best scored run is now S18 `28b5c189` 1.8484x (rank not re-fetched). Gap and target:
  GOAL "Target".
- Confirmed incumbent: S17 `fea71760` (`cand_r3_best7_s17_final.py`, `dev/s17/frozen.json`), scored `7ec5f9ab`
  1.8286x, R 1.003310 [1.001006, 1.005620] over S16 `ee5fe249` (`da0cd64b`). It holds S16, guarded S15n, P5 v1 DSM,
  P7's P shell and v1F tail, CHUNK4 at M128 / 256, the fused pool_router at T1 / 8 / 16 and composition C except
  pool128.
- With P4: S18 `95df42fe` = S17 + P3a small-M cells (SHARD2+INDEX M1, PAIR M8, PAIR+BLOCK M16 / 32), PDL metadata
  before wait + d1F glue, full-attention V-wait removal at v16F / v32F, P5 D8 / D4 at d16S / d32S, pool128;
  scored `28b5c189` 1.8484x (unpaired); its paired check against S17 waits for the 10-27 reset (`dev/s18/README.md`).
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
| Cross-stage stream continuity (P7) | data movement | 40% at most: one continuous stream 3.08x vs every stage at its roof 2.79x (S11 model, `662bb89`); the 2.36 us intercept includes launch, flush, ramp and tail that may not hide (`5cefdbc4`), and Lurrobert's next-stage L2 prefetch lost 0.4-10.6% (SOTA) | decode check | unmeasured: bytes moved vs time on each critical interval; prefetch of the banks each consumer actually reads |
| Attention at its KV roof (P5 sliding, P5b full) | algorithm | 39%: sliding 20%, full 19%; leader's attention component 1.80x ours (35%) | attention-host run | deciding: native kernels have lost to FA3 at verify sliding; untested: a coordinate-vs-transport oracle (page-coordinate gather issue dominates decode) and fixed-reference softmax with the denominator inside PV |
| MoE M = 128 / 256 at its stream (P3b) | data movement | 31%: 1.081x at v16 / v32 | MoE-host run | deciding: multiplicity rebuilds lost 1.13-1.44 |
| Projections below our roof (P10) | data movement | 15% to our roof; 30% at the leader's component time; exact 12-bit coding cuts 25% of dense bytes (FINDINGS "Weight encoding") | proj-host run | deciding: how the leader is 1.36x faster; odds medium-high (the public port's mapping already wins shapes) |
| Operands retained across stages (P11) | fusion | unmeasured: a retained-operand oracle (activations and pages kept on chip or in L2 between stages; lee101's MoE lost 20% to activation L2 traffic at bs 128 / 256); synchronization removal alone is +1.74% (7%) | days of build | waits: P7's bound, then the oracle |
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
| 1 | P4 | G1 | Integration/scored runs. Incumbent S17 `fea71760` (`cand_r3_best7_s17_final.py`, `dev/s17/frozen.json`): scored `7ec5f9ab` 1.828637x, paired `da0cd64b` R 1.003310 [1.001006, 1.005620] over S16 `ee5fe249` (1536 TASK + 390 nearby checks). S18 `95df42fe` on S17 ([dev/s18](../problems/muse_decode_layer/dev/s18/README.md)): P3a SHARD2+INDEX M1, PAIR M8, PAIR+BLOCK M16/32; PDL metadata before wait (M) + d1F glue (D); full-attention V-wait removal v16F/v32F; P5 D8 d16S / D4 d32S; S16 pool128 (all composition C adds on S17); CHUNK4 M128/256 already in S17. Scored `28b5c189` 1.848375x [1.820649, 1.876101] (board best, unpaired). Its paired check against S17 (leave-outs Vwait32, BLOCK, D8, D4, pool128; 16 held-out seeds; 1536 TASK + 260 nearby checks) waits for the 10-27 reset at priority -1; read it with `dev/s18/report.py`. Not yet composed: P5b d16F K-tile score reuse (.99979, `31730357`); gate-audit shape nominees (q1 full page-flight B16/B32, heavy LIW/TYPW, M256 PAIR2; `dev/gate_audit`) | A scored run above the leader's, and our six-load score on many seeds above it; the final candidate correct and not much slower at nearby shapes (other batch sizes and context lengths; the final review reruns them, llms.txt:112) | integration lane | active |
| 2 | P3a | G1, G2 | Small-M MoE, M=1-32 (decode loads, verify 1; 14.8-21.4% of wall above roof). S14: NT1 flags `{1:95, 8:69, 16:69, 32:69}`, 64.9 / 75.9 / 104.7 us at M=8 / 16 / 32 vs a 49-53 us pure stream. Every reuse rebuild lost 1.12-1.78 (`b5659458`, `2fad2371`); the 58.7 us math clock in FINDINGS is that failed ONCE schedule, not S14. Static count: integer / address issue at 74% of its pipe at M=32, HMUL2 and tensor 17% (`dev/nt1_timeline/PIPE.md`). PAIR epilogue 0.975 / 0.973 / 0.979 of S14's MoE at M=8 / 16 / 32 (`d650c04e`, `29ad388d`), to P4. Sent: FLIGHT / ILP / K16 / LUT rebuilds (`de47736b`). Review items measured (FINDINGS "P3a S15 review receipt and composition"): BLOCK 0.992-0.994 at M16 / 32 and SHARD2 at M1 to P4; CANCEL lost 1.002-1.023, PACK5 lost, TWO_BASE dropped (integer issue 37 -> 406). Next, by size (oracles `360810b1` at M8 / 16 / 32: loads-only on the same depth-2 pipeline 0.745 / 0.723 / 0.668, trivial dequant 0.930 / 0.913 / 0.828, pure stream 49-53 us): the shallow pipeline and W2-ready wait first (9-16 us short of streaming; W2-ready 11.6 us at M32), then dequant issue; reuse is bounded at ~9 / ~35 us at M16 / 32 (`cf1de160`) | Each load's MoE within ~1.3x of its stream | small-M lane | active |
| 3 | P7 | G1, G2 | Cross-stage stream continuity (Levers: 40% of the gap, a model bound). Every stage at its roof pays ~2.36 us of ramp and latency per stage; one continuous weight stream does not (S11: 2.79x vs 3.08x). First, the bound from the S15 timeline without an H200 run: HBM-idle time on the critical path per load and which boundaries carry it. Then remove it: the next stage's weights prefetched into L2 within the call (CONSTRAINTS boundary cases) while a latency-bound stage (attention, route, glue) runs; each producer's `griddepcontrol.launch_dependents` right after its last load is issued, not after it lands (34 us on Llama-1B, SOTA "Public competitor code"); consumers' weight loads issued before their dependency wait. Composition C (early release, R 1.006161) is the measured start. Glue pieces already measured (v1F tail `e5414dd7`, `11d531dc`) go to P4; new glue-only work is parked by size (Levers) | HBM-idle time on the critical path at each load measured, then at most half of it left in a whole-layer paired run | glue lane | active |
| 4 | P10 | G1 | Dense projections (32.9 of ~92 us critical at d1; 1.36x the leader's component time although at our fitted roof). In S15: NathenFernandes's MIT kernels at the shapes they win in our layer (R-20261025-2; qkv 0.71 at t=1, 0.82 at t=32; O 0.85-0.96; unpooler 0.81 / 0.90 at t=16 / 32; their fused pool / router loses, `e3c637d1`). Fused pool_router at T1 / 8 / 16: layer 1.0118x [1.0104, 1.0133] (`818bd6da`, to P4). Sparse-plane readers lost 1.5-14.6x (`3a91afed`, `83c65f7b`). Next: find how the leader is 1.36x faster on the projection component at our shapes (bytes per weight, achieved stream rate per shape, launch count, fusion with neighbours), then build that | Every dense stage at its stream roof, or below it through coding | public-proj lane | active |
| 5 | P5 | G1, G2 | Sliding/decode-1 attention (46 of62 layers). Exact-S15 FA3 verify diagnostic45ab5a9b: five launches, actual default splits4/2 at v16/v32, main14.1/19.4us plus combine4.1/4.4us; 1.3x KV targets2.280/4.559us, gather-vs-math not isolated. Rebuilt one launch with two independent KV-owning FP8 WGs/CTA and DSM combine. Paired af563b6b:24 seeds,384 TASK, v32 RAW2 **0.982930 SE0.001788**, earlyS2 **0.997895 SE0.002301**, A:A1.000291; both handed P4 S18, RAW2 preferred. No v16 point winner. D16 D8/D32 D4 already delivered/composed by P4; shared-full/VQ8-sliding claim dropped against exact code. Sent PIPE2 QK/PV/softmax overlap, ordinary edge,18 fresh seeds,A/A2/RAW2 controls:47e1f9d82ee745e9. Next: read its clocks/verdict; bulk prefetch still limited by QKV shared footprint, coordinate-vs-transport oracle remains open. [Evidence/handoff](../problems/muse_attention_decode/dev/window_rebuild/README.md) | Composed held-out six-load R improves G0; ~1.3x KV floor is a stage target | attention lane | active |
| 6 | P5b | G1, G2 | Full attention (Levers19% of gap). 31730357 K-tile reuse d16F.999788[.993369,1.006208] point handed P4; d32F1.007311. Roof batch97d6fa2b passes571TASK/8seeds: F balancing1.0049; G vector-row gather1.163/1.183; nativev1 P1.401; H2 reaches2CTA/SM but losesv16/32 1.215/1.303. Actual S15 clocks: d1 prologue/merge dominates2–4tiles; d16/32 pageissue1679/1698 vswait27/25; v16/32 N128softmax1633/1601 vsMMAissue305/303. Next batched structural arms R4/R8 retain row mapping with async page-coordinate rings; D retries tensor denominator under TASK rather than extra1e-6 probe. All6full loads/8freshseeds/A:A/native clocks; normal+forced480TASK pass. Attention check4df30d04a39e433e,sender2948858,pending. | Full attention within ~1.3x of its KV floor at every full shape | full-attention lane | active |
| 7 | P3b | G1, G2 | MoE at M=128 / 256 (verify 16 / 32; 17.3 / 24.2% of wall above roof at v16 / v32). S14: light experts decoded once on an N16 third stream (182.4 / 256.4 us). The heavy branch runs at 1.9-2.3x its stream: weights read 1.6x / 2.2x and activations re-read (171 / 403 MB). Decode-once with cluster activation sharing lost 1.13-1.44 (`8ee003db`), as did shared queues (`ea794250`). At the reset: launch-topology / stage-3 attribution lab. Next: remove the weight and activation multiplicity without the staging cost those recipes paid | 132.6 / 158.5 us (unique bytes at ~4 TB/s plus 1.3x) | MoE lane | active |
| 8 | P11 | G1 | Operands retained across stages, up to a whole-layer persistent kernel at small M: synchronization removal alone is +1.74% estimated (+2.22% if the whole synchronization tax went); a resident kernel needs competitive inner loops, precise readiness and operands kept across stages, not just fewer launches (`docs/reference/research_megakernel.md`); our persistent dense chain lost 1.5-6.5x (`87dc944b`, mechanism unknown) | Reopen when P7's stream-continuity bound or stage receipts lift the estimate above the rows ranked above it | none | parked: +1.74% estimated; waits for P7's bound |

## Questions for Jihoon
None open.
