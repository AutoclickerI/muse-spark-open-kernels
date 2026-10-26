# Plan

## Status (2026-10-26 12:40 KST, day 5 of 14; the kernel track closes 2026-11-05 16:59 KST)
- Board (`runs/boards/20261026T033522Z`): S17 `5f35323a` 1.8379x, 90th of 707; leader 2.3601x, 10th 2.1665x, 50th
  1.9518x. Gap and target: GOAL "Target".
- Scored S17 = frozen S16 `435df5f78a0d` + guarded S15n (nearby shapes) + v1 cooperative DSM sliding + large-M and
  VDSK qkv release + P7's P shell, minus the losing v16 DSM / pool128; its held-out paired confirmation is P4's.
  S15 `60254883` (`cand_r3_best7_s15.py`) is the last confirmed incumbent (1.026 over S14, `281b1383`).
- Pending with P4: composition C (PDL release, R 1.006161 [1.004298, 1.008028], `60bad72b`), full-attention
  redundant-V-wait removal (0.997 / 0.999 at verify 16 / 32 full, `1bab0b2d`), P7's v1F tail (`11d531dc`),
  pool_router (1.0118x, `818bd6da`), P5's d16 D8 / d32 D4.
- P12 component gaps (ours / leader's time): attention 1.80x, projections 1.36x, MoE 1.26x (`4e108beb`,
  `08cc9405`, `97f5e7f3`); matching all three is ~1.30x, the whole gap. Projections sit at our fitted roof yet
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
| Cross-stage stream continuity (P7) | data movement | 40%: one continuous stream 3.08x vs every stage at its roof 2.79x (S11 model, `662bb89`) | decode check | unmeasured: HBM-idle time on the S15 critical path, from the timeline (no H200 run) |
| Attention at its KV roof (P5 sliding, P5b full) | algorithm | 39%: sliding 20%, full 19%; leader's attention component 1.80x ours (35%) | attention-host run | deciding: native kernels have lost to FA3 at verify sliding |
| MoE M = 128 / 256 at its stream (P3b) | data movement | 31%: 1.081x at v16 / v32 | MoE-host run | deciding: multiplicity rebuilds lost 1.13-1.44 |
| Projections below our roof (P10) | data movement | 15% to our roof; 30% at the leader's component time; exact 12-bit coding cuts 25% of dense bytes (FINDINGS "Weight encoding") | proj-host run | deciding: how the leader is 1.36x faster |
| Whole-layer persistent kernel (P11) | fusion | 7%: +1.74% estimated synchronization tax | days of build | waits: stream continuity's bound |
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
| 1 | P4 | G1 | Integration and scored runs. Exact S15 `60254883` frozen; S14 retired, S15c dropped. Score gate implemented in `dev/s16/score_gate.py`: held-out paired six-load R lower95% bound >1, TASK only; shape selections use measured ratios, source-wide policies use geomean. Today's re-judgment: `dev/s16/gate_rejudgment.json`; early projection release reverses old acceptance (R=0.999729). S16 `435df5f78a0d` adds global C128 d1S, DSM C16 v1S/C8 v16S, NORM M1/PAIR M8/16/32, early attention/MoE release, O128/256 and pool128. Reset-only ablation scored `7139485dbeef4e0d`; latest scored `c6cc2874ab994ee9` and paired check `37c2879f62554dff` accepted (2304 local TASK checks per mode); state in `dev/s16/response.json`. Missing labs remain unmeasured. | A scored run above the leader's, and our six-load score on many seeds above it; the final candidate correct and not much slower at nearby shapes (other batch sizes and context lengths; the final review reruns them, llms.txt:112) | integration lane | active |
| 2 | P3a | G1, G2 | Small-M MoE, M=1-32 (decode loads, verify 1; 14.8-21.4% of wall above roof). S14: NT1 flags `{1:95, 8:69, 16:69, 32:69}`, 64.9 / 75.9 / 104.7 us at M=8 / 16 / 32 vs a 49-53 us pure stream. Every reuse rebuild lost 1.12-1.78 (`b5659458`, `2fad2371`); the 58.7 us math clock in FINDINGS is that failed ONCE schedule, not S14. Static count: integer / address issue at 74% of its pipe at M=32, HMUL2 and tensor 17% (`dev/nt1_timeline/PIPE.md`). PAIR epilogue 0.975 / 0.973 / 0.979 of S14's MoE at M=8 / 16 / 32 (`d650c04e`, `29ad388d`), to P4. Sent: FLIGHT / ILP / K16 / LUT rebuilds (`de47736b`). Next, each a small exact change: W2 waits on `done[e, nb]` (its own token block) instead of every block of expert e (M=16 / 32); W2's global scale cancelled through the following RMSNorm (multiply by nothing, eps / a^2 per expert, sign kept; not W13, SwiGLU follows); norm accumulators in 2 / 4 / 8 shards between PAIR and LOCAL; NVFP4 decode as sign and exponent assembly on two bases S and 1.5S (bit-exact on all 8,128 FP4 x E4M3 pairs per the review; count SASS first, it fails if integer issue grows); expert scale bytes as exact 5-bit packets where the real weights allow (up to 4.17% of tile bytes; measure compressibility first). The last two apply to P3b's heavy branch too | Each load's MoE within ~1.3x of its stream | small-M lane | active |
| 3 | P7 | G1, G2 | Cross-stage stream continuity (Levers: 40% of the gap, a model bound). Every stage at its roof pays ~2.36 us of ramp and latency per stage; one continuous weight stream does not (S11: 2.79x vs 3.08x). First, the bound from the S15 timeline without an H200 run: HBM-idle time on the critical path per load and which boundaries carry it. Then remove it: the next stage's weights prefetched into L2 within the call (CONSTRAINTS boundary cases) while a latency-bound stage (attention, route, glue) runs; each producer's `griddepcontrol.launch_dependents` right after its last load is issued, not after it lands (34 us on Llama-1B, SOTA "Public competitor code"); consumers' weight loads issued before their dependency wait. Composition C (early release, R 1.006161) is the measured start. Glue pieces already measured (v1F tail `e5414dd7`, `11d531dc`) go to P4; new glue-only work is parked by size (Levers) | HBM-idle time on the critical path at each load measured, then at most half of it left in a whole-layer paired run | glue lane | active |
| 4 | P3b | G1, G2 | MoE at M=128 / 256 (verify 16 / 32; 17.3 / 24.2% of wall above roof at v16 / v32). S14: light experts decoded once on an N16 third stream (182.4 / 256.4 us). The heavy branch runs at 1.9-2.3x its stream: weights read 1.6x / 2.2x and activations re-read (171 / 403 MB). Decode-once with cluster activation sharing lost 1.13-1.44 (`8ee003db`), as did shared queues (`ea794250`). At the reset: launch-topology / stage-3 attribution lab. Next: remove the weight and activation multiplicity without the staging cost those recipes paid | 132.6 / 158.5 us (unique bytes at ~4 TB/s plus 1.3x) | MoE lane | active |
| 5 | P5 | G1, G2 | Sliding and decode-1 attention (sliding is 46 of 62 layers; attention is 7.6-13.0% of wall above roof). In S15: last-CTA decode-1 sliding (one launch, 15.1 -> 11.1 us, `7e93cc81`). At the reset: early history fetch (the 2,048-key gather and V transpose issued before the PDL wait, smem 53.5 -> 28.9 KB, `dev/early_history/`), native DSM cluster sweep `f1c270e6`. Verify-32 sliding is 29.8 us per layer vs a 3.5 us KV floor (`c6bd22e2`). Sent: round 2 (decode 16 / 32 and verify-1 one-launch, 28 arms), verify_resident (verify 16 / 32: one launch, history before the PDL wait, DSM reduction). Next: K-ready and V-ready published separately wherever QK of the next tile waits on the V transpose (the shared `full[slot]` barrier) | Sliding <= 6-8 us per layer at every load | attention lane | active |
| 6 | P10 | G1 | Dense projections (32.9 of ~92 us critical at d1; 1.36x the leader's component time although at our fitted roof). In S15: NathenFernandes's MIT kernels at the shapes they win in our layer (R-20261025-2; qkv 0.71 at t=1, 0.82 at t=32; O 0.85-0.96; unpooler 0.81 / 0.90 at t=16 / 32; their fused pool / router loses, `e3c637d1`). Fused pool_router at T1 / 8 / 16: layer 1.0118x [1.0104, 1.0133] (`818bd6da`, to P4). Sparse-plane readers lost 1.5-14.6x (`3a91afed`, `83c65f7b`). Next: find how the leader is 1.36x faster on the projection component at our shapes (bytes per weight, achieved stream rate per shape, launch count, fusion with neighbours), then build that | Every dense stage at its stream roof, or below it through coding | public-proj lane | active |
| 7 | P5b | G1, G2 | Full attention (16 of 62 layers). In S15: register-fold N128 full verify (0.977 / 0.948 at verify 16 / 32, `9d9a01e1`). Full verify is softmax-bound: ~907 of ~1,690 cycles per 64-key tile (`f4d1a195`). At the reset: eight softmax variants (25 / 50% of exponentials as an FMA polynomial beside MUFU, row sums on the tensor core, two-warpgroup ping-pong; `dev/full_attention/softmax/`). Sent: the same at decode 16 / 32 full (attn_main, 1.5-1.6x its floor). Next: VQ8's producer publishes K-ready after the K/V copy and V-ready after the V transpose, so QK(j+1) stops waiting on V(j+1)'s transpose (+32 B smem at ST4); q_len=1 full decode as K-major native FP8 `wgmma m64n8k32` (keys or head dim on M, the 8 queries on N); two online-softmax states (even / odd tiles) per consumer warpgroup with `setmaxnreg` moving registers from the producer, merged at the end | Full attention within ~1.3x of its KV floor at every full shape | full-attention lane | active |
| 8 | P11 | G1 | Whole-layer persistent kernel at small M: +1.74% estimated (+2.22% if the whole synchronization tax went); a resident kernel needs competitive inner loops, precise readiness and operands kept across stages, not just fewer launches (`docs/reference/research_megakernel.md`); our persistent dense chain lost 1.5-6.5x (`87dc944b`, mechanism unknown) | Reopen when P7's stream-continuity bound or stage receipts lift the estimate above the rows ranked above it | none | parked: +1.74% estimated; waits for P7's bound |

## Questions for Jihoon
None open.
