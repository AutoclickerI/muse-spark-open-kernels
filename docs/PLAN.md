# Plan

## Status (2026-10-26 17:30 KST, day 5 of 14; the kernel track closes 2026-11-05 16:59 KST)
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
| Cross-stage stream continuity (P7) | data movement | S15 scenario R1.08557 (32.8% of initial gap), not a physical HBM ceiling. Best S18 tested raw gain H2 +0.15% =1.83% of that log bound; G duplicate also +0.13%, so causal gain unresolved | whole-decode paired lab | `73d2d80d`:39 CUDA/25 Triton builds,3456 TASK; ALL R.382568, EDITABLE .991852 lose. H2 1.001504[1.000124,1.002887] and B1 1.001355[1.000064,1.002647] raw G0 nominees to P4; both/G span1. Tested expansion closes; larger MoE/attention/coding first. Retention/opaque-reader structural work remains open |
| Attention at its KV roof (P5 sliding, P5b full) | algorithm | 39%: sliding 20%, full 19%; leader's attention component 1.80x ours (35%) | attention-host run | deciding: native kernels have lost to FA3 at verify sliding; untested: a coordinate-vs-transport oracle (page-coordinate gather issue dominates decode) and fixed-reference softmax with the denominator inside PV |
| MoE M = 128 / 256 at its stream (P3b) | data movement | 31%: 1.081x at v16 / v32 | MoE-host run | parked: mechanism oracle `3330edc2` found no build above 1% of the score |
| Projections below our roof (P10) | data movement | 15% to our roof; 30% at the leader's component time; exact 12-bit coding cuts 25% of dense bytes (FINDINGS "Weight encoding") | proj-host run | measured: gate1 sliding G0 +0.44% (e3643817); broad coding/merge loses or neutral. Raw-register oracle exposes schedule/activation cost beyond decoder; K128 grouped+adjacent-token also loses; re-rank this lever from complete raw/code schedule evidence |
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
| 1 | P4 | G1 | Integration/scored runs. Incumbent S18 `95df42fe` (`cand_r3_best7_s18_final.py`, [dev/s18](../problems/muse_decode_layer/dev/s18/README.md) `frozen.json`): scored `28b5c189` 1.848375x [1.820649, 1.876101]; paired `5d0282bb` (import-time lab on a lab-only prefill host; the decode check was withdrawn unsent) R 1.004321 [1.002314, 1.006332] over S17 `fea71760`, 16 held-out seeds, 1536 TASK + 260 nearby checks. S18 = S17 + P3a SHARD2+INDEX M1, PAIR M8, PAIR+BLOCK M16/32; PDL metadata before wait (M) + d1F glue (D); V-wait removal v16F/v32F; P5 D8 d16S / D4 d32S; S16 pool128. Loads: d1 1.0114, d16 1.0097, d32 1.0012, v1 1.0025, v16 1.0024, v32 0.9989 [.9981, .9997]: S18 loses v32S (S17 at .9948 of its time) and no leave-out recovers it. Leave-outs favour each piece where it acts (BLOCK 1.0021 and pool128 1.0013 with CI > 1; D8 at d16 1.0040, D4 at d32 1.0013, Vwait32 at v32 1.0027). S19 (`dev/s19`, `frozen.json`): `62ea5bf4` lab-confirmed R 1.014463 [1.013044, 1.015885] over S18 (`6ac9317e`, 16 fresh seeds, 672 TASK + 260 nearby; loads d1 .9973 [.9914, 1.0032], d16 1.0116, d32 1.0185, v1 1.0400, v16 1.0017, v32 1.0183). S19 = S18 + P10 exact-12 QKV at T=8, P5 RAW2 v32S, P5b R8 d16F/d32F and D v16F/v32F, P3a FAST M1 (neutral), pdl_M's `_boundary_late` reverted (v32 0.4%), baked prepare 147.7 s (was 254.7 s) and static contract checks (`contracts.py`). Round 1 `7c0fd3cc` (R 1.007437 [1.006506, 1.008368]) dropped P7 W (no_W .99731: loses on S19 at d1/v1), P10 at T=16 (d16 .9881), SPLIT2 M32 and PIPE2. Next: S19's scored decode run at the 10-27 09:01 reset (promotion completes on TASK pass); then S20 from fresh handoffs. | A scored run above the leader's, and our six-load score on many seeds above it; the final candidate correct and not much slower at nearby shapes (other batch sizes and context lengths; the final review reruns them, llms.txt:112) | integration lane | active |
| 2 | P10 | G1 | All36 dense sites measured in dev/dense_all (cc06d34c,8seeds4/7arms,TASK+AA+actual neighbours); GPU coding banks add24.52s incl24.40s compile. Register-A removes decoded staging, but large shapes still lose1.27–2.12. Decoder/MMA overlap (e7b0d578) gains1.3–6.4% within that schedule yet loses1.21–1.95 vs current; matched raw oracle shows next term is K64 synchronization/activation transport/tile order. K128+adjacent-token batch ff32499a also loses1.190–2.092 after5–19% matched-schedule savings; no large-shape handoff. Next structural test preserves native dense tile math/activation transport. Full QKV/gate merge loses G0 R.989549[.986697,.992409]. READY P4 handoff: exact12 fused gate only T1sliding, native held-out G0 R1.004405[1.003003,1.005808] vs S19,16freshseeds576TASK,e3643817; added prepare9.192s,total170.454s,GPU bank3.525ms. patch_gate.py + P4_HANDOFF.json. No T16 QKV change (S19 rejected it). Leader1.36125 is private aggregate component data; physical HBM and updated component residual remain unmeasured. | Every dense stage at its stream roof, or below it through coding | public-proj lane | active |
| 3 | P3a | G1, G2 | Small-M MoE M1-32,54% of gap. S19 62ea5bf4 integer/address receipt6f5afe0c273e4bbf:16seeds128pairs64AA,all TASK/packet/source/resources pass. AFFINE/S19 M8/16/32 1.00083/1.00217/.99703;M1 unchanged alias. PRIVATE/S19 M1/8/16/32 .99706/1.00869/1.01115/1.01568. All2CTAs;PRIVATE119regs M8+ with16Bstack,zero spills;adds226uniform sites despite164-166fewer INT. PRIVATE M1/AFFINE M32 exact S19 patches+SE/CI/seeds toP4;M1alias.99726 andAA1.00472 retained,weak point nominee. Core per16 unchanged17.0625INT,1PRMT,8HMUL2,2HMMA;exact457marker bins273INT/16PRMT/128HMUL2/32HMMA/8LDS plus scaleLDS outside. INT largest modeled demand,not proven hardware saturation. A/B single-affine magic and constant-index zero-shift searches establish no implementation;not measured losses. All4064 products exact. Next by ceiling:exact dynamic-selector/arithmetic decoder with lower counted issue before quota;W2-ready/grid scheduling remains open. Small address nominees batched in P4 R,no standalone sub1% run. Artifacts:dev/nt1_timeline/structural/intpipe/ | Each load's MoE within ~1.3x of its stream | small-M lane | active |
| 4 | P5b | G1, G2 | Full attention (19% of gap). S18+R8+D d53966e4 passes564TASK/8seeds with poisoned outputs/native graph-node proof, A:A all95%CI include1. Fixed max/rescale oracle v16/v32 .9342/.9498; paid native K-only max pass1.424/1.515 (pass71.9/133.3us): this recipe loses, running max/rescale is only5–7% here; exponent/packing remain. One-launch full lastCTA NK64/128 x16..256splits loses every d1/v1 point (best2.061/4.295), serialized row merge implicated; distributed DSM/native-VQ8 remains. C3 frees4KiB and reaches4CTA/SM but only d16F.988939,d32F.990809; C4 .993815/.996489 also to P4, concrete patchb2103b37 onto S18+R8/D, no standalone small-gain run. b9488225 unmeasured: saved stream left custom captures empty; corrected. Next by size: verify exponent/packing, d1/v1 parallel reduction/prologue, and decode coordinate/transport oracle; prefer loops whose ceiling survives these verdicts. | Full attention within ~1.3x of its KV floor at every full shape | full-attention lane | active |
| 5 | P5 | G1, G2 | Sliding20pct of gap. Oraclef3657641 (12seeds,316TASK): verify softmax/packing5.37/5.32us, real merge4.95/4.98us, setup/issue7.27/7.33us, launch~2.1us; no large precomputed-address gain. EXP7bd3eced (16seeds,328TASK) verify loses1.013426/1.008693: probability cycles rise despite SFU sites falling. Decode EXP.996181/.998138 points to P4 S19 with wideCI and d16 AA.988514; no shader/G0 gain established. Global-last-WG3a9a3174 (16seeds,164TASK) losesverify1.087312/1.051840, AA.999772/.999299. Same math/slices, but merge serializes64rows onto1WG: phase21.1/16.7us vsparallel4.5/3.4us. Retain FA3v16/RAW2v32. All receipts done. Next by size: preserve merge parallelism, reduce max/sum/packing and separately query/setup/copy issue; source-coordinate8B oracle does not close hoisting generally. [Evidence/handoff](../problems/muse_attention_decode/dev/sliding_oracle/README.md) | Composed held-out six-load R improves G0;1.3KV targets2.280/4.559us remain open | attention lane | active |
| 6 | P7 | G1, G2 | S18 full boundary cohort `73d2d80d` completes16 fresh paired seeds x8 cold,18 arms/12 shapes/3456 TASK all pass; A:A exact fingerprint. Q .666395, O .970939, pool .613680, unpool .710344, NT1 W13 .987878, W2 .999361, heavy W13 1.000721, b2 .997603, KV .995181; ALL .382568[.379952,.385202], EDITABLE .991852[.990393,.993313] lose. **H2 1.001504[1.000124,1.002887], B1 1.001355[1.000064,1.002647] raw G0 pass**; unchanged G1.001344[1.000094,1.002596], both nominee/G CIs span1, retained as caveat. 39sm90a CUDA/25sm90 Triton builds,146 SASS witnesses; QKV3 smem restores79regs; native OL ABI stand-in proves/fixes GNU-unique smem guard collision plus missing PDL launch. Model R1.08557 remains scenario only: ALL recovers0%, best raw H2 1.83% of log bound. [Handoff/results](../problems/muse_bf16_proj/dev/stream_continuity/extend/README.md): H2/B1 nominees plus readable patches/control evidence to P4 for fresh S19 composition (old W already dropped there). No broader continuity/P11 closure | Every requested boundary ranked and all-boundary recipe judged by held-out G0; P4 owns S19 integration | glue lane | complete; standalone schedules parked |
| 7 | P3b | G1, G2 | Mechanism decided by cheap S18 oracle `3330edc2`,16seeds x3fixtures,all nativeTASK/AA pass. PREPACK/PACK .9191/.9388(gather~15us);EARLY3/PACK3 1.0014/1.0036(same bytes/footprint,no gain);DUPCTA/DUPSYNC .8840/.8166(sync costs~26/56us);MC2/DUPSYNC .9998/.9985(half requests,no resolved net gain). Literal MC2-local is unsafe;own-copy controls isolate sync correctly. Old +39/+100us term was mixed reader granularity/issue/cache attribution,not bandwidth alone. Every timed-gather arm losesBASE. Free-gatherPACK .99567/.99169 gives at most~0.21% optimistic six-load transfer even at100% MoE layer share;no supported >1% next build. No P4 candidates;earlierB4 stays P4 decision. Propose parking by odds/size,retain31% theoretical ceiling,reopen on a new measured mechanism after fresh re-rank. Evidence branches/mechanism/. | A credible >1% six-load case before another structural build | MoE lane | parked: no build above 1% of the score after the mechanism oracle `3330edc2` (gather ~15 us, cluster sync 26 / 56 us; activation bytes are not the cost) |
| 8 | P11 | G1 | Operands retained across stages, up to a whole-layer persistent kernel at small M: synchronization removal alone is +1.74% estimated (+2.22% if the whole synchronization tax went); a resident kernel needs competitive inner loops, precise readiness and operands kept across stages, not just fewer launches (`docs/reference/research_megakernel.md`); our persistent dense chain lost 1.5-6.5x (`87dc944b`, mechanism unknown) | Reopen when P7's stream-continuity bound or stage receipts lift the estimate above the rows ranked above it | none | parked: +1.74% estimated; waits for P7's bound |

## Questions for Jihoon
None open.
