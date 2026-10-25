# Plan

## Status (2026-10-25 18:10 KST)
- G0 (R-20261025-1): the highest score on the overall kernel board by the close (2026-11-05 16:59 KST, day 3 of 14).
  By our fit (GOAL), overall = max(decode layer, component sum), and the decode layer wins for every top-50 team.
- Board (decode, 18:10): 172nd at 1.6073x (S9); leader 2.285x (flat since 09:27), 10th 2.11x, 50th 1.871x;
  S13 final's host 1.7186x would place ~106th. The leader's component boards: attention 2.84x (ours 1.59), MoE 2.52x
  (1.69), projections 1.61x (1.19); a public MIT proj submission (FINDINGS "Projections") runs 0.62-0.96x our time.
- Runs (FINDINGS "Our kernels on the H200"): S8 1.5742x, S9 1.6073x `b7c59570` (on the board), S10 check 1.6214x
  `43049b80` (our load time -2.6% vs S8 with the baseline flat; prepare 133.5 s). Scored runs don't rank changes under ~3%: the whole-layer paired A/B on S10
  (`ff3b9aca`) is the comparator, and it confirmed S9's small-M flags (+1.7-5.4% if removed) and the BF16 front
  (+3.2% at decode 16).
- S11 (`dbcb26bb`: S10 + balanced GEMVs + LI_swap N_full + vq8 page ring at verify 16 / 32 full + pinned tuner
  choices) beats S10 by ~4.4% geomean in a paired whole-layer A/B (`e42953c4`: +3.7 / +2.8 / +4.0 / +1.7 / +5.2 /
  +9.4% per load with S10's pieces back); its host run judged 1.6917x. The final S11 `acda6c62`
  (`cand_r3_best7_s11_final.py`; + gate split + late glue) wins every leave-one-out arm (`863f502c`: S10's pieces back
  +1.9 to +9.95% per load; host run 1.7068x). S12 final `240248b2` (S11 + coded GEMV at t=1 + finalizer split;
  `937801fe`, host 1.7148x). S13 final `958f1258` (S12 final + direct-partials attention at decode 1, paired 1.0377,
  + PERMW at M=256, pooled 1.0038; `cb38a050`, host 1.7186x) goes scored at the 00:00 UTC reset (09:00 KST).
- What the plan was missing (review of `88b5251` plus the board against our model):
  1. Our ceilings sat below the field (now measured: see "Ceilings" and "Where the time goes" below).
  2. Lossless coding of the dense weights (P10) was parked on a stale premise: the GEMVs now sit within 16-20% of a
     plain read. 12-bit codes cut 25% of the ~138 MB per decode-1 layer (+4 to +8% modeled; the only lever that
     lowers a floor); the first coded GEMV is decode-bound (`35c875e6`).
  3. Our rule readings were narrower than the rules (CONSTRAINTS "Boundary cases" widened: precision, not bytes per
     16 weights; routing by membership in any order; FP8 K/V expanded on chip; value-dependent work in kernels).
  4. The review's code items are in the rows (P3b, P5, P7); those already done or small are recorded there.
- Ceilings on S11 (`dev/levers/current.json`, `863f502c` profile; per-layer us at decode 1 / 16 / 32, verify 1 / 16 / 32):
  ours 95.7 / 179 / 243 / 139 / 333 / 462 = 1.707; every stage at its own stream roof (2.36 us + bytes at 4.58
  TB/s each) 2.79; one continuous stream per layer 49.3 / 112 / 141 / 80 / 179 / 228 = 3.08; the 4.8 TB/s floor
  3.31. The leader (2.285) sits at ~1.35x the continuous stream, we at ~1.8x. Most of our gap is inside stages
  (decode 1: 35 us of in-stage excess over the stage roofs, 12 us of per-stage fixed cost; verify 32: 231 us
  in-stage, mostly MoE M=256 and full verify attention). Latency-bound stages with almost no bytes (sliding
  attention, boundaries, route, MoE waits) cost ~25-30 us per decode-1 layer that a continuous weight stream could
  hide.
- Where the time goes (S11 H200 timeline `c6bd22e2`, `problems/muse_decode_layer/dev/timeline/README.md`): no-kernel
  gaps are 0.5-2% of wall, so launches are not the loss. Critical-path time above the byte roof, share of wall at
  d1 / d16 / d32 / v1 / v16 / v32: attention 25 / 13 / 12 / 19 / 12 / 13% (sliding increments 16-30 us per layer
  on 0.6-19 MB; full verify at 49-56% of roof; full decode at 66-73%, near the 2.56 TB/s page-row gather
  microbenchmark), MoE 6* / 12 / 10 / 16 / 20 / 18% (a flat 2.0-3.0 TB/s at every M; *no local experts in the d1
  fixture), glue b1 + route + b2 13 / 5 / 3 / 8 / 5 / 4%, dense projections 8-13% (near their stream roofs at
  small M). Literature (`docs/reference/research_megakernel.md`, `research_deepinfer.md`): fusion alone is ~+2.8%;
  the jump has to come from inside the stages. Lanes now work structural targets: one-launch sliding attention
  <= 6-10 us and full verify at decode full's rate (P5); MoE issue vs memory-pipeline diagnosis, then a rebuild to
  >= 4 TB/s (P3a, P3b); glue folded into producer epilogues, <= 1 us per step (P7); lossless coding on every dense
  role (P10).
- Agents: Codex lanes on gpt-6.1-sol: integration (P4), public-proj port (P10), sliding and decode-1 attention (P5),
  full attention at decode 16 / 32 and verify 16 / 32 (P5), glue (P7), small-M (P3a), MoE M=128/256 (P3b); P11 is
  parked.
- Promotion: by the six-load score on identical inputs and its spread across seeds (GOAL); our exact-ID or 0.003
  gates are not TASK.md's checks (CONSTRAINTS "Boundary cases").

## Critical path
Kernel loop, measured 2026-10-23:
1. A lane edits a candidate or a lab payload.
2. Local screen (`tools/local_eval.py`, sm_120): tests take 1-5 min; local timing ranks loosely.
3. `okc_run.py send` or `h200_lab.py send`: the queue wait is our queue position x ~6.6 min (FINDINGS "Judge queue":
   slots per run, not per minute), and each run takes 1-5 min. A decode candidate goes as a scored run while the
   day's last, so a winner lands on the board in this one run (no check-then-promote).
4. Read the receipt and per-shape profile; the decode lane integrates winners into the next scored candidate.

Step 3 dominates, since one result costs hours of wall time.
- Fixes: dense labs (many variants and seeds per run), priority queueing, Hopper glue exercised locally (AGENTS),
  labs on hosts with quota left (synthetic weights where the host lacks them), batching payloads per host.
- A lab records a failed cell (route flip, gate miss) and keeps timing the rest; one raise must not end the run.
- The receipt keeps only the last 16,000 characters of output (`log_tail`), so print compact lines last.
- Lanes run in parallel, one per stage plus integration. A lane never idles on the judge; `collect` lands receipts
  while it builds the next variant.
- Decode's 10 scored runs went by 11:30 KST on 10-25 (S1-S9 plus withdrawals); its 19 checks carry the paired
  whole-layer A/Bs until the 09:00 KST reset.
- Quota is capacity: each day gives 80 lab runs (proj 10 + 20, MoE 10 + 20, prefill 10 + 10) and 20 runs each on
  decode and attention. On 10-24 the lab hosts and decode were full but attention used 4 of 20: attention has no
  `prepare` to host labs, so its variants go as attention candidates (each run profiles all 12 shapes), and decode's
  runs beyond the day's S-candidates carry whole-layer variants.

## Ranked work
Targets come from the floor model above: each stage at <= 1.3x its floor, byte-bound stages at the judge's measured
stream roof, judged by whole-layer time (lever table in `problems/muse_decode_layer/dev/NEXT_CHECKS.md`). A measured loss closes only the recipe tested; a result
from a lab whose control failed, or a case our own gate skipped, is unmeasured, not a loss.

| Rank | Id | Serves | Work | Done when | Who | State |
|---|---|---|---|---|---|---|
| 1 | P4 | G1 | Integration and scored runs. S9 1.6073x (`b7c59570`) is on the board. S13 final `958f1258` (`cand_r3_best7_s13_final.py`: S12 final + direct-partials attention at decode 1 + PERMW at M=256; every piece wins its leave-one-out arm, `cb38a050`, host 1.7186x) is the 00:00 UTC scored send. S14 onto it: N16 light experts (`eea9d87b`, MoE 0.933 / 0.949 at M=128 / 256), P10's direct-BF16 qkv3 / o patches, bal2 pool_router. S15: the public proj port (P10). Audit what each prepare() tensor holds at what precision (CONSTRAINTS 5). Decode's 10 scored runs a day carry candidates; its 20 checks carry whole-layer A/B of single changes on the same day's inputs | A scored run above the leader's (2.285x), and our six-load score on many seeds above it too | integration lane | active |
| 2 | P10 | G1 | Dense projections (8-13% of wall above roof). NathenFernandes's judged v18 proj kernels (MIT, R-20261025-2) take 0.62-0.96x our time at every proj shape with the same 12-bit coding (FINDINGS "Projections"; ~+5-7% est.): port them behind S13 final's projection call sites with the MIT notice, keeping our fusions (pool_router, gate split, t=1 coded) only where a paired lab says they win per shape (`problems/muse_bf16_proj/dev/public_nf/`). Ours so far: 12-bit coded GEMV in S12 at t=1; direct-BF16 decoder qkv3 / o patches screened (`a645a795`); RS-WGMMA coded wins at t=16 (`04d89c7d`). Next after the port: denser codes (low byte + 4-bit exponent index has 2.545 bits of entropy, 10.5 bits per weight ideal) on their kernel structure | Every dense stage at or below their judge time per shape, bit-exact, in a whole-layer paired lab | public-proj lane | active |
| 3 | P5 | G1, G2 | Attention (+9.5 / +10.8%; the leaders' attention boards run ~2x ours). Sliding attention costs 10-25 us per layer against floors of 0.1-3.7 us in 46 of 62 layers (+5.2%); full verify sits at ~3x its floor (FA3 199 / 381 us at verify 16/32; +2.7 / +3.2%); full decode is at 1.5-1.65x. vq8 (persistent, stream-K, native FP8 WGMMA, PDL combine with norm/gate) wins every decode shape on the attention board (1.66-2.07x vs SGLang, `4aa070c6`) but only matches FA3 at verify. Native q8 verify8 lost 1.3x to FA3: per 64-key tile V transpose + barrier ~990 cycles, softmax ~770, P store 150 (`a1679f9c`). Closed recipes: q1 conditional rescale (`9828bfa0`); single-launch sliding as built (wins only verify 1). Measured: vq8 wins in the layer's chain only at verify-1 sliding (B1 q_len 8: 14.40 vs 17.47 us, `f0a79d79`; into S11; at decode-1 sliding attn_main 14.0 vs vq8 19.8 us, `d5bcbc7d`); its producer and consumer serialize (2,700-2,900 cycles per tile; a publish-first order with LAG <= ST-2 models ~1,400). The publish order fixed (LAG = ST-2, `44aa67ba`): verify 16 / 32 full 153.9 / 278.5 vs FA3 192.2 / 381.4 us (-20 / -27%, ~+1.4% score), verify-1 sliding 15.3 vs 16.8; cell `patch_vq8_verify.py` b6125185 into S11. Next: (0) per-phase clocks at full (1,800 cycles per tile vs the 1,400 model) and the decode 16/32 full rows against attn_main (104 / 191 us); (1) a sliding block that wins at every load: q/k prep, the 2,048-key gather and norm/gate in one launch with no metadata kernel; (2) vq8's decode-shape kernels into the layer where they beat S9's path in a paired lab; (3) native verify: FP32 `ex2` in place of `exp2pair`'s `ex2.approx.f16x2` (16 F2FP + 32 MUFU.EX2.F16 + 32 HADD2.F32 per tile) with the 256 x 2^(x-m) offset, then V transposed by LDSM.T + PRMT + STSM with a V channel permutation, P kept in registers for RS WGMMA (FA3 `permute_Aregs_fp8`); alternative: FP8 QK + BF16 PV with V expanded exactly on chip, no byte transpose; (4) FA3's split combine with the norm/gate epilogue; (5) parked: FP8-P staging. All 20 attention runs a day go to candidates | Full attention within ~1.3x of its KV floor and sliding <= 6 us per layer at every load | attention lane (sliding, decode 1); full-attention lane (decode 16 / 32, verify 16 / 32) | active |
| 4 | P7 | G1, G2 | Glue: b1 + route + b2 is 13 / 5 / 3 / 8 / 5 / 4% of wall above the byte roof at d1 / d16 / d32 / v1 / v16 / v32 (projections moved to P10). In the layer: attention early release (`4d995f13`), cluster glue at t <= 32 (`1365a290`), BF16 qkv front (`87cd3a84`), gate split where it wins (the concurrent gate GEMV shares HBM with attention, `0a48a058`), late glue (S11). The first producer-epilogue glue (last-split statistics, distributed b1 / b2 apply, fused routing metadata; `problems/muse_bf16_proj/dev/producer_glue/`) loses 1.20-1.29x (`f21c60b0`); fixed hybrid lab `15abb9c1` pending. Next: the mechanism of that loss from per-role CUPTI timelines before the next structural variant; MoERouteMeta from the cluster route kernel for the small-M consumer. Closed recipes: ramp hiding at attention -> O -> b1 -> pool_router (`cc718d3f`), cooperative glue (`9eeb39ba`), FA3 metadata on a side stream (`e1a56ee7`), own verify-dense GEMMs with norm statistics (`d4cbf3ca`), skipping zero-local rows. Too small to build: an unsorted top-8 (~0.3 us now), the T <= 8 zero tail (< 0.3 us) | Glue <= 6 us per layer | glue lane | active |
| 5 | P3a | G1, G2 | Small-M MoE, M=1-32 (+5.8 / +9.4%). H200 timeline of the M=1 call (`849e4017`): node 31.8 us = launch 1.0 + CTA span 32.2 + drain 1.3 (route 3.0, norm-zero wait 4.1, W13 10.0, W2-ready wait 6.4, W2 4.2, barrier 2.6, combine 1.3). Flags in S9 (`cf1de160`, confirmed `eb9e0382`): M=1 95 -5.58 us of 42.3; M=8/16/32 69 -4.11 / -4.83 / -4.46. Capping every local expert at 8 tokens (diagnostic) is 9.3 / 34.5 us faster at M=16 / 32: each extra 8-token block re-decodes its expert's weights (~7.4 us, from L2). Closed recipes: the dynamic M=1 schedule (1119: -1.78 vs 95's -5.68 us on the H200, `eb9e0382`; an L2 atomic ahead of each tile copy slows W13), route metadata from the router's epilogue, a last-CTA finalizer, router-epilogue L2 prefetch, the joint E2M1 x E4M3 table decoder. Left against a pure stream over the same 56.6 MB at M=1: 11.5 us (route 1.5, first tile 3.7, W2-ready wait 4.7, finalizer ~2.4). Closed: decode-once in registers (NT2 +11.8 / +7.9 us, W16 +19.0 / +17.2 at M=16/32, `c768a8e3`: per-tile latency at 16 warps/SM). Two-round schedules RND / ROT lose +4 to +9 us (`17512d86`, closed recipes: the wait moves to the barrier). Next: (1) the M=32 W2-ready wait and grid barrier (~12 us on the critical CTA: the expert-completion publish under a saturated memory system and every expert finishing together, `c768a8e3`); (2) metadata from the cluster route kernel (NT1 has no gate/up exchange or norm barrier to remove: its gate and up meet after the K reduction, W2 partials use atomics) | Each load's MoE within ~1.3x of its floor | small-M lane | active |
| 6 | P3b | G1, G2 | MoE at M=128/256 (+1.3 / +2.5% and +3.5 / +4.6%). N_full (NT1 for <=16-token experts on a side stream beside N32 WGMMA on the heavy list) is in S9: 0.8842x H_H at M=128 (`c01b1916`), 0.8701x mixed_parallel at M=256 (`14f942d5`), bit-equal; tuning lab `880c3881` landed (read its lines). Heavy branch on the forked stream, launched first: 0.9799 / 0.9882 (`880c3881`; priority and grid changes add nothing), folded into the cell (dfb3d78a, 72/72 bit-equal). Typed-smem WGMMA (HH_u8) 0.9702x; the in-register scale word costs 0.7%. Lab fixtures touch 16.1 / 18.2 of 32 local experts at M=128 / 256: N_full runs at 2.1x / 2.8x the HBM floor of those unique bytes (95 / 107 us), and ~71 SASS instructions per 16 weights put the M=128 decode issue alone at ~107 us, the size of the byte floor (estimate from instruction counts, not counters). Lean K loop (one descriptor/pointer update per stage, scale prep hoisted): 0.9613 / 0.9426 at M=128 / 256, bit-equal (`858ff2fa`), into S11; 207 SASS per stage of which the dequant core is 128 (32 per 16 weights). Closed: decode-once per expert tile in N64 form (1.02 / 0.98 vs N_swap, 4-6% worse than the lean N32 loop). Gate/up rows interleaved by 8 in prepare (LI_swap, in S11): 0.9923 / 0.9838 (`701d2867`). Finalizer split over 4 blocks per token 0.9850 / 0.9885 (`5e197b5e`), for S12. The branches barely overlap (heavy alone 99.8 / 177.7 us, NT1 alone 127.6 / 133.7, together 195.0 / 275.0): heavy fills every SM's shared memory and NT1 waits. Co-residency of the branches loses 1.02-1.40 (`f976b09d`, closed recipe); wide banks (3 LDS.128 per stage) 0.9936 / 0.9900 and the finalizer split go to S12. Next: (1) the remaining ~79 non-dequant instructions per stage and the dequant core itself; (2) one decode per expert tile at M=256 in another form (the producer/consumer recipe lost 1.13-1.46x, `9cd95e72`); (3) W13 gate/up rows interleaved in prepare so SwiGLU finishes in registers (no epilogue smem round trip, one barrier fewer, bit-exact; ~1-3 us at M=128); (4) W2 per-warp norm partials summed by the finalizer (< 1-2 us); (5) typed-smem WGMMA in N_full's heavy branch | Each within ~1.3x of its floor | MoE lane | active |
| 7 | P11 | G1 | Whole-layer persistent kernel (megakernel) at small M. Scoped op by op on the reset profile (`problems/muse_decode_layer/dev/megakernel/README.md`): +1.74% on the six-load score (+2.22% if the whole synchronization tax went). Prior art: Hazy "No Bubbles" 1.5x over SGLang at batch 1, Cohere 1.58x, MPK 1.0-1.7x (SOTA); our persistent dense chain lost 1.5-6.5x (`87dc944b`, mechanism unknown) | Reopen when a stage receipt or boundary evidence lifts the estimate above the rows ranked above it | none | parked: +1.74% estimated |
