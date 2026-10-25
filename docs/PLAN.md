# Plan

## Status (2026-10-25 11:45 KST)
- G0 (R-20261025-1): the highest score on the overall kernel board by the close (2026-11-05 16:59 KST, day 3 of 14).
  By our fit (GOAL), overall = max(decode layer, component sum), and the decode layer wins for every top-50 team.
- Board (SOTA "Leaderboard", 10:44): 166th of 1,014 at 1.5735x (S7R); leader 2.285x (+9% overnight), 10th 2.066x,
  50th 1.820x. The leader's component boards: attention 2.84x (ours 1.40), MoE 2.52x (1.69), projections 1.61x (1.19).
- Scored today: S1-S3 1.467-1.474x; S4-S6 withdrawn (a 16-bit scale copy made in prepare, CONSTRAINTS 5); S7 1.5597x
  `2f6471d5`; S7R 1.5735x `c1fbd0cf`; S8 1.5742x `a5b2f55b` (+ early release, M=128 qkv split-K, cluster glue: faster
  at 10 of 12 shapes; each load's ratio also moves 2-4% with the baseline's own run time). S9 sent 11:40 (`f55f6162`:
  S8 + BF16 qkv front + N_full at M=128/256 + small-M flags; modeled 1.674).
- What the plan was missing (review of `88b5251` plus the board against our model, 11:30):
  1. Our ceilings sat below the field. The leaders' MoE (2.52x) and decode-layer (2.28x) boards beat our "practical"
     model's ceilings (2.35x, 2.25x; `python -m sim`), the projection leader runs at 1.255x its floor (~3.82 TB/s
     effective), and our own 52-61 MB sweep read 4.04-4.25 TB/s. The 3.61 TB/s plain read (`4233660b`) is one
     kernel's rate, not the roof. Target: every stage at <= 1.3x its floor (2.59 modeled), byte-bound stages at the
     judge's best measured stream (every stage at 1.15x: 2.94).
  2. Lossless coding of the dense weights was parked on "no byte-bound stage yet", which no longer holds: S8's GEMVs
     run within 16-20% of a plain read of their bytes (`9cd95e72`). The dense weights are randn/sqrt(k) in BF16
     (`reference.generate_weights`): 2.545 bits of exponent entropy, and 16 contiguous exponents cover 99.993% of
     weights, so a 12-bit code with exact escapes cuts 25% of the ~138 MB per decode-1 layer. Modeled +4 to +8%
     (time above a 0-4 us fixed part scales with bytes); the only lever that lowers a floor. Now P10, active.
  3. Our rule readings were narrower than the rules (CONSTRAINTS "Boundary cases" widened): bytes per 16 weights was
     our audit, not a limit (duplicate layouts and padding are fine; widened values are not); routing is checked by
     membership and score, in any order; FP8 K/V may expand exactly to BF16 on chip; value-dependent work inside
     kernels is fine; the router's hi + lo pair is allowed at any batch (llms.txt:107).
  4. Ten code-level items, in the rows. MoE (P3b): the W2 vector loader and device-side N16/N32 lists are already in
     N_full (S7R's scalar W2 copies were R_full's, now deleted); W13 gate/up interleave ~1-3 us and W2 norm partials
     < 1-2 us at M=128; NT1 has neither exchange. Others: FP32 exponentials in the native
     path (still `ex2.approx.f16x2`, verified), LDSM.T V transpose, register-resident P for RS WGMMA, FP8-QK + BF16-PV
     (P5); gate projection overlapped with attention (P7); the QKV sum+cast fusion is S9's BF16 front.
- Limits (`problems/muse_decode_layer/dev/levers/scenarios.py`, S7R plus the paired wins S8 and S9 carry, modeled
  1.674; the model reproduces every S run within ~1.5%): every stage at 2.0 / 1.5 / 1.3 / 1.15x its floor gives
  1.96 / 2.26 / 2.59 / 2.94; today's leader is every stage at ~1.5x.
- Gain per lever on that model at 1.5x / 1.3x of its floor: attention +9.5 / +10.8% (sliding +5.2%, full verify
  +2.7 / +3.2%), small-M MoE +5.8 / +9.4%, glue +5.9%, projections +1.9 / +5.9%, M=256 MoE +3.5 / +4.6%, M=128 MoE
  +1.3 / +2.5% (independent counterfactuals; they don't add); lossless dense coding +4 to +8% on top of the
  projection row. Inside each row, deleting provably dead or repeated work comes first.
- Attention inside the layer (S7, us per layer, sliding / full): decode 1 10.2 / 19.4, decode 16 16.3 / 104.1, decode
  32 16.7 / 191.2, verify 1 20.7 / 31.8, verify 16 22.5 / 199.1, verify 32 24.5 / 381.4. Sliding sits 3-200x its floor
  in 46 layers per load (5 kernels at verify); full decode is at 1.5-1.65x its floor, full verify at ~3x.
- Agents: Codex is out of quota (Hermes `local_auth_cap` on every credential since 02:05 KST, still at 10:58; RESOURCES
  "Inventory"), so lanes run as Claude subagents: integration (P4), attention (P5), glue/proj (P7), small-M (P3a),
  dense coding (P10), MoE M=128/256 (P3b); P11 is parked.
- Promotion: by the six-load score on identical inputs and its spread across seeds (GOAL); a stage's seed mean is a
  diagnostic, and our exact-ID or 0.003 gates are not TASK.md's checks (CONSTRAINTS "Boundary cases").
- Bottleneck: judge latency (our runs start one at a time) and run-to-run noise (MoE routing per seed): compare
  variants in paired labs, not single scored runs.

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
| 1 | P4 | G1 | Integration and scored runs. S8 1.5742x (`a5b2f55b`) is the incumbent; S9 (`f55f6162`, sent 11:40) = S8 + BF16 qkv front + N_full at M=128/256 + small-M flags, modeled 1.674. S10: LT and the layer attention ext compiled in the native pool from prepare's start (~80 s off a 230 s cold prepare locally; S8 took 215 s on the judge), plus each landed paired win; the coded dense GEMVs (P10) need that headroom. Audit what each prepare() tensor holds at what precision (CONSTRAINTS 5). Decode's 10 scored runs a day carry candidates; its 20 checks carry whole-layer A/B of single changes on the same day's inputs | A scored run above the leader's (2.285x, rising), and our six-load score on many seeds above it too | integration lane | active |
| 2 | P5 | G1, G2 | Attention (+9.5 / +10.8%; the leaders' attention boards run ~2x ours). Sliding attention costs 10-25 us per layer against floors of 0.1-3.7 us in 46 of 62 layers (+5.2%); full verify sits at ~3x its floor (FA3 199 / 381 us at verify 16/32; +2.7 / +3.2%); full decode is at 1.5-1.65x. vq8 (persistent, stream-K, native FP8 WGMMA, PDL combine with norm/gate) wins every decode shape on the attention board (1.66-2.07x vs SGLang, `4aa070c6`) but only matches FA3 at verify. Native q8 verify8 lost 1.3x to FA3: per 64-key tile V transpose + barrier ~990 cycles, softmax ~770, P store 150 (`a1679f9c`). Closed recipes: q1 conditional rescale (`9828bfa0`); single-launch sliding as built (wins only verify 1). Next: (1) a sliding block that wins at every load: q/k prep, the 2,048-key gather and norm/gate in one launch with no metadata kernel; (2) vq8's decode-shape kernels into the layer where they beat S9's path in a paired lab; (3) native verify: FP32 `ex2` in place of `exp2pair`'s `ex2.approx.f16x2` (16 F2FP + 32 MUFU.EX2.F16 + 32 HADD2.F32 per tile) with the 256 x 2^(x-m) offset, then V transposed by LDSM.T + PRMT + STSM with a V channel permutation, P kept in registers for RS WGMMA (FA3 `permute_Aregs_fp8`); alternative: FP8 QK + BF16 PV with V expanded exactly on chip, no byte transpose; (4) FA3's split combine with the norm/gate epilogue; (5) parked: FP8-P staging. All 20 attention runs a day go to candidates | Full attention within ~1.3x of its KV floor and sliding <= 6 us per layer at every load | attention lane | active |
| 3 | P7 | G1, G2 | Glue (+5.9%) and projections (+1.9 / +5.9%). In S8/S9: attention early release (`4d995f13`), M=128 verify qkv split-K (`b15a3bde`), cluster glue at t <= 32 (`1365a290`), BF16 qkv front (`87cd3a84`: -4.2 / -4.5 us per decode-1 layer). S8's GEMVs sit 16-20% above a plain read of their bytes (qkv 12.6 vs 10.8 us, pool_router 22.9 vs 18.9, `9cd95e72`); cuBLAS at t=32 is worse (+12-43%). Closed recipes: verify split-K beyond M=128 qkv, bulk-ring GEMVs standalone (`87b2667f`), attn_main reading split partials (`aab93d73`), FA3 metadata on a side stream (`e1a56ee7`), cooperative glue (`9eeb39ba`), CUDA-core warp-per-row GEMV, skipping zero-local rows. Next: (0) the judge's stream roof: the best cold-read recipe (TMA bulk vs LDG / cp.async, bytes in flight per SM, 132-1056 CTAs) at each dense size (8.4-75.5 MB) with `lab_cold_l2`; it sets every byte-bound target; (1) GEMVs at that roof: balanced split-K (`26fd09a7`: qkv -0.9 to -3.0 us, pool_router -1.6 to -3.0, unpooler -0.6; ~-3 / -3 / -7 us per decode 1 / 16 / 32 layer, chain pairing in flight) into S10, then the 0.6-3 us left; (2) the gate split: a PDL gate GEMV (1,024 of 2,304 qkv rows, 16 MiB) streams beside attention and applies sigmoid to its output, with a no-gate mode patched into attention's epilogues and a trigger where attention lacks one (~-4.6 us per decode layer, est.; qkv alone 11.6 -> ~7 us; stream roof `2b3573ba` and chain `c2b2da75` in flight); (3) MoERouteMeta from the cluster route kernel for the small-M consumer. Too small to build (analysis): an unsorted top-8 (route_cl's integer-key top-8 already takes ~0.3 us), router beside the pooler (<= 0.5-1 us), the T <= 8 zero tail (< 0.3 us) | Projections at the measured roof and glue <= 6 us per layer | glue/proj lane | active |
| 4 | P3a | G1, G2 | Small-M MoE, M=1-32 (+5.8 / +9.4%). H200 timeline of the M=1 call (`849e4017`): node 31.8 us = launch 1.0 + CTA span 32.2 + drain 1.3 (route 3.0, norm-zero wait 4.1, W13 10.0, W2-ready wait 6.4, W2 4.2, barrier 2.6, combine 1.3). Flags in S9 (`cf1de160`, confirmed `eb9e0382`): M=1 95 -5.58 us of 42.3; M=8/16/32 69 -4.11 / -4.83 / -4.46. Capping every local expert at 8 tokens (diagnostic) is 9.3 / 34.5 us faster at M=16 / 32: each extra 8-token block re-decodes its expert's weights (~7.4 us, from L2). Closed recipes: the dynamic M=1 schedule (1119: -1.78 vs 95's -5.68 us on the H200, `eb9e0382`; an L2 atomic ahead of each tile copy slows W13), route metadata from the router's epilogue, a last-CTA finalizer, router-epilogue L2 prefetch, the joint E2M1 x E4M3 table decoder. Left against a pure stream over the same 56.6 MB at M=1: 11.5 us (route 1.5, first tile 3.7, W2-ready wait 4.7, finalizer ~2.4). Next: (1) decode each weight tile once for all of a busy expert's tokens at M=16/32 (bounded at ~9 / ~35 us per call); (2) W2-ready waits at M=32 (11-13 us on the slowest CTA); (3) metadata from the cluster route kernel (NT1 has no gate/up exchange or norm barrier to remove: its gate and up meet after the K reduction, W2 partials use atomics) | Each load's MoE within ~1.3x of its floor | small-M lane | active |
| 5 | P10 | G1 | Lossless coding of the dense BF16 weights (TASK.md:102: "or smaller, losslessly"): one byte of sign + mantissa plus a 4-bit exponent index (16 contiguous exponents cover 99.993% of randn/sqrt(k) weights; entropy 2.545 bits, so 10.5 bits per weight is the ideal), exact escapes for any other value, decoded in the GEMV: -25% of the ~138 MB per decode-1 layer, modeled +4 to +8%. Integer issue per weight must stay under its HBM time. Then denser codes if the 12-bit one measures | Bit-exact on any weights; each dense stage's time falls with its bytes in a whole-layer paired lab | dense-coding lane | active |
| 6 | P3b | G1, G2 | MoE at M=128/256 (+1.3 / +2.5% and +3.5 / +4.6%). N_full (NT1 for <=16-token experts on a side stream beside N32 WGMMA on the heavy list) is in S9: 0.8842x H_H at M=128 (`c01b1916`), 0.8701x mixed_parallel at M=256 (`14f942d5`), bit-equal; tuning lab `880c3881` landed (read its lines). Heavy branch on the forked stream, launched first: 0.9799 / 0.9882 (`880c3881`; priority and grid changes add nothing), folded into the cell (dfb3d78a, 72/72 bit-equal). Typed-smem WGMMA (HH_u8) 0.9702x; the in-register scale word costs 0.7%. Lab fixtures touch 16.1 / 18.2 of 32 local experts at M=128 / 256: N_full runs at 2.1x / 2.8x the HBM floor of those unique bytes (95 / 107 us), and ~71 SASS instructions per 16 weights put the M=128 decode issue alone at ~107 us, the size of the byte floor (estimate from instruction counts, not counters). Next: (1) fewer instructions per weight on the heavy WGMMA path (dequant and scale path); (2) one decode per expert tile at M=256, where heavy experts decode each tile once per 32-row group, without the producer/consumer overheads (that recipe lost 1.13-1.46x, `9cd95e72`, mechanism not isolated); (3) W13 gate/up rows interleaved in prepare so SwiGLU finishes in registers (no epilogue smem round trip, one barrier fewer, bit-exact; ~1-3 us at M=128); (4) W2 per-warp norm partials summed by the finalizer (< 1-2 us); (5) typed-smem WGMMA in N_full's heavy branch | Each within ~1.3x of its floor | MoE lane | active |
| 7 | P11 | G1 | Whole-layer persistent kernel (megakernel) at small M. Scoped op by op on the reset profile (`problems/muse_decode_layer/dev/megakernel/README.md`): +1.74% on the six-load score (+2.22% if the whole synchronization tax went). Prior art: Hazy "No Bubbles" 1.5x over SGLang at batch 1, Cohere 1.58x, MPK 1.0-1.7x (SOTA); our persistent dense chain lost 1.5-6.5x (`87dc944b`, mechanism unknown) | Reopen when a stage receipt or boundary evidence lifts the estimate above the rows ranked above it | none | parked: +1.74% estimated |
