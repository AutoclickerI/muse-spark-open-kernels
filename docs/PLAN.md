# Plan

## Status (2026-10-25 10:50 KST)
- G0 (R-20261025-1): the highest score on the overall kernel board by the close (2026-11-05 16:59 KST, day 3 of 14).
  By our fit of the ranks (FINDINGS "Boards"), every top-50 team's overall score is its decode-layer speedup.
- Board (SOTA "Leaderboard", 10:40): 173rd of 1,014 at 1.5597x (S7, before S7R); leader 2.285x, 10th 2.066x, 50th 1.820x. The
  leader's component boards: attention 2.84x (ours 1.40), MoE 2.52x (1.69), projections 1.61x (1.19).
- Incumbent S7R 1.5735x `c1fbd0cf` (S7 = S6 with H_H on uint8 scales, 1.5597x `2f6471d5`; + M=256 on R_full).
  S4-S6 were withdrawn for a 16-bit scale copy made in prepare (CONSTRAINTS 5). Runs vary up to ~4.5% per shape on
  unchanged paths, so promote from paired labs.
- Limits (`problems/muse_decode_layer/dev/levers/scenarios.py` on S7; the model reproduces every S run within ~1.5%):
  every stage at 2.0x its floor gives 1.96, at 1.5x 2.30, at 1.3x 2.64; today's leader is every stage at ~1.5x. A plain
  stream reads 256 MiB cold at ~3.61 TB/s on the judge (`4233660b`), 1.33x the 4.8 TB/s floor, so 1.3x is the
  practical limit for byte-bound stages.
- Gain per lever on S7 at 1.5x / 1.3x of its floor: attention +10.1 / +11.3% (sliding +5.7%, full verify +2.5%),
  glue +7.0%, small-M MoE +6.6 / +9.9%, M=256 MoE +6.0 / +7.2% (R_full takes ~0.9 of it), M=128 MoE +2.5 / +3.6%,
  projections +2.0 / +5.8% (independent counterfactuals; they don't add). Inside each row, deleting provably dead or
  repeated work comes first.
- Attention inside the layer (S7, us per layer, sliding / full): decode 1 10.2 / 19.4, decode 16 16.3 / 104.1, decode
  32 16.7 / 191.2, verify 1 20.7 / 31.8, verify 16 22.5 / 199.1, verify 32 24.5 / 381.4. Sliding sits 3-200x its floor
  in 46 layers per load (5 kernels at verify); full decode is at 1.5-1.65x its floor, full verify at ~3x.
- Agents: Codex ran out of quota at 02:05 KST (Hermes `local_auth_cap` on every credential; RESOURCES "Inventory"),
  so lanes have run as Claude subagents since 09:20, one per row below (P11 parked).
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
Targets come from the floor model above: each stage within ~1.3-1.5x of its floor, judged by whole-layer time
(sensitivity table in `problems/muse_decode_layer/FOCUS.md`). A measured loss closes only the recipe tested; a result
from a lab whose control failed, or a case our own gate skipped, is unmeasured, not a loss.

| Rank | Id | Serves | Work | Done when | Who | State |
|---|---|---|---|---|---|---|
| 1 | P4 | G1 | Integration and scored runs. S7R 1.5735x (`c1fbd0cf`) is the incumbent; S8 = S7R + p7c (glue lane) and S9 = S8 + the small-M flags + M=256 N_full (integration lane) are next. Audit every cell's prepare() tensors (dtype, bytes per 16 weights) before it enters a candidate (FINDINGS "Failures"). Spend decode's 20 checks a day on whole-layer A/B of each step | A scored run above the leader's (2.285x, rising), and our six-load score on many seeds above it too | integration lane | active |
| 2 | P5 | G1, G2 | Attention (+10.1 / +11.3%; the leaders' attention boards run ~2x ours). Inside the layer (Status) sliding attention costs 10-25 us per layer against floors of 0.1-3.7 us in 46 of 62 layers (+5.7% at 1.5x floor); full verify sits at ~3x its floor (FA3 199 / 381 us at verify 16/32; +2.5%); full decode is at 1.5-1.65x. vq8 (one persistent CTA per SM, stream-K over request/head pair/tile, native FP8 WGMMA, PDL combine with norm/gate) wins every decode shape on the attention board (1.66-2.07x vs SGLang, `4aa070c6`) but only matches FA3 at verify (0.93-0.99x). Native q8 verify8 lost 1.3x to FA3: V transpose + barrier and softmax take ~70% of each tile (`a1679f9c`). Closed recipes: q1 conditional rescale (`9828bfa0`); single-launch sliding as built (wins only verify 1). Next: (1) a sliding block that wins at every load: q/k prep, the 2,048-key gather and norm/gate in one launch with no metadata kernel, judged by decode-layer time per load; (2) vq8's decode-shape kernels into the layer where they beat S7R's path in a paired lab; (3) verify full: the V transpose off the consumer's critical path and the softmax cost, then the review items (gather warps, split counts matched to residency, QK issued before the transpose); (4) FA3's split combine with the norm/gate epilogue; (5) parked: FP8-P staging. Use all 20 attention runs a day as candidates | Full-layer attention within ~1.5x of its KV floor and sliding <= 6 us per layer at every load | attention lane | active |
| 3 | P7 | G1, G2 | Glue (+7.0%) and projections (+2.0 / +5.8%). Ready in p7c / S8 (paired H200 chains): attention early release (`launch_dependents` at attn_main entry lets the M=1 O load during attention: decode 1 -0.95 / -2.06 us per sliding / full layer, `4d995f13`), M=128 verify qkv split-K (20.42 vs 23.26 us, `b15a3bde`), cluster glue at t <= 32 (one cluster per token, norm partials through DSM: -1.9 to -3.2 us per layer, `1365a290`). Staged: the BF16 qkv front (fused sum+cast, PDL attn_main; local -2 to -7 us; lab `8665cfc9`). Closed recipes: verify split-K beyond M=128 qkv, bulk-ring GEMVs standalone (`87b2667f`), attn_main reading split partials (`aab93d73`), FA3 metadata on a side stream (`e1a56ee7`), cooperative glue (`9eeb39ba`), skipping zero-local rows. Next: (1) the small-M projections' excess (GEMVs at 1.5-2.0x their floor): bytes in flight per SM, tails; (2) router beside the pooler, top-8 broadcast and integer argmax; (3) the T <= 8 zero tail; (4) MoERouteMeta from the cluster route kernel for the small-M consumer (local, exact) | Projections <= 1.3x floor and glue <= 6 us per layer | glue/proj lane | active |
| 4 | P3a | G1, G2 | Small-M MoE, M=1-32 (+6.6 / +9.9%). H200 timeline of the M=1 call (`849e4017`): node 31.8 us = launch 1.0 + CTA span 32.2 + drain 1.3, the critical CTA's phases covering the span (route 3.0, norm-zero wait 4.1, W13 10.0, W2-ready wait 6.4, W2 4.2, barrier 2.6, combine 1.3); the old "missing 16 us" was those waits, event framing, seed mix and the dirty flush. Flags, paired vs INC (router+MoE, 16 seeds): M=1 Z+SLOT+ZOUT+RSQ+NREG+PDL -5.72 us of 42.4; M=8/16/32 Z+ZOUT+RSQ+NREG+PDL -3.54 / -4.55 / -3.69; into S9. Closed recipes: route metadata from the router's epilogue (+2.9-3.3 us, serial epilogue), a last-CTA finalizer, router-epilogue L2 prefetch, the joint E2M1 x E4M3 table decoder (90.6 vs LOP3's 26 instructions per 16 weights). Left against a pure stream over the same 56.6 MB at M=1: 11.5 us (route 1.5, first tile 3.7, W2-ready wait 4.7, finalizer ~2.4). Next: (1) the dynamic M=1 schedule (local -2 us more); (2) M=32 timeline (W2 skew at the grid barrier); (3) decode-once for heavy experts at M=16/32 (11.5 blocks re-decode 7.8 experts), uint8 scales only; (4) metadata from the cluster route kernel (no extra launch) | Each load's MoE within ~1.5x of its floor | small-M lane | active |
| 5 | P3b | G1, G2 | MoE at M=128/256 (+2.5 / +3.6% and +6.0 / +7.2%). M=256: R_full (light and heavy experts forked once, joined before the finalizer) is in S7R (0.9227x mixed_parallel; verify 32 -33 / -29 us per layer in `c1fbd0cf`); N_full (NT1 for <=16-token experts on a side stream beside N32 WGMMA on the heavy list) measured 0.8701x (`14f942d5`), handoff next; a route with all id loads in flight takes 10.6 -> 6.6 us. M=128: H_H on uint8 scales (S7) 234 us vs floor ~106; typed-smem WGMMA (HH_u8) 0.948x of H_H's W13/W2 on M=256's schedule (`14f942d5`); W2 threshold 8 0.9824 (`4c469bf9`, u16 heavy path, rebuild on u8). The one-read M=128 recipe (`640d67e7`) doesn't isolate reread cost. Next: (1) N_full into S9; (2) HH_u8 and the branch fork at M=128, on u8; (3) decode-once producer/consumer with a separate consumer axis (the D split's 2-3x loss is implementation suspect) | Each within ~1.5x of its floor | MoE lane | active |
| 6 | P11 | G1 | Whole-layer persistent kernel (megakernel) at small M. Scoped op by op on the reset profile (`problems/muse_decode_layer/dev/megakernel/README.md`): the dependencies and round trips it removes are worth +1.74% on the six-load score (+2.22% if the whole synchronization tax went); the 47-79 us per layer above a 3.6 TB/s stream is mostly mainloop, dequant and attention work that the stage rows attack. Prior art: Hazy "No Bubbles" 1.5x over SGLang at batch 1, Cohere 1.58x, MPK 1.0-1.7x (SOTA); our persistent dense chain lost 1.5-6.5x (`87dc944b`, mechanism unknown) | Reopen when a stage receipt or boundary evidence lifts the estimate above the rows ranked above it | none | parked: +1.74% estimated |
| 7 | P10 | G1 | Lossless weight compression (llms.txt:107). Hardware generic compression is granted on the judge's H200 but real NVFP4 and BF16 weights read at the random-byte rate (0.997-1.002x; zeros 2.03x), closed (`4233660b`). Software coding of BF16 exponents (-34.6 MB of 138 MB at decode 1) pays only on a byte-bound stage | A byte-bound stage exists for it to cut | proj lane | parked: no byte-bound stage yet |
