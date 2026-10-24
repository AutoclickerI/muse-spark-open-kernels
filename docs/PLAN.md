# Plan

## Status (2026-10-25 00:40 KST)
- G0 (R-20261025-1): the highest score on the overall kernel board by the close (2026-11-05 16:59 KST, day 3 of 14).
  By our fit of the ranks (FINDINGS "Boards"), every top-50 team's overall score is its decode-layer speedup.
- Board (SOTA "Leaderboard", 10-24 23:35 KST): we are 178th of 478 on the decode layer at 1.429x (v9 `0e94268`,
  1.422-1.434x on four router seeds). Leader 2.146x, 10th 1.967x, 50th 1.734x; rank 50 gains ~0.1x per 11 h.
- Where our layer time goes against one-read floors: MoE 2.2-4.3x its floor at every load, full-layer attention 1.5-3x,
  projections ~1.6x (bound by per-launch fixed cost, not HBM: L2 oracle `09d19813`), glue 15-28 us per layer. Floor
  model: every stage at 1.5x its floor + 6 us glue gives ~2.09x; at 1.3x, ~2.38x.
- Lever table (`problems/muse_decode_layer/dev/NEXT_CHECKS.md`, best7_reset profiles + measured deltas): overall gain if
  each stage reached 1.3x its floor: small-M MoE +13.2%, attention +11.5%, glue +6.7%, projection GEMVs +6.6%, M=256
  MoE +6.4%, M=128 MoE +6.2% (together ~2.3x). Ranked work follows this order.
- Scored-run line: S1 `cand_r3_best7_scale.py` (`42f543a0`: best7_paired + the lossless recoded scale bank, NT1 9-12%
  faster at M=8-128, `58ae2e2c`) and S2 (+ fused M=256 MoE, 385 vs Humming 448 us, `97be11d4`) go at the 09:01 KST
  reset. S3 `cand_r3_best7_scale_recoded_m256.py` (`cadef19a`: recoded fused M=256, 365 us, `9e8a6a15`) follows.
- Daily caps: decode, attention and prefill 10 scored + 10 checks; MoE and proj 10 + 20. Labs ride the proj, MoE and
  prefill quota; the judge rejects some words before queueing (FINDINGS "Failures"); `okc_run.py send` screens for them.
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
- The receipt keeps only the last 16,000 characters of output (`log_tail`), so print compact lines last.
- Lanes run in parallel, one per stage plus integration. A lane never idles on the judge; `collect` lands receipts
  while it builds the next variant.

## Ranked work
Targets come from the floor model above: each stage within ~1.3-1.5x of its floor, judged by whole-layer time
(sensitivity table in `problems/muse_decode_layer/FOCUS.md`). A measured loss closes only the recipe tested.

| Rank | Id | Serves | Work | Done when | Who | State |
|---|---|---|---|---|---|---|
| 1 | P4 | G1 | Integration and the daily scored runs: fold every stage win into the next candidate, chosen by seed-mean and paired-lab evidence. S1 `42f543a0`, S2 `b0805447`, S3 `cadef19a` queued for the 09:01 reset; S4 = S3 + D32g at M=128 (273 vs NT1-recoded 290 us, `dc3fea0a`) + the one-LOP3 NT1 decoder at M=1-32 (1-3%) | A scored run above the leader's (2.146x now, rising) with the seed-mean model above it too | decode lane | active: S4 build; lever-table refresh from S1-S3 receipts |
| 2 | P3a | G1, G2 | Small-M MoE, M=1-32 (lever +13.2%). NT1 recoded: 44 / 84 / 90 / 126 us at M=1/8/16/32 vs floors ~12-18 / 44 / 45 / 53. At M=1 fixed phases bind (`dc3fea0a`): every CTA re-derives the routing (~2.9 us), grid W13->W2 wait+barrier ~4.5 us, W2 at ~1.7 TB/s vs W13's ~3.6. At M=16-32 dequant (~27 us) and MMA (~35 us) bind; NT1 spends 27-31 instructions per 16 weights. Next, in one lab: (a) routing metadata built once per call (per-expert token lists, offsets, slot maps) by the route kernel, read by every CTA; (b) per-expert ready flags instead of the grid barrier, so W2 starts on its expert's rows; (c) W2 at W13's bandwidth; (d) at M=16-32, a warp-specialized split: producer warps dequantize to a BF16 smem tile, consumer warps run the MMA | Each load's MoE within ~1.5x of its floor on H200 seed means | small-M lane; glue lane emits the metadata from its route kernel | active |
| 3 | P5 | G1, G2 | Attention (lever +11.5%). Verify full q8: FA3 208 / 390 us at B16/B32 vs floors 63 / 127, page-1-gather bound (~1.6 TB/s); sliding layers spend 15-30 us of fixed work (floor 0.1-3.7 us). Single-launch sliding `32f2f07a` loses at every shape but verify 1 (12.1 vs 16.2 us; verify 16/32: 36 / 80 vs 24 / 31 us). FP8PV parked (`0812f097`). Next: the verify q8 FA2-style kernel; conditional softmax rescale (skip when the running max doesn't change) in `attn_main` q1 and a deferred denominator reduction in the q8 `xqa_group.cu` | Full-layer attention within ~1.5x of its KV floor and sliding <=6 us at B1 | attention lane | active |
| 4 | P7 | G1, G2 | Glue (lever +6.7%) and projections (+6.6%). Dense GEMVs run at 1.2-2.4 TB/s cold and only 2-3.7 TB/s L2-resident (`09d19813`): per-launch fixed cost binds, and the proj board field is ~30% faster. Glue: boundaries 12.9-14 us + route 3.6 us per layer at M=1 vs <0.1 us floor. Closed recipes: PDL prefetch `7426e17c`, verify split-K `19f3a13b`, persistent chain `87dc944b`. Next: fixed-cost decomposition lab `95ab383d` (14 register-MMA cells); glue fusion lab `80b70f7f`; an upper bound for overlapping attention with qkv by KV-head group (attention on group g starts when its qkv rows land) before building it | Projections <=1.3x floor and glue <=6 us per layer | proj lane (GEMV fixed cost), glue lane (boundary/route fusion, qkv-attention overlap bound) | active: `95ab383d`, `80b70f7f` at the judge |
| 5 | P3b | G1, G2 | MoE at M=128/256 (levers +6.2% / +6.4%). M=128: D32g 273 us (W13+act 174, W2+final 81, finish 12) vs floor ~106; NT1 rereads tiles 2.18x. M=256: recoded C1g2112 365 us vs floor ~124. Next: (a) per-expert backend mix, recoded NT1 for experts with <=16 tokens and packed WGMMA32 above (35-50% fewer tile groups at +12-14% padding, `seed_sweep.json`); (b) W13 and W2 tile, grid and backend chosen independently; (c) the W2 finalizer split across CTAs where it still serializes | Each within ~1.5x of its floor on H200 seed means | MoE lane (M=128), free lane (M=256) | active |
| 6 | P10 | G1 | Lossless compression of the weights (llms.txt:107: "compress them losslessly"). (a) Hopper generic memory compression: allocate the weights with `cuMemCreate` and `CU_MEM_ALLOCATION_COMP_GENERIC` in `prepare`; one lab answers whether the judge's H200 grants it (`CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED`, `cuMemGetAllocationPropertiesFromHandle`) and whether it cuts read time for real NVFP4 experts and BF16 dense weights. (b) Software coding of BF16 exponents (12 bits/value, -34.6 MB of the 138 MB decode-1 bytes) pays only once a stage is byte-bound, which the dense GEMVs aren't yet | The lab shows a per-stage time change on real weights, or support/ratio rule it out | proj lane | next: lab now |
| 7 | P9 | G2 | Fold per-problem notes (`dev/STATUS*.md`, `RESULTS.md`, `PORTING.md`, `LIVE_STATUS.md`) into FINDINGS rows plus one FOCUS.md of at most 30 lines per lane | Each problem has a FOCUS.md; its facts are in FINDINGS | each lane, at its next resume | active |
