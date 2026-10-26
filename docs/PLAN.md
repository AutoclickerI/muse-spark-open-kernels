# Plan

## Status (2026-10-26 11:30 KST, day 5 of 14; the kernel track closes 2026-11-05 16:59 KST)
- G0 (R-20261025-1): the highest score on the overall kernel board; by our fit (GOAL) that is our decode-layer
  speedup. Board (11:30 KST): S16 `c6cc2874` 1.8242x, 94th of 705 (S15 `9c5fc18d` 1.8003x); leader 2.3601x,
  10th 2.1665x, 50th 1.9357x, all rising (SOTA "Leaderboard").
- Incumbent S15 `60254883` (`problems/muse_decode_layer/cand_r3_best7_s15.py` = S14 + the public proj port's
  winning shapes + last-CTA decode-1 sliding + register-fold N128 full verify): paired over S14 at a geomean of
  1.026 (`281b1383`, FINDINGS "Our kernels on the H200"). Our old gate
  had rejected it for v1 alone; the gates now follow the score (Ranked work). S16 (`435df5f78a0d`: DSM cluster sliding at v1 / v16, PAIR epilogue, early release, library O / pool) awaits its paired confirmation `37c2879f`; S17 adds S15n, ILP and later nominations (P4).
- **What 1st takes:** ~1.29x over S16. The leader sits at ~1.3x the one-continuous-stream bound (3.08x); their
  component boards run 1.4-1.8x ours (attention 2.91 vs 1.59, MoE 2.52 vs 1.69, proj 1.61 vs 1.19; P12 is
  re-measuring ours with S15's kernels). No single stage closes the gap. Each lane removes its stage's time
  above roof by whatever removes it: structural rebuilds, and the waits, recomputation and work our winning kernels
  don't need (readiness per block or operand instead of per stage, statistics computed upstream, algebraic
  identities). The size of a change doesn't matter, only its six-load gain per H200 run.
- Where the time goes (S15 timeline `7eaa262a`, `dev/s15_timeline/H200.md`; critical-path time above the roof as
  a share of wall at d1 / d16 / d32 / v1 / v16 / v32): MoE 18.7 / 18.2 / 21.4 / 14.8 / 17.3 / 24.2%; attention
  7.6 / 8.5 / 9.2 / 13.0 / 9.2 / 8.6%; projections 1.9 / 2.6 / 3.9 / 1.9 / 4.1 / 7.4%; glue 3.6 / 0.5 / 0.5 /
  0.7 / 1.3 / 1.5%; GPU idle <= 1.7%. MoE is the largest gap at every load, about twice attention.
- Ceilings on S11 (per-layer us at the six loads 95.7 / 179 / 243 / 139 / 333 / 462): every stage at its stream
  roof 2.79x; one continuous stream 3.08x; the 4.8 TB/s floor 3.31x.

## Critical path
Loop: a lane edits a candidate or lab payload → local screen (`tools/local_eval.py`, 1-5 min, sm_120, ranks
loosely) → H200 via `okc_run.py send` / `h200_lab.py send` (queue ~6.6 min per position ahead, 1-5 min per run)
→ read the receipt → integration pairs winners in the whole layer → scored run.
- **Bottleneck: H200 runs.** Each day gives 150 (5 problems x 10 scored + 20 checks). On 10-25 the prepare hosts
  were empty by ~17:00 KST and decode by ~21:00, and seven lanes then waited 12 hours for the reset.
- Fixes in place: batched payloads per run; every problem hosts labs (attention at import time, R-20261025-3,
  `--import-budget` default 150 s, proven by `33d8bb68`); labs queue with `--not-before` for the reset; integration's
  A/B gets priority -1 on decode.
- Next fix: pack each run fuller (more arms and seeds per payload, all of a lane's variants in one run), spend
  decode checks only on whole-layer pairs, and send each lane's reset batch within the first hour so the day's
  second round fits.

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
| 2 | P3b | G1, G2 | MoE at M=128 / 256 (verify 16 / 32; 17.3 / 24.2% of wall above roof at v16 / v32). S14: light experts decoded once on an N16 third stream (182.4 / 256.4 us). The heavy branch runs at 1.9-2.3x its stream: weights read 1.6x / 2.2x and activations re-read (171 / 403 MB). Decode-once with cluster activation sharing lost 1.13-1.44 (`8ee003db`), as did shared queues (`ea794250`). At the reset: launch-topology / stage-3 attribution lab. Next: remove the weight and activation multiplicity without the staging cost those recipes paid | 132.6 / 158.5 us (unique bytes at ~4 TB/s plus 1.3x) | MoE lane | active |
| 3 | P3a | G1, G2 | Small-M MoE, M=1-32 (decode loads, verify 1; 14.8-21.4% of wall above roof). S14: NT1 flags `{1:95, 8:69, 16:69, 32:69}`, 64.9 / 75.9 / 104.7 us at M=8 / 16 / 32 vs a 49-53 us pure stream. Every reuse rebuild lost 1.12-1.78 (`b5659458`, `2fad2371`); the 58.7 us math clock in FINDINGS is that failed ONCE schedule, not S14. Static count: integer / address issue at 74% of its pipe at M=32, HMUL2 and tensor 17% (`dev/nt1_timeline/PIPE.md`). PAIR epilogue 0.975 / 0.973 / 0.979 of S14's MoE at M=8 / 16 / 32 (`d650c04e`, `29ad388d`), to P4. Sent: FLIGHT / ILP / K16 / LUT rebuilds (`de47736b`). Next, each a small exact change: W2 waits on `done[e, nb]` (its own token block) instead of every block of expert e (M=16 / 32); W2's global scale cancelled through the following RMSNorm (multiply by nothing, eps / a^2 per expert, sign kept; not W13, SwiGLU follows); norm accumulators in 2 / 4 / 8 shards between PAIR and LOCAL; NVFP4 decode as sign and exponent assembly on two bases S and 1.5S (bit-exact on all 8,128 FP4 x E4M3 pairs per the review; count SASS first, it fails if integer issue grows); expert scale bytes as exact 5-bit packets where the real weights allow (up to 4.17% of tile bytes; measure compressibility first). The last two apply to P3b's heavy branch too | Each load's MoE within ~1.3x of its stream | small-M lane | active |
| 4 | P5 | G1, G2 | Sliding and decode-1 attention (sliding is 46 of 62 layers; attention is 7.6-13.0% of wall above roof). In S15: last-CTA decode-1 sliding (one launch, 15.1 -> 11.1 us, `7e93cc81`). At the reset: early history fetch (the 2,048-key gather and V transpose issued before the PDL wait, smem 53.5 -> 28.9 KB, `dev/early_history/`), native DSM cluster sweep `f1c270e6`. Verify-32 sliding is 29.8 us per layer vs a 3.5 us KV floor (`c6bd22e2`). Sent: round 2 (decode 16 / 32 and verify-1 one-launch, 28 arms), verify_resident (verify 16 / 32: one launch, history before the PDL wait, DSM reduction). Next: K-ready and V-ready published separately wherever QK of the next tile waits on the V transpose (the shared `full[slot]` barrier) | Sliding <= 6-8 us per layer at every load | attention lane | active |
| 5 | P5b | G1, G2 | Full attention (16 of 62 layers). In S15: register-fold N128 full verify (0.977 / 0.948 at verify 16 / 32, `9d9a01e1`). Full verify is softmax-bound: ~907 of ~1,690 cycles per 64-key tile (`f4d1a195`). At the reset: eight softmax variants (25 / 50% of exponentials as an FMA polynomial beside MUFU, row sums on the tensor core, two-warpgroup ping-pong; `dev/full_attention/softmax/`). Sent: the same at decode 16 / 32 full (attn_main, 1.5-1.6x its floor). Next: VQ8's producer publishes K-ready after the K/V copy and V-ready after the V transpose, so QK(j+1) stops waiting on V(j+1)'s transpose (+32 B smem at ST4); q_len=1 full decode as K-major native FP8 `wgmma m64n8k32` (keys or head dim on M, the 8 queries on N); two online-softmax states (even / odd tiles) per consumer warpgroup with `setmaxnreg` moving registers from the producer, merged at the end | Full attention within ~1.3x of its KV floor at every full shape | full-attention lane | active |
| 6 | P12 | G1 | Our per-stage gap to the leader. The component boards time the layer's stages at the layer's shapes; ours there are old kernels. Package S15's MoE, attention and projection paths as component submissions (scored quota the score doesn't use) and compare with the leader's component times, to rank stages by their gap to the leader rather than to the roof | Our current stage kernels on all three component boards | public-proj lane | next |
| 7 | P10 | G1 | Dense projections (1.9-7.4% of wall above roof, most at v32). In S15: NathenFernandes's MIT kernels at the shapes they win in our layer (R-20261025-2; qkv 0.71 at t=1, 0.82 at t=32; O 0.85-0.96; unpooler 0.81 / 0.90 at t=16 / 32; their fused pool / router loses, `e3c637d1`). Next: pool_router (ours keeps the router fused) and denser lossless codes (2.545 bits of exponent entropy; 11.42 bits per weight measured for K8192, ~4.9% fewer bytes) on their kernel structure | Every dense stage at its stream roof, or below it through coding | public-proj lane | next |
| 8 | P7 | G1, G2 | Glue (b1 + route + b2; 3.6% of wall above roof at d1, <= 1.5% elsewhere); the lane also takes MoE's route / finish / tail at M=128 / 256 (16-17 us, P3b's ledger). At T <= 8 (the call's total, decode 1 / verify 1): b2's last 7,168 residual elements are alpha2 * stream1 and fixed once b1 has run, so b1 writes them and their sum of squares and b2 reduces only the first 1,024 (RMS denominator stays 8,192). Producer-epilogue fusion lost 1.15-1.29 (`f21c60b0`, `15abb9c1`). PDL audit: nvcc sinks intended preloads past the dependency wait in 17 of 19 cluster-glue variants; fixing all lost 0.885 (register pressure), metadata-only +0.3% (`0f74021a`) is confirmed or dropped at the reset (`dev/pdl_audit/`). Boundaries at d16 / d32 are 5-6 us critical (`dev/boundary_attribution/`). Sent: decode-1 chain, boundary batch and boundary rebuild, re-run on the attention host (`ce5b991a`, `5a95984a`, `9f6849fc`) after the prefill host returned no lab lines. Next: where each producer issues `griddepcontrol.launch_dependents` (after its last load is issued, not after it lands: 34 us on Llama-1B, SOTA "Public competitor code") | Glue <= 6 us per layer | glue lane | active |
| 9 | P11 | G1 | Whole-layer persistent kernel at small M: +1.74% estimated (+2.22% if the whole synchronization tax went); a resident kernel needs competitive inner loops, precise readiness and operands kept across stages, not just fewer launches (`docs/reference/research_megakernel.md`); our persistent dense chain lost 1.5-6.5x (`87dc944b`, mechanism unknown) | Reopen when stage receipts lift the estimate above the rows ranked above it | none | parked: +1.74% estimated |

## Questions for Jihoon
None open.
