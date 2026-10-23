# Plan

## Status (2026-10-23 21:30 KST)
- Kernel track day 1 of 14 (closes 11-05 17:00 KST); serving opens 11-06 05:00 KST.
- Best H200 checks: proj 1.142x, attention 1.388x (1.384x on the board), MoE 1.535x, decode layer
  1.271x, prefill 1.345x (SOTA "Current best"). Scored runs of MoE V6 and v7 and prefill v3 are queued.
- Overall board at 19:30: about 1.19x time-weighted against about 1.38x for 50th place (FINDINGS "Boards").
  Decode layer and MoE carry about 80% of that weight.
- Serving patch: Hopper gates #2/#4/#5, prefill row split, FA3 QK-norm/KV-store fusion, drafter fc split,
  fused MoE finalize (#1, #3), and a single-bank MoE port (canonical V6 bank, 54.4 GB/GPU). None of it has run on Hopper.
- Bound (`docs/reference/bound.md`): serving score at most 6.13x at spec HBM (5.89x at our measured
  4.25 TB/s, tau 4.16) against the practical model's 3.0x. The kernel bounds are proj 2.02x, attention 6.29x,
  MoE 3.29x, decode 3.34x, prefill 2.93x. The biggest gaps from our receipts are attention 4.9x, decode 2.9x, MoE 2.2x.
- Bottleneck: judge latency. About 11 of our runs start per hour, so a check sent behind about 35
  others lands 2-3 h later.

## Critical path
Kernel loop, measured 2026-10-23:
1. A lane edits a candidate.
2. Local screen (`tools/local_eval.py`, sm_120): tests take 1-5 min; local timing ranks loosely.
3. `okc_run.py check`: the queue wait is our queue position x ~5.5 min, and each run takes 1-5 min.
4. Read the receipt and per-shape profile, then promote and send a scored run.
5. Port into `serving/sglang`, then the local harness: layer tests and TP1/TP2 agreement, minutes each.
6. Serving check: from 11-06 only, 3 a day.

Step 3 dominates, since one result costs hours of wall time.
- Fixes: denser checks (P8), priority queueing (done), and Hopper glue exercised locally (AGENTS).
- Step 6 can't run before 11-06, so the harness carries the serving risk until then.
- Lanes run in parallel: one per problem plus serving, bound and research. A lane never idles on the
  judge; `collect` lands receipts while it builds the next candidate.

## Ranked work
| Rank | Id | Serves | Work | Done when | Who | State |
|---|---|---|---|---|---|---|
| 1 | P1 | G1 | Serving: an agreement metric that resolves stock's own spread, then request-based (suffix/n-gram) drafting alongside DFlash | Metric reports stock-vs-patch within the stock-vs-stock spread; acceptance gain measured on the multi-turn fixture | serving lane | active |
| 2 | P2 | G2, G1 | Prefill: RS128 grouped experts on the canonical bank (provisional serving path) and FP8 long-context attention; nine candidates at 09:00 KST | H200 check above 1.345x; ported with agreement | prefill lane | next: 09:00 KST checks |
| 3 | P3 | G2, G1 | MoE V8 (V6 resident grid plus v7 incremental tiles) on the canonical bank, swapped into the serving port | H200 check above 1.535x; port tests pass | MoE lane | active |
| 4 | P4 | G3, G2 | Decode layer round 3: cand_tuned (1.271x) plus the best measured stage per batch size (MoE v7/V8, attention hybrid q_len=1, proj tuned dispatch, neutral structural fusions); `problems/muse_decode_layer/dev/NEXT_CHECKS.md` at 09:00 KST | H200 check above 1.271x on the way to the 50th-place mix | decode lane | active: local build; checks at 09:00 KST |
| 5 | P5 | G2, G3 | Attention verify kernel (XQA-style q8 x G8, page-1 gathers) to beat FA3 on verify loads | Verify loads above 1.0x on H200; board above 1.384x | attention lane (Claude agent: Codex at capacity) | active |
| 7 | P7 | G3, G2 | Proj: beat cuBLAS on qkv_gate/o (in-kernel split-K reduce) and m>=16 pooler/unpooler | H200 check above 1.142x; at the 09:00 reset, one check of the exact submission.py (its source was promoted from a receipt the okc_run file-edit bug may have mislabelled) | proj lane | next: 09:00 KST check |
| 8 | P8 | G2 | One H200 diagnostic: HBM concurrency (LDG/cp.async/bulk), page-1 KV gather, L2 hints/APW, graph PDL boundary, small-N BF16 MMA/WGMMA, NVFP4 decode issue | H200 rates and lane consequences in FINDINGS | probe lane | active: check `849bd4b650e54c7f` queued |
| 9 | P9 | G2 | Fold per-problem notes (`dev/STATUS*.md`, `RESULTS.md`, `PORTING.md`, `LIVE_STATUS.md`) into FINDINGS rows plus one FOCUS.md of at most 30 lines per lane | Each problem has a FOCUS.md; its facts are in FINDINGS | each lane, at its next resume | next |

## Open questions
- G2 above G3? The overall top-50 cut (G3) closes 11-05, before serving opens, and its decode loads never
  run in serving. Recommend keeping G2 first while MoE and decode-layer work serves both (not yet asked).
