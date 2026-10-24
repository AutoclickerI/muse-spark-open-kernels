# Serving round (parked)

Not a current goal (R-20261025-1). No lane works on serving until the serving round opens (the track opens
2026-11-05 noon PT, 2026-11-06 05:00 KST). Then this folder seeds the new goal tree; until then nothing here
binds or ranks work.

## Files
- [FINDINGS.md](FINDINGS.md): serving-track and prefill findings moved out of `docs/FINDINGS.md` on 2026-10-25.
- [serving_levers.md](serving_levers.md), [moe_port.md](moe_port.md), [muse_dflash.md](muse_dflash.md),
  [prefill_gemm.md](prefill_gemm.md), [prefill_glue.md](prefill_glue.md), [upstream_sglang.md](upstream_sglang.md):
  references for the patch, the expert port, the drafter, prefill and upstream SGLang.
- Code: `serving/` (the organizers' `RULES.md`, our `patch.diff`, the local harness), `problems/muse_prefill_layer/`
  (also a live H200 lab host), `sim/serving.py`.

## Additional parked notes

- [Expert bank](expert_layout.md): the serving loader, reader ABI and residency evidence from `docs/reference/`.
- [Reference sections](reference_notes.md) and [lane sections](lane_notes.md): source text removed from mixed live docs. Historical goal tags and next steps in these archives do not apply to current work.
- [Prefill send notes](prefill_dev/NEXT_CHECKS.md), [V11 profile](prefill_dev/V11_PROFILE.md), [glue](prefill_dev/glue/README.md), [CUTLASS dense](prefill_dev/cutlass_dense/README.md), [A30 dense](prefill_dev/dense_a30/README.md), [mid dense](prefill_dev/mid_dense/README.md), and [expert dispatch](prefill_dev/sweep_dispatch/README.md): prefill-only lane notes. Code and receipts remain at their original paths.

## Serving rules (bind serving submissions only)
Source: `upstream/llms.txt` "Serving" and `serving/RULES.md`.
- **Serving patch.** A git diff against SGLang `a55ab7d5`: only source files and JSON under `python/sglang/`,
  not `srt/model_loader/`; no renames, mode or binary changes; at most 720 KB, no added line over 400
  characters; prebuilt `sgl-kernel` isn't rebuilt; the judge's flags are fixed but what they do may change.
- **Same model.** 8 routed experts per token, the 2,048-token window, every norm, the registered parameter
  names and shapes.
- **Drafter.** DFlash with its released weights, made faster but not changed; other drafting only from the
  requests (prompt lookup, n-grams).
- **Outputs.** Agree with stock "nearly as well as it agrees with itself"; exactly 220 tokens per request;
  TTFT at most 2x stock's per load.
- **Quota.** 1 submission and 3 checks a day; the top 50 on the overall kernel board at the kernel close get
  queue priority.
- Boundary cases read earlier: our own cubin in the patch, even base64 in a `.py`, is a binary change
  (forbidden); re-sharding weights after load and scheduling changes are allowed within the flags and the
  TTFT cap; tree verification and adaptive draft length are allowed when the harness shows agreement
  (R-20261023-8 below).

## Jihoon's serving answers (for the serving round)

### R-20261023-1 — 2026-10-23 14:37 KST — the goal is the serving prize
> Serving prize (Recommended)

- Source: transcript:147. Answered "What is the real goal? The prizes are only on the serving board
  (Nov 5–19). The kernel boards pay nothing, but they lead into serving through queue priority and
  kernels we can port into SGLang."
- Effect now: none until the serving round (R-20261025-1: "serving will be studied in the serving round").

### R-20261023-8 — 2026-10-23 21:55 KST — tree verification and adaptive draft length
> Allowed if agreement holds (Recommended)

- Source: transcript:6054. Answered "In serving, may we verify a tree of DFlash drafts or adapt the
  draft length, keeping the released drafter and stock-agreeing outputs?"
- Effect in the serving round: allowed when the local harness shows agreement with stock.

### R-20261024-1 — 2026-10-24 00:30 KST — serving above kernel work after the kernel close
> G3 first until 11-05

- Source: `/home/jihoon/.claude/projects/-home-jihoon/70737187-e5b5-4d2f-a79a-2ce9414b9a0a.jsonl`:6780
  (question-tool label, asked from the jihoon-a5 session). Answered "when they conflict, does G2 (H200
  kernels at serving's shapes) rank above G3 (top 50 on the overall kernel board by 11-05, which only
  moves our serving runs to the front of the queue)?"
- Effect now: none; the kernel board is the only goal until the close (R-20261025-1).

Transcript: `~/.claude/projects/-home-jihoon-competitions-muse-kernel/ae5ada91-d8f5-49d0-a518-22df30a603ba.jsonl`.
