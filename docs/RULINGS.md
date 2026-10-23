# Rulings

Jihoon's words behind current standing decisions, verbatim, oldest first: a rule Jihoon confirmed, the
G0 and ranking decisions, answers to agents' questions. Ids are `R-<YYYYMMDD>-<n>`. A chosen
question-tool label is Jihoon's word; the question and option text are agent context. Transcript:
`~/.claude/projects/-home-jihoon-competitions-muse-kernel/ae5ada91-d8f5-49d0-a518-22df30a603ba.jsonl`.

## R-20261023-1 — 2026-10-23 14:37 KST — the goal is the serving prize
> Serving prize (Recommended)

- Source: transcript:147. Answered "What is the real goal? The prizes are only on the serving board
  (Nov 5–19). The kernel boards pay nothing, but they lead into serving through queue priority and
  kernels we can port into SGLang."
- Effect: G0, and the subgoal ranking G1 > G2 > G3 (GOAL).

## R-20261023-2 — 2026-10-23 14:37 KST — private GitHub repo
> Private GitHub repo (Recommended)

- Source: transcript:147. Answered "Where should the repo live?"
- Effect: the repo is github.com/AutoclickerI/muse_kernel, private (RESOURCES).

## R-20261023-3 — 2026-10-23 16:48 KST — anything within the rules; no H200 rental
> you can do anything that can improve performance within our spec. I cannot rent H200.

- Source: transcript:1690. Answered four pending decisions: building on Humming, a check to measure
  scattered KV reads, renting an 8x H200 node, deleting the 16.6 GB `.sif`.
- Effect: every performance change within the hard limits is pre-authorized (GOAL "Decision rights").
  No H200 rental; H200 numbers come only from the judge (GOAL "Non-goals").

## R-20261023-4 — 2026-10-23 21:14 KST — widest scope for the bound and the work
> update our docs now. restructure our repo as the findings. and let's start from setting the lower bound first. how much can we get theorically? we can do any ordering, approximation, or any other change including binary use to achieve the highest performance.

- Source: transcript:5061.
- Effect: CONSTRAINTS "Boundary cases" take the widest reading. Any ordering is allowed. Approximation
  is allowed at the allowed formats. Prebuilt image binaries and JIT builds are allowed. The bound
  (PLAN P6) also reports an "outside the rules" column, labelled inadmissible.
- Conflict: approximation below the allowed precision (FP8/FP4 activations, fewer experts, skipped
  keys) breaks hard limits 2 and 5-7. Binary changes in the serving patch break limit 9. The hard
  limits win (llms.txt:107-110, 126, 137). Whether our own cubins or hand-edited SASS may go into a
  kernel submission was asked 2026-10-23.

## Dropped agent rules (do not reintroduce)

| Rule an agent introduced | Dropped | Match |
|---|---|---|
| Stop opening new optimization branches after two failed ideas in a row (prefill lane) | 2026-10-23: a self-imposed stop; no owner or external source | `no new optimization branches` |
| "Keep a portable fallback path where that's cheap" (old AGENTS) | 2026-10-23: replaced by the practice of local stand-ins that test Hopper glue | `portable fallback path` |
| proj is "worth only enough work to keep rank" (old CEILING) | 2026-10-23: an effort cap with no source; proj kernels also feed serving | `worth only enough work` |
| Lanes wait for queued judge receipts before starting new work | 2026-10-23: idle waiting; `okc_run.py collect` lands receipts on its own | `no new checks will be sent` |
