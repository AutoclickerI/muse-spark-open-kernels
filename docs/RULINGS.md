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
- Effect: G0 (GOAL).

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
  limits win (llms.txt:107-110, 126, 137). Our own cubins and SASS on the kernel boards: R-20261023-6.

## R-20261023-5 — 2026-10-23 21:55 KST — Codex default, never gpt-6-sol: kept as rules
> Keep both (Recommended)

- Source: transcript:6054. Answered "Keep these two instructions as standing rules: 'for subagents, use
  codex rescue as a default' and 'don't use gpt-6-sol from now on'?" The words behind them (2026-10-23):
  "for subagents, use codex rescue as a default." (transcript:2211) and "don't use gpt-6-sol from now on.
  there is no reason to use it. we can just use gpt-6.1-sol instead." (transcript:2755)
- Effect: rule 15; rule 16 is now set by R-20261023-9.
- Confirmed: in the words.

## R-20261023-6 — 2026-10-23 21:55 KST — anything the board accepts and the rules don't ban
> do anything accepted in leaderboard and not explicitly banned in rules.

- Source: transcript:6054. Answered "May a kernel-board submission embed our own compiled cubin or
  hand-edited SASS (the serving patch can't; it forbids binary changes)?"
- Effect: our own cubins and hand-edited SASS are allowed in kernel submissions. Kernel-board boundary
  cases are read the same way: allowed when the judge accepts them and no rule explicitly bans them.
- Conflict: llms.txt:139 keeps discretionary disqualification for entries outside "the spirit of the
  challenge"; that is a review risk, not an explicit ban, so the case is allowed.

## R-20261023-7 — 2026-10-23 21:55 KST — autotuning in prepare()
> Allowed (Recommended)

- Source: transcript:6054. Answered "May prepare() autotune: time candidate kernels on synthetic inputs
  of the fixed shapes and pick per shape (untimed, nothing derived from the timed data)?"
- Effect: allowed (CONSTRAINTS "Boundary cases").

## R-20261023-8 — 2026-10-23 21:55 KST — tree verification and adaptive draft length
> Allowed if agreement holds (Recommended)

- Source: transcript:6054. Answered "In serving, may we verify a tree of DFlash drafts or adapt the
  draft length, keeping the released drafter and stock-agreeing outputs?"
- Effect: allowed when the local harness shows agreement with stock (CONSTRAINTS "Boundary cases").

## R-20261023-9 — 2026-10-23 22:55 KST — Codex models: gpt-6.1-sol and gpt-6-astra
> use gpt-6.1-sol. not old models.

> use 6.1-sol and 6-astra.

> Retry gpt-6.1-sol only (Recommended)

- Source: transcript:6579 (after lanes had been rerouted to gpt-5.6-sol, gpt-6-luna and gpt-6-astra on
  capacity errors); transcript:6649 answered "Make 'use gpt-6.1-sol, not old models' a standing rule
  (extending rule 16 ...)?" and "When gpt-6.1-sol itself is at capacity ..., what should a stopped lane do?"
- Effect: rule 16.
- Confirmed: the answer to the standing-rule question.

## R-20261024-1 — 2026-10-24 00:30 KST — G3 above G2 until the kernel close
> G3 first until 11-05

- Source: `/home/jihoon/.claude/projects/-home-jihoon/70737187-e5b5-4d2f-a79a-2ce9414b9a0a.jsonl`:6780
  (question-tool label, asked from the jihoon-a5 session). Answered "when they conflict, does G2 (H200
  kernels at serving's shapes) rank above G3 (top 50 on the overall kernel board by 11-05, which only
  moves our serving runs to the front of the queue)?"
- Effect: GOAL subgoal ranking G1 > G3 > G2 until 2026-11-05 17:00 KST, then G1 > G2.

## R-20261024-2 — 2026-10-24 03:09 KST — optimize the score, spend everything else
> Ok. aware of that quota. submit regularly. and please carefully discriminate the things to optimize and things to spend. for example, if there is no constraint about the memory, we can just freely spend the resources. if power not specified, we can use more power. just focus on our optimization and treat everything else as a resource. do not treat it as a resource and try to minimize it together.

> Yes, add to GOAL (Recommended)

- Source: transcript:9505; the yes at transcript:9779 (asked whether to record "optimize only the score;
  everything not scored and not bounded by a hard limit ... is a resource to spend, never minimized").
- Effect: GOAL "Optimize vs spend". "submit regularly" was carried out once: `okc_run.py send` takes a
  scored run while the day's last.

## Dropped agent rules (do not reintroduce)

| Rule an agent introduced | Dropped | Match |
|---|---|---|
| Stop opening new optimization branches after two failed ideas in a row (prefill lane) | 2026-10-23: a self-imposed stop; no owner or external source | `no new optimization branches` |
| "Keep a portable fallback path where that's cheap" (old AGENTS) | 2026-10-23: replaced by the practice of local stand-ins that test Hopper glue | `portable fallback path` |
| proj is "worth only enough work to keep rank" (old CEILING) | 2026-10-23: an effort cap with no source; proj kernels also feed serving | `worth only enough work` |
| Lanes wait for queued judge receipts before starting new work | 2026-10-23: idle waiting; `okc_run.py collect` lands receipts on its own | `no new checks will be sent` |
