# Rulings

Jihoon's words behind current standing decisions, verbatim, oldest first: a rule Jihoon confirmed, the
G0 and ranking decisions, answers to agents' questions. Ids are `R-<YYYYMMDD>-<n>`. A chosen
question-tool label is Jihoon's word; the question and option text are agent context. Transcript:
`~/.claude/projects/-home-jihoon-competitions-muse-kernel/ae5ada91-d8f5-49d0-a518-22df30a603ba.jsonl`.

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
  is allowed at the allowed formats. Prebuilt image binaries and JIT builds are allowed.
- Conflict: approximation below the allowed precision (FP8/FP4 activations, fewer experts, skipped
  keys) breaks hard limits 2 and 5-7. The hard limits win (llms.txt:107-110, 137). Our own cubins and
  SASS on the kernel boards: R-20261023-6.

## R-20261023-5 — 2026-10-23 21:55 KST — Codex default, never gpt-6-sol: kept as rules
> Keep both (Recommended)

- Source: transcript:6054. Answered "Keep these two instructions as standing rules: 'for subagents, use
  codex rescue as a default' and 'don't use gpt-6-sol from now on'?" The words behind them (2026-10-23):
  "for subagents, use codex rescue as a default." (transcript:2211) and "don't use gpt-6-sol from now on.
  there is no reason to use it. we can just use gpt-6.1-sol instead." (transcript:2755)
- Effect: rule 11; rule 12 is now set by R-20261023-9.
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

## R-20261023-9 — 2026-10-23 22:55 KST — Codex models: gpt-6.1-sol and gpt-6-astra
> use gpt-6.1-sol. not old models.

> use 6.1-sol and 6-astra.

> Retry gpt-6.1-sol only (Recommended)

- Source: transcript:6579 (after lanes had been rerouted to gpt-5.6-sol, gpt-6-luna and gpt-6-astra on
  capacity errors); transcript:6649 answered "Make 'use gpt-6.1-sol, not old models' a standing rule
  (extending rule 16, now rule 12 ...)?" and "When gpt-6.1-sol itself is at capacity ..., what should a stopped lane do?"
- Effect: rule 12.
- Confirmed: the answer to the standing-rule question.

## R-20261024-2 — 2026-10-24 03:09 KST — optimize the score, spend everything else
> Ok. aware of that quota. submit regularly. and please carefully discriminate the things to optimize and things to spend. for example, if there is no constraint about the memory, we can just freely spend the resources. if power not specified, we can use more power. just focus on our optimization and treat everything else as a resource. do not treat it as a resource and try to minimize it together.

> Yes, add to GOAL (Recommended)

> of course, our loop for fast iteration and fast experiments should be the optimization target. but for the competition, our all other preprocessing and practices are not our scope. we have resource to spend. and resource to optimize. and we should discriminate it.

- Source: transcript:9505; the yes at transcript:9779; the refinement (loop speed is a target) at transcript:9835 (asked whether to record "optimize only the score;
  everything not scored and not bounded by a hard limit ... is a resource to spend, never minimized").
- Effect: GOAL "Optimize vs spend". "submit regularly" was carried out once: `okc_run.py send` takes a
  scored run while the day's last.

## R-20261024-3 — 2026-10-24 10:40 KST — H200 lab runs in prepare()
> Allowed (Recommended)

- Source: question-tool answer in this session (asked whether lanes may keep running unrelated H200 benchmarks
  inside prepare() of a proj/MoE submission via `tools/h200_lab.py`, given llms.txt's "do work outside the timed
  calls"; the reading offered: that clause targets moving the timed computation out of the timed calls, and a lab
  computes nothing the timed calls return).

## R-20261024-4 — 2026-10-24 20:30 KST — 1st on the overall kernel board is G0 until the close
> our goal is to reach 1st place in the leaderboard. https://openkernelchallenge.dev/leaderboard. so in SOTA docs, include the result in the leaderboard-based score as well. so, what is needed more to reach much higher score?

> Kernel #1 replaces G0

- Source: transcript:19287 (message) and transcript:19398 (question-tool label). Answered ""1st place in the
  leaderboard": which board do you mean, and should it change the goal tree?" The chosen option read:
  "1st on the overall kernel board becomes the ultimate goal until 11-05. Serving-only work (prefill port,
  TTFT margin, HiCache) drops below every decode-layer lever until the kernel close."
- Effect: the board in G0 is the overall kernel board (openkernelchallenge.dev/leaderboard), aiming for 1st
  place. SOTA keeps a leaderboard section (overall rank and modeled overall score). R-20261025-1 makes it the
  only goal.

## R-20261025-1 — 2026-10-25 00:01 KST — the board score is the only goal
> I think we missed some context. our only goal for now is having high score in leaderboard. that's the only target. there is no other goals now. serving will be studied in the serving round. if any documents are mentioning about other goal, remove those and make the direction clear.

- Source: transcript:23005.
- Effect: G0 is the highest score on the overall kernel board until the kernel close; the subgoals are the
  decode-layer speedup and its stage kernels (GOAL). Serving goals, rules and work leave the live docs; their
  material and Jihoon's serving answers are parked in `docs/serving/` until the serving round opens.

## R-20261025-2 — 2026-10-25 18:10 KST — port NathenFernandes's MIT proj kernels with attribution
> Port with attribution (Recommended)

- Source: question-tool answer in this session. The question: NathenFernandes (proj board #14) published their
  judged proj kernels under MIT (github.com/NathenFernandes/muse-bf16-proj @ 86aff75); on the same judge protocol
  they take 0.62-0.96x our time at every shape; how may we use their code? The chosen option read: "Review their
  CUDA, run it in our H200 labs, and put the winners into our decode layer with their MIT notice."
- Effect: that repo's code may be reviewed, built, run on the judge and shipped in our decode layer, keeping the
  MIT copyright and permission notice in the shipped source. Covers this repo only; other teams' code is asked
  case by case.

## R-20261025-3 — 2026-10-25 20:40 KST — H200 labs at import time of an attention submission
> Allowed (Recommended)

- Source: question-tool answer in this session. The question: the proj, MoE and prefill lab hosts were out of checks
  until 09:00 KST; attention had 20 checks but no prepare(); may labs run in the attention submission at import time,
  untimed and computing nothing kernel_fn returns, the same reading as R-20261024-3? The chosen option read: "Same
  principle as prepare-time labs: nothing the timed calls return is computed outside them."
- Effect: `tools/h200_lab.py` may use muse_attention_decode as a host, running payloads at module import.

## R-20261026-1 — 2026-10-26 20:08 KST — port operatorstack and lee101/openkernels with attribution
> Both, with attribution (Recommended)

- Source: question-tool answer in this session. The question: operatorstack/open-kernel-challenge-solutions (MIT,
  team bigboateng, attention 1.8228x) and lee101/openkernels (Apache-2.0, attention v56 1.8429x off-board) beat our
  attention-board 1.6696x; may we use them on R-20261025-2's terms? The chosen option read: "Same terms as
  R-20261025-2 for both repos, all their kernels (attention first, then MoE/proj ideas). Ship winners with the MIT /
  Apache-2.0 notices."
- Effect: both repos' code may be reviewed, run in our H200 labs and shipped in our decode layer, keeping the MIT
  notice, or the Apache-2.0 license and any NOTICE text, in the shipped source. Other teams' code is still asked case
  by case.

## R-20261103-1 — 2026-11-03 21:15 KST — public repo, renamed, with its full history
> I want to open it now. so, clean up the repo. make it readable, and open to public.

> rename "https://github.com/AutoclickerI/muse_kernel" as well. (2026-11-03 23:58 KST)

- Source: Claude Code session `0045ab8f-54dd-41a4-8c54-fd714e9ef389`. Question-tool answers: "Publish with history",
  "Now, include serving (Recommended)", and for scope "showcase. but you don't have to clean up the repo research
  logs. we can just keep the logs as-is."
- Effect: the repo is public as github.com/AutoclickerI/muse-spark-open-kernels (renamed from `muse_kernel`; the old
  URL redirects) with its full history. README.md and `final/` are the front
  door; the research logs stay as they were. Replaces R-20261023-2 of 2026-10-23 ("Private GitHub repo").

## Dropped agent rules (do not reintroduce)

| Rule an agent introduced | Dropped | Match |
|---|---|---|
| Stop opening new optimization branches after two failed ideas in a row (prefill lane) | 2026-10-23: a self-imposed stop; no owner or external source | `no new optimization branches` |
| "Keep a portable fallback path where that's cheap" (old AGENTS) | 2026-10-23: replaced by the practice of local stand-ins that test Hopper glue | `portable fallback path` |
| proj is "worth only enough work to keep rank" (old CEILING) | 2026-10-23: an effort cap with no source | `worth only enough work` |
| Lanes wait for queued judge receipts before starting new work | 2026-10-23: idle waiting; `okc_run.py collect` lands receipts on its own | `no new checks will be sent` |
