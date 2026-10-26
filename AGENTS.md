# AGENTS

Our entry in the Muse Spark Open Kernel Challenge (openkernelchallenge.dev): the highest score on the overall
kernel board, which by our fit (GOAL) is our decode-layer speedup. Owner: Jihoon.

**Before any work, read [docs/GOAL.md](docs/GOAL.md) and [docs/CONSTRAINTS.md](docs/CONSTRAINTS.md).**
On demand: [PLAN](docs/PLAN.md) · [RESOURCES](docs/RESOURCES.md) · [FINDINGS](docs/FINDINGS.md) ·
[SOTA](docs/SOTA.md) · [RULINGS](docs/RULINGS.md) · `docs/reference/`.
Organizer files are authoritative: `upstream/llms.txt`, `problems/<name>/{TASK,AGENTS}.md`.

## Practice

- Next work: your brief's row. Without a brief you orchestrate: a lane on every `active` or `next` row of
  [PLAN](docs/PLAN.md) "Ranked work"; read receipts, integrate winners, fold results into the docs.
- Spend on the levers whose ceilings can close the GOAL target (PLAN "Levers"), at the cheapest fidelity that
  ranks them; when a ceiling, mechanism or the order changes, a fresh agent re-ranks Levers while work goes on.
- Judge runs are the critical path (PLAN "Critical path"): never idle; each run goes to the next decision
  worth most to the score (PLAN rank), on any host with quota.
- Take the widest reading the rules allow (CONSTRAINTS "Boundary cases"); ask Jihoon if unclear.
- Split a large idea into small experiments that decide it (oracles, bounds); fold small wins into the next
  integration batch, not separate confirmations. No fallbacks or ritual checks; delete stale docs you touch.
- Remote runs only through `tools/okc_run.py send <file> --note "<what changed>"` (scored while any are left) or
  `tools/h200_lab.py send <payload> [more ...]` (labs, batched per host); receipts in `runs/`. Screen with
  `tools/local_eval.py` first.
- Local runs (sm_120) rank timing loosely and can't run Hopper paths: force each Hopper branch's glue with a stand-in
  kernel; H200 claims need `runs/` receipts.
- The orchestrator runs `tools/sync_upstream.sh` and reads the diff; lanes don't.
- Delegate per CONSTRAINTS 11-12 (recovery: RESOURCES "Inventory").

## Commands

README "Kernel loop" (local screen, check, submit); RESOURCES "Setup" (judge image, `tools/img`).
