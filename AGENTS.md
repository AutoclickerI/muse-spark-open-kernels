# AGENTS

Our entry in the Muse Spark Open Kernel Challenge (openkernelchallenge.dev): make SGLang serve Muse
Spark Open faster on H200s. Owner: Jihoon.

**Before any work, read [docs/GOAL.md](docs/GOAL.md) and [docs/CONSTRAINTS.md](docs/CONSTRAINTS.md).**
On demand: [PLAN](docs/PLAN.md) · [RESOURCES](docs/RESOURCES.md) · [FINDINGS](docs/FINDINGS.md) ·
[SOTA](docs/SOTA.md) · [RULINGS](docs/RULINGS.md) · `docs/reference/` (each file opens with when to read it).
The organizers' files are authoritative: `upstream/llms.txt`, `problems/<name>/TASK.md` and `AGENTS.md`,
`serving/RULES.md`.

## Practice

- Use idle capacity, caches and other lanes' outputs (RESOURCES); fix the critical path (PLAN).
- Take the widest reading the rules allow (CONSTRAINTS "Boundary cases"); ask Jihoon when unclear.
- No conservative or incremental steps, fallbacks or ritual checks; delete stale docs when you touch them.
- Remote runs only via `tools/okc_run.py ... --note "<what changed>"` (default): receipts in `runs/`, priority
  queue, at most 4 of ours at the judge. Launch it detached; `collect` finishes orphans.
- Screen with `tools/local_eval.py` first; a candidate that fails a local test doesn't go remote (default).
- Local timing is sm_120, not H200: use it to rank loosely. H200 claims come only from `runs/` receipts.
- Hopper-only paths can't run here. Force each Hopper branch's Python glue locally with a portable stand-in kernel.
- Time several variants inside one diagnostic check: the judge starts ~11 of our runs an hour.
- Write kernels so they drop into `serving/sglang/python/sglang/`. Serving runs only verify loads and prefill.
- Run `tools/sync_upstream.sh` before starting; read the diff if official files changed.
- Delegate to `codex:codex-rescue` (default). Reroute a capacity-stopped job by resuming its thread
  on another model (RESOURCES "Inventory").

## Commands

README "Kernel loop" (local screen, check, submit); RESOURCES "Setup" (judge image, `tools/img`, harness).
