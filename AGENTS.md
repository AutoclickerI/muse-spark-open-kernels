# AGENTS

Our entry in the Muse Spark Open Kernel Challenge (openkernelchallenge.dev): make SGLang serve Muse
Spark Open faster on H200s. Owner: Jihoon.

**Before any work, read [docs/GOAL.md](docs/GOAL.md) and [docs/CONSTRAINTS.md](docs/CONSTRAINTS.md).**
On demand: [PLAN](docs/PLAN.md) · [RESOURCES](docs/RESOURCES.md) · [FINDINGS](docs/FINDINGS.md) ·
[SOTA](docs/SOTA.md) · [RULINGS](docs/RULINGS.md) · `docs/reference/` (each file opens with when to read it).
The organizers' files are authoritative: `upstream/llms.txt`, `problems/<name>/TASK.md` and `AGENTS.md`,
`serving/RULES.md`.

## Practice

- Next work: the highest-ranked `active` or `next` row in [PLAN](docs/PLAN.md) "Ranked work" that no lane owns.
- The critical path is judge latency (PLAN "Critical path"): never idle on it; use idle GPUs, caches and
  other lanes' outputs (RESOURCES).
- Take the widest reading the rules allow (CONSTRAINTS "Boundary cases"); ask Jihoon when unclear.
- No conservative or incremental steps, fallbacks or ritual checks; delete stale docs when you touch them.
- Remote runs only via `tools/okc_run.py send <file> --note "<what changed>"` (default): a scored run while the
  day's last (the board keeps our best), receipts in `runs/`, priority queue. It detaches itself (log in `runs/detached/`); `collect` finishes orphans.
- Screen with `tools/local_eval.py` first; a candidate that fails a local test doesn't go remote (default).
  `tools/screen_all.py --send` screens every unsent candidate in parallel and sends each pass.
- Time any kernel on the H200 without a candidate or the target problem's quota: `tools/h200_lab.py send <payload>`
  (runs in the prepare of proj's best source; proj and MoE have spare runs). Decode's 20 runs a day go to candidates.
- Local timing (sm_120) ranks loosely; H200 claims come only from `runs/` receipts.
- Hopper-only paths can't run here. Force each Hopper branch's Python glue locally with a portable stand-in kernel.
- Write kernels so they drop into `serving/sglang/python/sglang/`. Serving runs only verify loads and prefill.
- Run `tools/sync_upstream.sh` before starting; read the diff if official files changed.
- Delegate per CONSTRAINTS 15-16 (resume command: RESOURCES "Inventory").

## Commands

README "Kernel loop" (local screen, check, submit); RESOURCES "Setup" (judge image, `tools/img`, harness).
