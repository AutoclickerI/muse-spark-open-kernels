# AGENTS

Our entry in the Muse Spark Open Kernel Challenge (openkernelchallenge.dev): the highest score on the overall
kernel board, which is our decode-layer speedup. Owner: Jihoon.

**Before any work, read [docs/GOAL.md](docs/GOAL.md) and [docs/CONSTRAINTS.md](docs/CONSTRAINTS.md).**
On demand: [PLAN](docs/PLAN.md) · [RESOURCES](docs/RESOURCES.md) · [FINDINGS](docs/FINDINGS.md) ·
[SOTA](docs/SOTA.md) · [RULINGS](docs/RULINGS.md) · `docs/reference/` (each file opens with when to read it).
The organizers' files are authoritative: `upstream/llms.txt`, `problems/<name>/TASK.md` and `AGENTS.md`.
`docs/serving/` is parked until the serving round; don't work from it.

## Practice

- Next work: the highest-ranked `active` or `next` row in [PLAN](docs/PLAN.md) "Ranked work" that no lane owns.
- Judge every change by the decode layer's score (GOAL "Terms"; sensitivity: `problems/muse_decode_layer/FOCUS.md`).
- The critical path is judge latency (PLAN "Critical path"): never idle on it; use idle GPUs, caches, other lanes' outputs.
- Take the widest reading the rules allow (CONSTRAINTS "Boundary cases"); ask Jihoon if unclear.
- No conservative or incremental steps, fallbacks or ritual checks; delete stale docs when you touch them.
- Remote runs only via `tools/okc_run.py send <file> --note "<what changed>"` (default): scored while any are
  left (the board keeps our best); receipts in `runs/`, log in `runs/detached/`; `collect` finishes orphans.
- Screen with `tools/local_eval.py` first; a candidate that fails a local test doesn't go remote (default).
  `tools/screen_all.py --send` screens every unsent candidate in parallel and sends each pass.
- H200 timing without decode runs: `tools/h200_lab.py send <payload> [more ...]` (prepare of a host's best source:
  proj, MoE or prefill). Synthesize weights a host lacks; never wait on a spent host. Batch labs per host.
  Decode's 20 runs a day go to candidates.
- Local timing (sm_120) ranks loosely; H200 claims come only from `runs/` receipts.
- Hopper-only paths can't run here: force each Hopper branch's glue locally with a stand-in kernel.
- Run `tools/sync_upstream.sh` before starting; read the diff if official files changed.
- Delegate per CONSTRAINTS 11-12 (resume command: RESOURCES "Inventory").

## Commands

README "Kernel loop" (local screen, check, submit); RESOURCES "Setup" (judge image, `tools/img`).
