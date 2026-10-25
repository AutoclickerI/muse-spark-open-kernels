# Resources
Everything usable for G0. Binding usage rules are in CONSTRAINTS (1, 8, 9, 10). Measure capacity,
usage and free space when needed (`nvidia-smi`, `df -h /data2`, `okc problems`); they aren't written here.

## Setup
From a fresh checkout to a local screen, about 5 minutes when the image and caches below already exist:
1. `tools/sync_upstream.sh` (seconds): refreshes `upstream/` and the official problem files; read `git diff`.
2. `tools/img python tools/local_eval.py problems/<p>/submission.py` (1-5 min): tests plus CUDA-graph
   timing against `baseline.py`, inside the judge's image. `--tests-only` checks correctness only.
3. `python3 tools/okc_run.py send <file> --note "..."` (hours of queue; see PLAN "Critical path"): a
   scored run while the problem has one left today, else a check, else at the 09:00 KST reset.
   It needs the okc token that Jihoon's `okc login` created.

The image and its unpacked sandbox already exist on `/data2`; reuse them. JIT caches persist, so a
second compile of the same kernel is fast.

## Inventory
| Resource | Use for | Access and non-obvious limits |
|---|---|---|
| The judge: 1 H200 SXM per kernel run | The only H200 timing | `tools/okc_run.py` (receipts in `runs/`, ledger `runs/ledger.jsonl`, waiting in `runs/pending/` and `runs/waiting/`). Quotas: CONSTRAINTS 9. No cancel, only `okc withdraw` |
| This box: 2x RTX PRO 6000 Blackwell (sm_120) | Correctness, graph capture, loose ranking, nsys, cuobjdump/nvdisasm, nvcc 13 targeting sm_90a | `CUDA_VISIBLE_DEVICES=0` or `1` per lane. ncu is blocked (`RmProfilingAdminOnly=1`). Hopper-only paths (wgmma, TMA, FA3, sm90 DeepGEMM/CUTLASS) don't run |
| capp DGX `edgexpert0/1`, `zgx0/1`: GB10 (sm_121, LPDDR5x about 273 GB/s) | Extra correctness lanes; useless for H200 timing | Shared machines; check what else runs there first (CONSTRAINTS 10) |
| Judge image `lmsysorg/sglang@sha256:8cfc5a72…` | Exact software of the judge: torch 2.14.1+cu130, Triton 3.8, sgl_kernel 0.4.9, flashinfer 0.7.0.post1, DeepGEMM 0.2.1.post1+muse, TileLang 0.1.12, CUTLASS DSL 4.8, humming_kernels 0.1.12 | `tools/img <cmd>` runs the sandbox `/data2/jihoon/muse_kernel/images/sglang-dev-muse-spark-open-8cfc5a72.sandbox` (`--nv --contain --cleanenv`). The `.sif` is beside it. JIT caches are in `/data2/jihoon/muse_kernel/img-home` (its `bin/ruff`, a copy of the host's ruff 0.16.6, serves local_eval's undefined-name check), and `/tmp` maps to `.../img-work`. There is no squashfuse, so never run the `.sif` directly |
| Host env `~/anaconda3/envs/comp_kv` (torch 2.13, cu130) | Tests without the image (`--no-baseline`) | `baseline.py` needs `sglang` |
| `okc` 0.7.1 (`~/.okc/bin`) | Boards (`okc leaderboard <p> --json`), `okc problems`, `okc withdraw` | Never `okc check`/`submit` directly (AGENTS) |
| `tools/screen_all.py` | Screen every candidate the judge hasn't measured, in parallel (tests and graph capture in the image), then send each pass | Logs in `runs/screen/` (ignored). Queues each pass with `okc_run.py send` |
| `tools/h200_lab.py` | Any H200 measurement (kernel variants at any shape, library kernels in the judge's image) without a candidate | Payload `lab(deadline, weights) -> list[str]` appended to a proj, MoE, prefill, decode or attention best scored source; prepare-time budget ~280 s, attention import-time budget `--import-budget` (default 45 s, weights/state None); `read <job>` waits for its lines; 9,000 characters per run |
| `sim/` | Bounds and ceilings: `python -m sim` (problems), `-m sim.calibrate` (SGLang's measured times), `-m sim.bound` (`docs/reference/bound.md`) | Model inputs: `sim/routing.json`, `sim/hw.py` |
| `prof/` | Judge-profile analysis (`prof/analyze.py`), the generator of `docs/reference/profile.md` (`prof/write_report.py`), and the SASS audit (`prof/compile_static.py`, `prof/sass/`) | Writes snapshots; regenerate rather than edit |
| `runs/calibration/` | SGLang's own kernels profiled on the judge, per problem | `runs/calibration/profiles/*_sglang.*.txt` |
| Per-problem lanes' outputs | Candidates `problems/<p>/cand_*.py`, judge profiles `problems/<p>/profiles/`, notes `problems/<p>/dev/` | Reuse a component another lane already measured on the H200 |
| Codex (`codex:codex-rescue`) | Delegated lanes | Models and rerouting: CONSTRAINTS 12. gpt-6.1-sol's capacity comes and goes by the minute. Out of quota looks like a job stuck in `starting` or `retrying` for hours; `~/.hermes/logs/errors.log` then shows `local_auth_cap status=429` for every credential (2026-10-25 02:05 KST), and lanes move to Claude subagents until it clears. To resume a stopped job: `CODEX_RESUME_THREAD_ID=<session> node ~/.local/src/codex-plugin-cc/plugins/codex/scripts/codex-companion.mjs task --background --write --model <m> "<delta>"`. Tell every lane not to call `spawn_agent`: the Hermes proxy rejects a sub-agent's report back (`unsupported_websocket_event`) and the parent's turn dies (three attention turns, 2026-10-23). A job can also stall for hours on `Codex retrying (responseStreamDisconnected)` with status still `running`: watch its log's age, cancel, and resume the thread (a thread whose cancelled turn still holds it answers `already has an active writer`: start fresh with a self-contained brief). A lane's child processes can die when its turn ends; `okc_run.py send` starts its own session, so a lane's send survives it |
| Repo | Code and docs | github.com/AutoclickerI/muse_kernel, private (R-20261023-2) |

## Consumers
Files outside the repo that agents read for this project. They point to rules and never restate them.
- `/home/jihoon/.claude/projects/-home-jihoon-competitions-muse-kernel/memory/*.md`
