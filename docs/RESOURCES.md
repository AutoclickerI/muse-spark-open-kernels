# Resources
Everything usable for G0. Binding usage rules are in CONSTRAINTS (1, 8, 9, 10). Measure capacity,
usage and free space when needed (`nvidia-smi`, `df -h /data2`, `okc problems`); they aren't written here.

## Setup
From a fresh checkout to a local screen, about 5 minutes when the image and caches below already exist:
1. `tools/sync_upstream.sh` (seconds): refreshes `upstream/` and the official problem files; read `git diff`.
2. `python3 tools/local_eval.py <file> --tier static` first (no Torch/import/prepare/GPU), then
   `CUDA_VISIBLE_DEVICES=0 tools/img python tools/local_eval.py <file> --tier changed --against <previous.py>`
   for changed native units. One `--tier wrapper --transport prepare|import|lab-only` smoke per transport.
   `--tier screen` (the existing default) runs TASK tests and local graph timing, two replays/one input by default;
   `--tests-only` keeps its meaning. Full held-out paired confirmations run on the H200.
3. `python3 tools/okc_run.py send <file> --lane <P-id> --decision "..." --note "..."` (hours of queue; see PLAN "Critical path"): a
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
| Judge image `lmsysorg/sglang@sha256:8cfc5a72…` | Exact software of the judge: torch 2.14.1+cu130, Triton 3.8, sgl_kernel 0.4.9, flashinfer 0.7.0.post1, DeepGEMM 0.2.1.post1+muse, TileLang 0.1.12, CUTLASS DSL 4.8, humming_kernels 0.1.12 | `tools/img <cmd>` runs the sandbox `/data2/jihoon/muse_kernel/images/sglang-dev-muse-spark-open-8cfc5a72.sandbox` (`--nv --contain --cleanenv`). Rebuild it with `apptainer build --sandbox <dir> docker://lmsysorg/sglang@<digest>` (the digest `GET /api/v1/serving` pins). JIT caches are in `/data2/jihoon/muse_kernel/img-home` (its `bin/ruff`, a copy of the host's ruff 0.16.6, serves local_eval's undefined-name check), and `/tmp` maps to `.../img-work`. There is no squashfuse, so never run a `.sif` here |
| Host env `~/anaconda3/envs/comp_kv` (torch 2.13, cu130) | Tests without the image (`--no-baseline`) | `baseline.py` needs `sglang` |
| `okc` 0.7.1 (`~/.okc/bin`) | Boards (`okc leaderboard <p> --json`), `okc problems`, `okc withdraw` | Never `okc check`/`submit` directly (AGENTS) |
| `tools/screen_all.py` | Screen candidates in parallel and queue each pass immediately, including while other screens run | Logs in `runs/screen/` (ignored). `--send --lane <P-id> --decision "..."` uses `okc_run.py send`; OOM retries alone do not hold successful peers |
| `tools/h200_lab.py` | H200 measurements on any host; time/output/risk planned batches | Existing flags and `lab(deadline, weights[, state]) -> list[str]` still work. New payloads return structured packets and declare `LAB_RESOURCES` (below); `--lab-only` skips the scored base and runs at import on any host, always as a check; `--phase import`, `--import-budget` (150 s default), `--base-file`, `--not-before`, `--check`, `--priority` remain. `read <job>` recovers packets/old lines; programmatic readers use `lab_packet` |
| `sim/` | Bounds and ceilings: `python -m sim` (problems), `-m sim.calibrate` (SGLang's measured times), `-m sim.bound` (`docs/reference/bound.md`) | Model inputs: `sim/routing.json`, `sim/hw.py` |
| `prof/` | Lab payloads for `h200_lab.py` (`prof/lab/`), a profile report from saved receipts (`prof/write_report.py`), the embedded-source extractor for SASS inspection (`prof/compile_static.py`, output `prof/sass/`); the rest are parked serving probes | Outputs are ignored; regenerate rather than edit |
| `runs/calibration/` | SGLang's own kernels profiled on the judge, per problem | `runs/calibration/profiles/*_sglang.*.txt` |
| Per-problem lanes' outputs | Candidates `problems/<p>/cand_*.py`; each lane's live work in `problems/<p>/dev/<lane>/` (its README has the handoff and next step) | Reuse a component another lane already measured on the H200 |
| Codex (`codex:codex-rescue`) | Delegated lanes | Models and rerouting: CONSTRAINTS 12. gpt-6.1-sol's capacity comes and goes by the minute. Out of quota looks like a job stuck in `starting` or `retrying` for hours; `~/.hermes/logs/errors.log` then shows `local_auth_cap status=429` for every credential (2026-10-25 02:05 KST; 2026-10-26 13:24 KST), `~/.hermes/logs/agent.log` says when it clears (`retry after <s>`). A capacity error on one model is retried on gpt-6.1-sol (CONSTRAINTS 12); a quota stop on every model is the global instructions' case for Claude subagents, so critical-path lanes move there until it clears. To resume a stopped job: `CODEX_RESUME_THREAD_ID=<session> node ~/.local/src/codex-plugin-cc/plugins/codex/scripts/codex-companion.mjs task --background --write --model <m> "<delta>"`. Tell every lane not to call `spawn_agent`: the Hermes proxy rejects a sub-agent's report back (`unsupported_websocket_event`) and the parent's turn dies (three attention turns, 2026-10-23). A job can also stall for hours on `Codex retrying (responseStreamDisconnected)` with status still `running`: watch its log's age, cancel, and resume the thread (a thread whose cancelled turn still holds it answers `already has an active writer`: start fresh with a self-contained brief). A lane's child processes can die when its turn ends; `okc_run.py send` starts its own session, so a lane's send survives it |
| Repo | Code and docs | github.com/AutoclickerI/muse_kernel, private (R-20261023-2) |

## Experiment loop (P1)

Every sending lane: add `--lane P4|P10|P3a|P3b|P5b|P5|P7` for the decision served, and
`--decision "one line describing what this run settles"` to both send tools. The current PLAN rank orders
the local queue on every host; waiting entries refresh ranks when dispatching. Both fields reach the ledger.
Calls without a lane remain accepted; after P1 closes they are recorded as `lane=unspecified` and queue
after the ranked decisions, independently of host. An existing explicit `--priority` still works without `--lane`. With a lane, PLAN rank takes precedence over `--priority`.

Migration for lanes with old commands: keep existing `--check`, `--scored`, `--not-before`, `--phase`,
`--import-budget`, `--base-file`, timing flags and optional numeric `--priority`. Add `--lane <active P-id>`
and `--decision "what this run settles"` for PLAN scheduling; stop using `--lane P1` after its row closes.
Unlabeled calls keep working but no longer borrow a completed lane's rank. Add `LAB_RESOURCES` to share a run;
undeclared payloads keep the old interface and reserve a full packet. Replace raw `log_tail.splitlines()`
parsers with `lab_packet.result_lines()` or `read_packets()` so encoded results remain visible.

Payload declarations and output:
```python
LAB_RESOURCES = {"compile_s": 30, "run_s": 40, "output_bytes": 2000, "context_poison": "none"}
def lab(deadline, weights, state):
    # Run the experiment; preserve every seed pair and its source identities.
    return {"packet": {"identities": identities, "seeds": seeds, "paired": pairs,
                       "checks": task_and_nearby_checks, "complete": complete},
            "logs": diagnostics}
```
`output_bytes` reserves the complete encoded `LAB_PACKET` line plus newline (512..8744 bytes); budget an
additional 256 bytes per payload for markers. Measure with `len(lab_packet.encode_packet(name, sha, packet).encode())+1`.
The packet includes a SHA256 checksum; import batches print early and reprint after host prepare so setup
failures or tuning logs do not erase evidence. Readers verify every emitted copy and collapse identical
packets in their final emission order; conflicting copies fail verification.
Ordinary transports reserve 9,000 bytes total, leaving 7,000 characters of the judge's 16,000-character tail
for profiler/errors. `--lab-only` reserves 12,000 bytes and leaves 4,000: its deliberate prepare stop skips
timing/profiling. The per-payload maximum stays 8,744 bytes, allowing one maximum packet and small peers
in a single lab-only check.
Logs only use unreserved space. A too-large actual packet is an explicit `LAB_PACKET_OVERSIZE` failure,
never truncated or treated as measured evidence: split that experiment or fix the declaration before sending.
The CLI splits oversized batches into separate runs; `build -o lab.py` writes `lab.py`, `lab.2.py`, etc.
The Python `build()` API requires exactly one planned batch; call `plan_batches()` first for multiple batches.

Time per payload is declared compile+run time; prepare batches subtract `--base-prepare <seconds>` (default
130 s) from 285 s, while import batches use `--import-budget`. Deadlines are proportional to declared work,
using any remaining time; payloads must still honor them. `context_poison="last"` puts a probe last;
`"device"` isolates it in its own run. There is no payload-count cap. Old payloads without declarations
reserve a whole output allocation, so they run separately until their authors declare smaller requirements.
A waiting batch is merged only with matching host/base/phase/budget and decision metadata, using its exact snapshot.

`--lab-only --import-budget 700` supplies the minimal self-preparing import host on any problem; weights/state
are None and the payload owns synthetic setup. It deliberately stops the judge at prepare after emitting
its packet. It spends check quota, and the host's failure is separate from the experiment's TASK evidence.
Use `--phase import` with a real base when its own official timing is also wanted.

Readers: `tools/h200_lab.py read <job>` verifies checksums and displays evidence. Code that used to search
`receipt['result']['log_tail'].splitlines()` must use `lab_packet.result_lines(tail)` for existing line results,
or `read_packets(tail)` for structured packets. S18/S19 readers already use `paired_evidence`; archived XZ85
receipts still replay. The old list-of-lines interface is preserved losslessly, with completeness marked unknown
at the transport level; new experiments should emit the explicit evidence fields above.

Promotion: S18 and S19 call `tools/promotion.py.evaluate(receipt, raw, manifest)`; new integration directories
should do the same. Host identity is absent from the gate. Candidate/incumbent/checker identities, disjoint
complete primary seeds, native positive measurements, TASK checks per arm, nearby checks, valid A:A control,
and the six-load R lower bound decide promotion. Emit `arm_status` per arm with `sha256`, `native`, `complete`,
`seed_ids`, `task_checks`, `expected_task_checks`, `nearby_task_checks`, `nearby_expected_checks`, `failures`.
Failures are `{scope: "arm"|"device"|"run", arms: [...], reason: ...}`; an unscoped fault voids every arm.
Optional-arm failures leave primary R measured. A missing official pass or nearby checks withholds promotion;
invalid primary/control evidence is unmeasured. Official TASK receipts must match the candidate shipment,
e.g. `dev/s19/report.py <lab-receipt> --official-task <decode-receipt>`.

Local tiers keep the existing screen flags. `--tier changed --against <previous.py>` compiles changed literal
native sources for sm_90a; `--unit <SOURCE_NAME>` names units explicitly and is repeatable. `--payload` allows
static/changed checks of headerless lab files. Static state-key checking proves closed literal prepare dictionaries;
dynamic state/templated sources still require the lane's targeted checks and H200 evidence. Wrapper smokes use
tiny stand-ins and prove transport only. S18's wrapper scripts now smoke transport rather than replay full
confirmation seeds locally. The ordinary TASK checker remains the correctness criterion.

Regression suite: `python3 -m unittest discover -s tools/tests -v`. It replays `5d0282bb` on both host paths,
faults an optional arm in the real S18 producer using tiny stand-ins, checks maximum-packet tail preservation,
resource splits and CLI flags, static/changed/wrapper tiers, PLAN ranks and ledger fields, and streaming screens.
Judge-image checks and a real changed-unit sm_90a compile are recorded locally under `runs/screen/p1/` (ignored).
Live transport proof: proj H200 check `1acc6c67671c4220` retained three valid checksummed packets, including
an exact 8,744-byte packet, isolated a deliberate optional failure, completed its following peer, and
recorded P1/decision/PLAN priority 2 in the ledger. Evidence/replay: FINDINGS "Judge, quotas and loop",
`runs/loop/p1_live_20261026/`, `tools/tests/live_loop.py`. The lab-only host deliberately stops at prepare;
its failed host status does not invalidate complete payloads.

## Consumers
Files outside the repo that agents read for this project. They point to rules and never restate them.
- `/home/jihoon/.claude/projects/-home-jihoon-competitions-muse-kernel/memory/*.md`
