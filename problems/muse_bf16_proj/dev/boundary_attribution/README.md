# P7 S14 boundary attribution and reset batch

Current state: **queued for 2026-10-26 09:00 KST on prefill; no H200 job ID yet**.
One batched run contains exact confirmation payload `6cb428bc` unchanged plus the boundary lab.
Nothing in this directory is promoted. Base is `cand_r3_best7_s14_final.py`, SHA256
`f1c77ce692055c89474eab38bd2b45ec1ea34c94cf5a93d1992dbbcf448a3cac`.

## What the apparent 20–25 us means

`reconcile_s11.py` reproduces `s11_accounting.json` from the lever JSON and timeline README.
The two S11 measurements use different fixtures; these numbers must **not** be subtracted as a
matched experiment. Public load weights are sliding=46, full=16.

| Load | Lever boundary residence/layer | Lever excess weighted us | Timeline b1+b2 completion increment/layer |
|---|---:|---:|---:|
| decode 16 | 19.952 us | 1235.624 | 5.149 us |
| decode 32 | 24.932 us | 1544.424 | 5.924 us |

The lever adds kernel residence, including time resident while waiting on a programmatic predecessor.
Its boundary floor is **0.017067 us/layer**, constant across these loads: it omits the token-scaled
partial/residual traffic and the cold-read ramp. The S11 timeline has b1 increments 2.40–2.94 us and b2
2.66–3.14 us, despite much longer residence. At 1.67–3.28 MB reads, the 2.36 us cold-read + bytes/4.58 TB/s
model alone is about 2.72–3.08 us per boundary. Thus the +7.2% lever estimate does not establish
7.2% reclaimable boundary work. This is an accounting diagnosis, not a new S14 timing claim.

S14 source inspection: prepare selects CL8/256/one-round b1/b2 for all four target rows. Both already
combine split partials, branch normalization, residual update, and next-norm statistics in a single kernel.
The selected small-M expert path has a dependency wait but no early dependent release. There is no
selected dense-chain stream split for these rows. Whether a join or a late launch is actually exposed
is left to the captured S14 DAG and CUPTI intervals; source inspection cannot establish that timing.

## The reset experiment

`build_lab.py` embeds the exact `6cb428bc12b1180f6072a064ff9444dc7616ec2bdd3bc171ef3f13e26e3c99f8`
bootstrap and its original 12-fixture, 16-new-seed shared-JIT confirmation. It records the prepared S14
namespace/state, then reuses it for four decode-16/32 sliding/full fixtures, 16 further seeds, eight rotated
cold READ timing rounds. Unchanged JIT objects and native modules are shared. R/RP share their changed
MoE module; P/RP share their changed unpool kernel. A2 is a true same-handle A:A control.

| Arm | Change | Candidate SHA256 prefix |
|---|---|---|
| R | Release unpooler from the small-M MoE grid at entry; dependency wait still protects data | `290434ce` |
| P | First unpool weight tiles issue L2 requests before dependency wait; original async pipeline retained | `98437f67` |
| RP | R and P together | `51d89088` |
| C | CL4 rather than CL8 for b1/b2, target rows only | `48692d45` |
| S | Serialize boundary launches, target rows only; control for PDL residence/overlap | `684aaed6` |

R is an attribution prototype whose native source affects other small-M specializations too. Only the four
specified rows are measured here; any winning integration must restrict it to the measured rows and screen
that concrete patch before promotion. S compiles the identical GPU body into a separate module with the
host launch attribute changed; same-handle ready-input controls below provide an independent work estimate.

For A/RP/C/S, the lab records the whole-layer CUDA graph, all nodes and edges (including non-kernel joins),
programmatic edge type/ports, and a median-span CUPTI trace from five cold replays. Intervals, streams,
resources, stage frontiers, and b1/b2 producer overlap are retained. It also records unprofiled event timings
so CUPTI overhead is visible. Each original b1/b2 is replayed in isolation with the **same kernel handle and
actual ready buffers**; output/input footprints and cold-read models include partial dtype/splits and residual
traffic. Times before/after producer end are overlap/tail proxies, **not instruction-level wait cycles**.
`report.py` reconstructs completion frontiers and immediate DAG predecessors for b1/b2/unpooler/MoE.

The archive retains every seed, graph edge, kernel interval, name, and resource count. It interns metadata,
normalizes graph IDs reversibly, and encodes timing deltas; timing precision is 0.001 us. Derived frontiers
are reconstructed. The actual local receipt is 8,640 characters under the 9,000-character wrapper limit.

## Checked locally

- RP/C/S normal and forced screens: each 13 correctness tests, 12 captured timed shapes; static checks
  24/24/25 respectively, all passed. Receipts include exact source hashes.
- Native C4 and serialized controls: 96 changed-input replays at T16/32, BF16/FP32 split partials and both
  roles. Serialized results bit-exact. C4 changes reduction order; worst relative L2 `8.9573e-5`, below the
  explicit 0.001 component budget. Whole-layer output/routing gates remain unchanged.
- Exact sm90a unpool configurations (S2, NS4, TB16/32): original and hint retain respectively
  **48/64 registers, 40/48 LDGSTS, eight HGMMA**, identical shared memory. CCTL precedes the wait.
  All three PDL MoE specializations release before waiting; the fourth has no PDL wait.
- DAG API tested on a three-node fork/join and two driver-launched Triton PDL nodes, including a
  programmatic edge and its ports. The runtime kernel-parameter API rejects Triton driver handles;
  the tracer uses CUPTI graph IDs and activity names instead.
- Full portable batch orchestration: 12 confirmation + four boundary fixtures, 16 seeds each, eight rounds,
  all paired gates pass. Original output exceeded the receipt limit; after lossless structural compaction,
  `check_archive.py` round-trips every observation within declared timing quantization and verifies the cap.
  `local_raw.json.xz`, reconstructed `local_batch.json` and receipts are local sm120 evidence, **not H200 speedup evidence**.

## Schedule and handoff

`scheduled.json` records owned deferred worker **3517006**, prefill wrapper SHA256
`d29911c50f95545574f06aa4e7aecb2e4a21456916999e5948bb78f695dfec53`, held until epoch
1792972800 (2026-10-26 09:00 KST). Payload SHA256 is
`61e784c71648203147f3defda0b4db737afcf8111f4862d2c30dee1cc6804944`.
Its log is `runs/detached/d29911c50f955455.log`. The earlier owned worker 3230043 was verified by
PID, cmdline and wrapper hash and replaced before any judge send; `replaced_schedule.json` preserves it.
No other worker was canceled. Decode quota remains reserved for P10b/integration.

`tools/h200_lab.py --phase import` lets this self-preparing synthetic lab run on a prepare host without
passing incompatible host weights/state or sharing prepare's 300-second cap. The host's prepare and
kernel remain unchanged. Import labs cannot absorb prepare-time waiters. The batch owns one S14
prepare and has a 550-second import budget.

After reset, obtain the job from the detached log/ledger and run `python3 tools/h200_lab.py read JOB`.
Feed its text to `report.py RECEIPT.txt --out h200_batch.json`. Judge wins using paired whole-layer
ratios and the A:A control, separately by shape; do not promote from the profile spans or local timings.
The prior metadata patch remains provisional at `../pdl_audit/patch_metadata_s14.py` (`f39bc1f3`), applies
onto exact S14 `f1c77ce6`, and awaits the embedded shared-JIT confirmation. No new winning integration
patch exists yet. `patch.py` (`d74c009c`) builds the boundary prototypes onto that same exact S14 base.

Rebuild ignored payloads from committed sources:

```
python3 problems/muse_bf16_proj/dev/pdl_audit/build_followup.py --shared-jit
python3 problems/muse_bf16_proj/dev/boundary_attribution/build_lab.py
```

Next action after results: attribute b1/b2 intervals to their DAG producers and compare ready-input work,
serialized work and critical increments; freeze only paired winning rows, generate their S14 integration
patch, and rerun normal/forced screens. If most residence is waiting, correct the boundary opportunity
estimate and target the measured upstream tail rather than treating that wait as boundary arithmetic.
