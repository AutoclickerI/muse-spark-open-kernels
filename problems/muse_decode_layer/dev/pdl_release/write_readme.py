"""Render the current source audit and evidence as the lane handoff."""
import json
from pathlib import Path
H=Path(__file__).resolve().parent
r=json.loads((H/'trigger_audit.json').read_text());m=json.loads((H/'manifest.json').read_text())
s='''# S15 producer-release audit and S15e

Base `cand_r3_best7_s15.py` SHA256 `60254883` is unchanged. Experimental
[cand_r3_best7_s15e.py](../../cand_r3_best7_s15e.py), SHA256 `084e004b`, keeps every
shape plan, arithmetic expression, weight layout, cache key and precision boundary.
This lane changes programmatic launch/release/wait plumbing only.

The proposed lever has a narrower opportunity here than in the donor: most owned
S15 kernels **already release at entry**. No owned scored kernel explicitly releases
after waiting for its last TMA. Delaying existing entry releases is not an earlier
release. S15e keeps those and closes the missing MoE/VDSK and consumer-enable edges.
No “34 us” benefit is assumed for Muse.

NVIDIA's [PDL guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html)
requires every producer CTA to trigger (or exit), a PDL-enabled consumer launch, and
a dependency wait before dependent reads. The wait observes completed/flushed producer
results; triggering does **not** make outputs ready. Consumer prologues may overlap
when they use only independent inputs. Events still record full completion.

## Current trigger table

Every family in S15's twelve scored shape paths is below, plus portable/off-shape
families exercised by the all-shape screen. “Implicit exit (external)” means S15
requests ordinary library completion; it is not a claim about undocumented internal
FA3/cuBLAS SASS. Exact source locations and caveats are in
[trigger_audit.json](trigger_audit.json).

| Kernel / producer family | S15 trigger | Successor waits before producer-output read? | Successor launch / barrier |
|---|---|---|---|
'''
for v in r['rows']:
 s+='| '+ ' | '.join(v[k].replace('|','/') for k in ('kernel','current','successor_wait','successor_launch'))+' |\n'
s+='''
## Call paths at the six loads

Expand the named families using the trigger table above. Each row below is one of
the12 timed shapes (46 sliding +16 full layers per load); common norm/router/experts
subkernels are retained rather than hidden inside one timing total. Native Lt
heuristic choices depend on prepare, so external algorithms are named by their
measured family rather than an invented fixed binary symbol.

| Load / layer | Ordered kernels / families |
|---|---|
'''
for v in r['shapes']:
 label=('decode' if v['q_len']==1 else 'verify')+' '+str(v['batch'])+' / '+('sliding' if v['window'] else 'full')
 s+='| '+label+' | '+' → '.join(v['stages'])+' |\n'
s+='''
## Changes and independent arms

[build.py](build.py) composes S15e and each grouped leave-one-out arm from the exact
frozen S15 source. The full candidate has no runtime fallback or duplicated old
implementation. Group omission is a build-time experiment.

| Producer / edge | S15e change | Consumer safety |
|---|---|---|
| Small-M `moe_var` flags95/69 | Entry release, guarded by existing PDL flag | Existing router wait retained; unpooler preloads weights independently and waits before X |
| Generic recoded V9 `moe_fused` | Entry release | Input launch stays ordinary; successor owned unpooler waits |
| Large-M `route_live` | Entry release and PDL launch | Wait after independent shared-count initialization, before ids |
| Heavy/light W13 and W2 `gemm` | Entry release and PDL launch | Wait after static address/descriptor setup, before TASK/activation reads |
| Small-expert W13/W2 `phase` | Entry release and PDL launch | Wait after mbarrier/shared initialization, before counts/ACT |
| Large-M `finish` | Entry release and PDL launch | Wait before OFF/IDS/SS/Y reads; parent side-stream joins stay full-completion events |
| VDSK qkv | Entry `gdc_launch_dependents`; launch_pdl=True | X/W are the call's original inputs; VQ8 consumer already waits |
| Portfolio q1 `attn_main` consumer | PDL launch and wait before qkv loads | Metadata/address prologue uses original inputs; producer sumcast already releases at entry |
| Glue | No edit | All selected owned kernels already release at entry and wait before dependent reads |
| cuBLAS/Lt, FA3, torch casts | No binary/library replacement | Ordinary completion boundaries preserved; no invented library wait or release |

Groups: **E**, **no_attention**, **no_moe**, **no_projections**, **no_glue**.
`no_glue` is intentionally source-identical to E and is a second no-op control.
Each source/producer edit and expanded CUDA diff is recorded in [manifest.json](manifest.json).
CUDA edits:27 added /8 removed lines across four source strings; Python VDSK/extension-name
edits are separate. Source serialization makes a raw whole-file diff misleading.

## Verification

`screen.py` runs tools/local_eval in the judge image on **GPU0 only**, reusing the
65-case nearby audit set and seeds419731/427650 plus all13 official TASK cases.
Each fixture is checked eagerly and after two graph replays by the real TASK
collect/check_implementation, including routing membership, all keys, new KV cache
writes, FP32 reference and query-derived tolerances. No timing inference from sm120.
Normal and forced modes exercise portable launches and the existing explicit
Hopper-only launch stand-ins. These cannot prove native Hopper concurrency.

The standalone whole-file static screen compiles every CUDA literal for sm90a;
[native.py](native.py) additionally compiles the four changed sources with -O3 and
records the release/wait instructions in Hopper SASS. Native concurrency and its
performance still require the paired H200 receipt.

## Batched paired H200 experiment

One decode-host payload: **A=S15, A2=A:A, E, no_attention, no_moe, no_projections,
no_glue**;12 shapes /6 weighted loads;8 fresh seeds;8 rotated/swapped cold graph
replays per arm and seed.1GiB **read** flush before every timed replay and measured
empty-graph subtraction. r=arm/S15; smaller is faster. Per-load ratio mean, standard
error and seed range are retained along with absolute per-shape times.

Every arm uses the exact same original prepared S15 plans, weights and lossless
codes. Changed modules are compiled under content hashes and rebound explicitly:
E never times S15's cached MoE/attention binary by accident. Identical Triton kernels
share only a matching full JIT dependency closure; VDSK is compiled separately.
The full-layer gates compare output L2 and route membership on every fresh seed;
local TASK checks are independent. No tolerance is weakened for a failing arm.

The payload honors prepare's remaining deadline and reports missing cases/seeds as
incomplete. No native gain is claimed before its receipt. Priority3 on decode,
not-before1792972800 (2026-10-26 00:00 UTC/09:00 KST), behind priority-1 integration
and priority0 P5. Identity and results are added below after admission.
'''
p=H/'validation.json'
if p.exists():
 d=json.loads(p.read_text());s+='\nCurrent admission: '+json.dumps(d,separators=(',',':'))+'\n'
p=H/'queue.json'
if p.exists():
 d=json.loads(p.read_text());s+=f"\nQueued sender **{d['sender_id']}**, wrapper **`{d['wrapper_sha256'][:16]}`**. Not-before **{d['not_before_epoch']}**, priority **{d['priority']}**; no judge ID yet. See [queue.json](queue.json).\n"
(H/'README.md').write_text(s)
