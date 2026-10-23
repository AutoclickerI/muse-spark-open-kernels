# MoE checks for the next 09:00 KST reset

Local-only round, 2026-10-23. **No checks or scored runs were sent in this round, and no
submission worker was started.** The coordinator owns the old queue and promotion;
`finish_queue.py` remains stopped. `submission.py` was not changed by this round.
Use `tools/okc_run.py` after 09:00 KST, detached, with a note. Its local priority queue
owns the global four-run judge limit; do not create another sender/scheduler.

## Order

| Order | Source | Decision and purpose |
|---:|---|---|
| 1 | `problems/muse_moe_w4a16/cand_probe.py` | **Diagnostic check only.** Twelve variants on every load: native residency/staging controls, RS-WGMMA token widths, adaptive scheduling, and 64/128-row weight blocks. Establish which canonical-bank reader wins at M32/128/256 before spending checks on policy variants. |
| 2 | `cand_rs128.py` or `cand_rs.py`, chosen from #1 | First production check. Both are self-contained and locally pass. Start with `cand_rs128.py` if its adaptive variants win. Default public policy is native below M64 and grouped at M64+. If RS also wins at M32, test a broad threshold such as M24/32, rescreen locally, then check that policy. Do not blindly enumerate every N. |
| 3 | `cand_ready9.py`, conditional | V9 plus per-expert **token-block** W13 readiness. Use if the coordinator's V9 result wins and the probe/profile indicates a readiness tail; local gain over V9 is within noise. Otherwise reserve this check. |
| 4 | The other row-block candidate, conditional | Check only if #1 suggests a real crossover that #2 cannot cover, or the production result disagrees with the probe. Combine readers over the **same** V6 bank. |

First reconcile pending V9 `f6028a478d594b31`, v4n1, humming6/6k and the old hybrid receipt
with the coordinator. The dual-bank hybrid is not a serving candidate. If V9 becomes the
checked native best, transplant its small-M call into the selected RS candidate and rerun
the local screen before #2; its packed bank is identical. The current RS files deliberately
retain V6 as the controlled native path until that evidence arrives.

Suggested first invocation **after reset**, through the coordinator's queue:

```bash
python3 tools/okc_run.py check problems/muse_moe_w4a16/cand_probe.py \
  --note "Canonical-bank diagnostic: native residency/staging, RS N8/16/32/64/128, adaptive N, 64/128 weight rows; graph device intervals, not a scored candidate"
```

A passing production check must beat the then-current checked best before promotion/scoring.
Read full MoE latency, not only a GEMM interval. V6's H200 reference is 1.4916x
(`eefecf1eeb724feb`); the supplied top-50 target is 1.711x. If decode-32 stays unchanged,
**both** large verify loads need about 1.51x acceleration over V6 to reach that target.
Improving all three target loads equally needs about 1.315x each. These are geometric-mean
arithmetic, not performance predictions. Humming3's batch-256 win alone is insufficient.

## What the diagnostic measures

Every call computes all variants on its current inputs, returns the final variant's output,
and performs all instrumentation inside the call. It never caches results, tunes by input
values on the host, changes the harness or skips timed work. Exactly one packed bank is
prepared. A 256 MiB non-weight eviction buffer is read before **each** variant, outside its
logged interval but inside the compound timed call, so later variants do not get a warm-bank
advantage. The compound check's overall ratio is deliberately meaningless as an optimization.
Do not put this diagnostic on the scored board.

Log format:

```text
MUSE_SWEEP m=256 call=16 stamp_control_ns=...
MUSE_SWEEP m=256 variant=10 raw_ns=...
```

| Variant | Work |
|---:|---|
| 0 | V6 NT1, two requested stages, occupancy-sized grid capped at two CTAs/SM |
| 1 | Same compiled NT1 body, two stages, one CTA/SM |
| 2 | Same NT1 body, three requested stages, one CTA/SM |
| 3–7 | 64-row RS-WGMMA, fixed token N=8/16/32/64/128, two activation stages |
| 8 | 64-row RS, per-expert adaptive N, two stages |
| 9 | 64-row RS, adaptive N, three stages |
| 10 | 128-row RS (two m64 fragments share each activation stage), adaptive N, two stages |
| 11 | 128-row RS, adaptive N, three stages |

M1/16/32 are decode loads; M8/128/256 are verify loads. For M>512 the diagnostic omits
native variants 0–2. Device `%globaltimer` intervals include a timestamp-kernel cost;
`stamp_control_ns` provides an adjacent-empty-stamp control. Keep raw intervals and compare
multiple samples; do not present these as paired official per-variant speedups.

Logging occurs at power-of-two invocation counts to retain both eager and graph samples
without flooding the receipt. All variants still execute on every call. In the local harness,
three warmups precede capture, so call 8 and 16 are graph replays. Eager intervals include CPU
launch/allocation gaps and should not select a tile. Native stage requests are reduced only
when the GPU's shared-memory limit requires it: local sm120 cannot exercise the three-stage
native Hopper geometry. The compiler's native launch bounds remain fixed for variants 0–2;
this isolates launch residency/staging, not a different register allocation.

## Local results and static Hopper compilation

All ratios below are **sm120**, paired against local SGLang; they are not H200 forecasts.
The GPUs are shared, and small differences between separate runs are not decisive.

| Candidate | Local GM | Verify M128 / M256 latency | Status |
|---|---:|---:|---|
| `cand_rs.py` | 1.014x | 730.9 / 978.9 us | Five standard tests and fresh-input graph timing pass; portable MMA emulation of RS |
| `cand_rs128.py` | 1.039x | 685.5 / 957.9 us | Same checks pass; shares activations across two m64 weight fragments |
| `cand_ready9.py` | 1.264x | 410.9 / 526.0 us | Standard tests/graphs pass; native V9 control was 1.258x in its earlier local run |
| `cand_ready.py` | 1.208x | 420.0 / 581.9 us | V6 plus token-block readiness; held below Ready9 |
| `cand_retire.py` | 1.174x | 445.5 / 607.4 us | Also retires per-token norm/combine before the reset barrier; local regression, held |
| `cand_probe.py` | diagnostic | all twelve variants logged | Final source passes standard tests and fresh-input graph timing; compound ratio is not useful |

The readiness change waits for 512 W13 row tiles of the **current 8-token block**, rather
than every block of the expert. Retire additionally publishes each route's complete W2
rows/norm statistics and combines ready tokens before the global reset. It preserves a
final barrier before reusing counters. The added publication/fence/reset cost failed to
produce a local gain; this is why it is not in the main check order.

Offline SM90a compilation passed for every candidate. All 20 grouped specializations
(N8–128, W13/W2, stages 2/3) have **zero stack and zero spill loads/stores** for each row
geometry. W13 registers for N8/16/32/64/128 are approximately 64/79/72/92/140 for the 64-row
path (two stages), and 70/82/108/126/195 for the 128-row path. Ready9 uses 124 registers,
zero spills. Ready/Retire still compile the dispatched native NT1 at 96 registers with
spills; their unused NT2 instantiations are not the executed path.

Actual SM90a SASS for the 64-row W13 variants contains four HGMMA sites, zero HMMA sites,
LDGSTS activation copies, warpgroup barriers, and no LDL/STL sites. These are static sites,
not dynamically weighted counts. Runtime WGMMA/descriptor/barrier correctness on H200 is
still unverified; the portable path cannot establish it.

Evidence:
- `local_rs.log`, `local_rs128.log`, `local_ready9.log`, `local_ready.log`, `local_retire.log`
- `local_probe_final.log`: final diagnostic source, including graph intervals for all loads
- `compile_round2.log`, `compile_rs128.log`, `compile_probe_ready9.log`, `compile_probe_final.log`
- `grouped_sass_summary.txt`, `grouped_sm90.sass`

## Large-M path and one-bank invariant

`cand_rs.py` and `cand_rs128.py` use V6's **exact existing tile layout**:
`w13p [32,512,36864]`, `w2p [32,256,36864]`. One local bank is 905,969,664 bytes
(864 MiB; 50.625 GiB across 60 layers). Neither state retains original packed matrices,
a Marlin/Humming/Quack repack, a BF16 weight bank, or another converted tile bank.
Global scales and the BF16 norm parameter are tiny sidecars.

The reader combines four canonical tiles for m64; the 128-row path consumes two such
fragments. It reuses the exact rotate/AND decoder and encoded group scales. V6's K
permutation is applied to **activation pairs during the gather into shared memory**, so no
weight conversion/repack is needed for RS-WGMMA. Inputs are BF16; WGMMA accumulates FP32;
FP32 global scales are applied in the epilogue, with BF16 gate/up, SwiGLU and down-output
boundaries. Norm is per expert before routing-weight multiplication and summation.

Routing is histogram + prefix/plan + scatter in separate fixed launches, using int32
counts, positions and pair ids. All live metadata is rebuilt each call. Expert counts
select N8/16/32/64/128 on device. The grouped path has no fully-resident-grid barrier or
inter-expert readiness counters. It stages activations with cp.async; packed weights are
read directly into registers. **TMA weight staging is not implemented.**

Both row geometries passed every fixed N and both stage depths at M65 with wide scales.
One captured graph then replayed hot -> empty -> normal -> hot routing with fresh activations;
empty routing returned exact zeros. Both also computed M753, M3012 and M16384 with one local
expert on every token and fresh graph inputs. Three rows (first/middle/last) per large replay
were compared to the full-FP32 reference: relative L2 about 0.0045–0.0048, two replays each.
This is **sampled large-M reference validation**, not a full 16K reference or a performance
comparison with the prefill-layer winners. It exercises hot N128 tiles and int32 routing;
large-M many-expert coverage remains to be checked on H200.

`exact_decode.log` independently verifies all 16 E2M1 codes times all 254 finite E4M3 scale
encodings (negative, subnormal and zero included), numerically exact BF16 values before
the FP32 global scale. `stress_rs.log` / `stress_rs128.log` contain the tile and large-M checks.
At M16384, worst-case 8-route capacity allocates 2 GiB for intermediate plus down activations,
plus 128 MiB output and small metadata. This is current-call workspace, not resident weights;
actual serving graph-pool/workspace peaks are unmeasured.

Serving integration still needs actual EP-rank weight loading, separate gate/up FP32 global
scales from the real checkpoint (the task supplies one W13 scale), and surrounding collectives.
For `muse_prefill_layer`, pass its normalized expert input/routing into this grouped reader,
using the same bank as decode. Benchmark complete expert work, including gather/SwiGLU/norm;
compare against native prefill v3 and timed expansion + DeepGEMM. No serving or prefill-layer
H200 speedup is claimed.

The layout report recommends Marlin canonical for its stock path; this round follows the
owner's explicit V6-canonical choice. Quack's installed `prepare_w4_weight` builds a different
fragment bank, so the implementation uses the RS primitive directly with the canonical reader.
Research references: `research/moe_decode.md` A2/A5, `research/hw_lowlevel.md` MoE probe plan,
`PROFILE.md` static audit; [PTX WGMMA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-multiply-accumulate-instructions),
[Quack NVFP4 layout](https://github.com/Dao-AILab/quack/blob/77a750c0d9548d4a6e23c0f64062d61b3ea420ff/quack/blockscaled/nvfp4_utils.py).

Further work after measurement: stage canonical K512 weight strips with bulk copies to reduce
direct-load latency/address issue; expand row blocks only if register/residency evidence supports
it. Do not add another resident layout to recover a library path.

Frozen source and embedded-CUDA SHA-256 values are in `round2_manifest.json`. Final verification
matched them to the offline compile logs, checked all six candidates' five standard tests and
six graph-timed loads, both grouped tile/routing/large-M suites, and all twelve diagnostic
variants at graph replay call 16 on every scored load. Rebuilds live in `build_grouped.py`,
`build_rs128.py`, `make_probe.py`, and `make_ready.py`; diagnostic sources are generated files.
