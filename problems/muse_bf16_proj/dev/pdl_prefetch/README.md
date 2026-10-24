# Dense weight prologues under PDL

This experiment serves G1: improve best7's decode layer at M1/16/32 and verify-1
(M8), sliding and full. H200 whole-layer receipts decide the projection cells.
The projection board's rank is evidence of headroom, not this experiment's goal.

`build.py` freezes `cand_r3_best7_paired.py` and the current dense dispatch from
best6's `1b80ffc037cc454c` receipt. It uses best7's paired glue geometries,
attention choices, NT1 experts and raw scale bank. It does not rerun its tuners.
Each baseline trace is checked against best7's own `_best7_run` on three fresh
public streams. The host prepare/state are left intact.

The payload completes any host's weights at the layer's real shapes and dtypes.
Each missing tensor has a name-derived seed rooted at **72140**, so host coverage
does not change other synthetic tensors. MoE supplies its first 32 real packed
NVFP4 experts, FP8 block scales, FP32 global scales and `post_norm_w` (aliased to
`post_expert_norm`). Dense BF16 weights, FP32 router/bias and the other norms and
gates are synthetic on that host. Projection hosts reuse dense weights and
synthesize packed experts; NVFP4 is never expanded to a prepared higher-precision
bank. Attention input, FP32 residual stream, FP8 KV and metadata always use the
embedded public generator at context 80,000 and stream seeds
172903/180822/188741. The protected second output line names synthetic and host
tensors.

All arms use identical inputs and baseline-derived route IDs, weights and local
IDs per refreshed stream. The routing/norm kernel remains timed and computes
each arm's normalized expert input. The first synthetic screen showed why the
route snapshot is necessary: projection rounding can change a near-tied top-8
and therefore the expert workload. This fixed-route diagnostic measures
transport at equal work; live-route candidate agreement remains a separate
promotion gate. Raw/router 1e-4, BF16 1%, whole-layer 0.3% and exact used-route ID
gates remain in force. The baseline also agrees with the unmodified live-route
`_best7_run` on every stream.

`local_exclusions.json` records a second local rejection: in verify-1 sliding,
QKV BF16 tie rounding is amplified by FP8 attention beyond the unchanged 0.3%
layer gate (0.372-0.379%). Compound A0/A4/R0/P0/P2 arms therefore keep best7's
QKV in that case, explicitly marked `c6 Q=B` in the receipt. The other roles and
all sixteen standalone rows retain their full kernel comparisons. This is a
screened experiment portfolio, not evidence for promoting those QKV cells.

`kernel.cu` contains three actual weight transports:

- `pp_hint_gemv` in `runtime.py.inc`: bulk L2 hints before the dependency wait,
  followed by the incumbent split-K tensor-core arithmetic. H1/H4 hint the first
  one/four BK tiles only in the existing custom producers; A0/A4 replace all
  producers with the same arithmetic without/with bulk hints.
- R: 1D `cp.async.bulk` copies original BF16 weight rows to a shared-memory ring
  before `griddepcontrol.wait`. Activation copies start after the wait. Three or
  four slots hold future weight tiles; mbarrier phases protect slot reuse.
- P: the same ring, using one bulk copy per tile. Prepare losslessly tiles the
  BF16 bank and adds shared-row padding (12.5% for BK256, 25% for BK128). Every
  original BF16 value is retained. No activation or expert expansion is prepared.
- W: 512-thread CTAs, one row per warp and four/eight independent 128-bit weight
  vectors per thread. The first group loads before the dependency wait. Split-K
  yields roughly 512-576 CTAs for the major M1 shapes. The standalone 528-CTA
  read control has 512 threads and four independent vectors per thread.

Early and late native variants have the same arithmetic, layout and geometry.
Their only difference is whether the initial weight requests precede the wait.
All native paths execute directly on local sm120, including the bulk copies;
there is no reduced shared-memory configuration. Every path is also compiled
for sm90a. `audit.py` checks SASS weight requests before ACQBULK in every early
kernel and after it in every late kernel, plus the bulk-hint Triton variants.
[NVIDIA's PTX specification](https://docs.nvidia.com/cuda/archive/13.0.0/parallel-thread-execution/index.html)
defines the nonblocking bulk prefetch and the dependency wait semantics.

The native output ABI is `[split, token, row]` FP32. When the incumbent producer
returns BF16, a timed PDL reduction/cast preserves that BF16 boundary. An earlier
screen found ~0.35% layer drift from omitting this cast; that version was not
sent. At M32 the ring uses KS512, matching the incumbent split length: KS1024
could move a near-tied routing decision on the full-attention fixture. Gates
remain unchanged: raw FP32 and FP32 router 1e-4, BF16 1%, whole-layer A/B 0.3%,
five times these bounds per row, finite outputs and exact routing IDs.

The lab captures eight whole-layer cases. Six coarse AB/BA pairs compare all
eight variants and matching early/late controls. The finalist uses 24 pairs
over three refreshed streams; single-stage whole-layer ablations compare the
overall finalist and the best native family. A separate baseline graph supplies
an A/A control. Sixteen standalone role/shape rows compare the current dense
producer, the new kernels and the 528-CTA read, using the same original banks.
The 512-thread GEMV and packed rings both run at every M=1/8/16/32 role.
Standalone producer timings include the incumbent BF16 conversion boundary;
every variant uses the same three activation banks.
Every replay follows a 128MiB L2 flush outside the events. Times are raw graph
microseconds. Read rates can be calculated from unique weight bytes; they are
effective rates, not measured DRAM counters. P includes padding traffic, and
the M8 tensor tile can reread weights at larger token counts.

The compact receipt keeps every case within a two-payload lab's 4,500-character
share. `L` is the whole layer, `AB` the isolated role changed in the whole layer,
`S` standalone; Q/O/P/U mean QKV/output/pool+router/unpooler. A slash after a
native timing ratio means early/late. `ct` and `kB` describe the fastest standalone
variant's grid and nominal initial weight bytes per CTA, not outstanding bytes
measured on an SM. Full local samples and rejected variants are preserved in
ignored `rows_local.json`; H200 claims require the receipt.

Build and screen:

```bash
python problems/muse_bf16_proj/dev/pdl_prefetch/build.py
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_bf16_proj/dev/pdl_prefetch/validate.py
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_bf16_proj/dev/pdl_prefetch/audit.py
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_bf16_proj/dev/pdl_prefetch/check_weights.py
python tools/h200_lab.py build prof/lab/dense_pdl_prefetch.py --host muse_moe_w4a16 -o problems/muse_moe_w4a16/lab/lab_dense_pdl_prefetch.py
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_bf16_proj/dev/pdl_prefetch/screen.py --share-s 80
python tools/h200_lab.py send prof/lab/dense_pdl_prefetch.py --host muse_moe_w4a16 --priority 0.5 --note "P7 best7 fixed-route synthetic dense PDL weight prologues and many-CTA GEMV on real host experts"
```

`projection.py` is the standalone drop-in: construct `PPDense` outside capture;
call it within the layer and retain the incumbent BF16 boundary. It imports no
repository module. It is not enabled in best7 while the H200 evidence is absent.
`manifest.json`, `native_validation.json`, `compile_sm90a.json` and
`validation.json` bind the source and screens to hashes. The send record will
name the detached log and the immutable queued source. Read the landed receipt
once on resume with `tools/h200_lab.py read <job>`; there is no judge wait loop.
