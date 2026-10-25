# Producer epilogues (P7)

Base: `problems/muse_decode_layer/cand_r3_best7_s11_final.py`, SHA256 `acda6c62`.
Nothing in this directory is promoted. `patch.py` leaves `PG_ROLES=()` until a
paired whole-layer win; `--trial --hybrid` admits the three experimental roles.

The structural target is to absorb b1 into O, route into pool_router, and b2 into
unpooler, with <=1 us of added critical-path work each at decode loads. Merely
removing the named glue node is insufficient: compare producer-plus-glue
completion-frontier increments and unprofiled whole-layer paired ratios.

## Implementations

* `kernels.py`: balanced split-K GEMV with monotonic publication and one owner
  per token applying the exact boundary or pool norm/top8. This first prototype
  loses on H200; preserve it as an attribution control, not an integration patch.
* `stats_kernel.py`: last K split reduces each output tile and writes A/B plus
  four sufficient norm moments. Four/eight owners per token wait for all tiles,
  then apply disjoint output chunks. Sliced unpool accounts for all 8192 residual
  columns even though its branch has only 1024. Route sums router hi+lo in S11's
  split order. Optional MoERouteMeta v1 uses a histogram and stable integer sort.
* `fixed_o_epilogue.cuh`, `fixed_o.py`, `build_fixed_o.py`, `fixed_o.cu`:
  preserve S11's native decode-1 O config6 body, its pre-wait weight loads, and
  BF16 branch rounding; append tile statistics and eight apply owners.
  `ProducerHybrid` uses this for decode-1 O and ProducerStats elsewhere.

The boundary identity uses `u=alpha*stream`, `v=beta*postweighted_branch`,
`r=rsqrt(mean(branch^2)+1e-8)`. The next norm's squared sum is
`sum(u^2)+2*r*sum(u*v)+r^2*sum(v^2)`. A/B remain FP32. Clamp its mathematically
nonnegative variance at zero before adding epsilon. Local comparisons below
measure the resulting rounding difference; this is not an exact two-pass sum.

MoERouteMeta emits all 711 int32 ABI words in a 2944-byte padded buffer. The
small-M consumer is available at `muse_decode_layer/dev/nt1_timeline/meta_cell.py`;
this lab emits metadata but does not yet replace S11's expert consumer. Consumer
integration needs its own paired whole-layer evidence.

## Publication and replay

All scratch is allocated in prepare. Each producer CTA keeps its own completed
uint64 generation. Every invocation writes its partial before a GPU-scope
release/acquire arrival. The last split publishes reduced tile statistics with
a second arrival. Apply owners acquire the exact `generation * tile_count`
target, then read published data through `.cg` loads. Each CTA retires its own
generation only after completing its work. Native O uses the same protocol,
with thread fences/barriers before release arrivals and cache-global reads.

There is no launch reset or graph grid-id epoch. Invocations are stream/PDL
ordered: the entry wait completes before reading generation or inputs. Thus a
later invocation cannot overwrite live prior-generation scratch. At most 128
apply owners wait, below H200's 132 SMs, leaving capacity for unfinished
producers even if owners are scheduled early. This follows the split-barrier
and replay-epoch issues described in `docs/reference/research_deepinfer.md`;
it is not an oversubscribed all-CTA barrier.

`check.py --stats` covers T=1,2,8,16,31,32, both early/late logical owners,
three roles, 40 replays per cell, NaN-poisoned scratch, zero inputs, uint32 and
uint64 generation wrap. Metadata checks all 711 fields and none/some/none local
routes. `check_fixed_o.py` covers 88 native O replays with the same poison/zero
and wrap cases. These tests run on sm120; Hopper execution evidence is below.

## Whole-layer lab

`build.py` embeds the frozen base, the timeline lane's verified selected
prepare, paired controls, and CUPTI activity collector. Eight small-M whole
layer fixtures, 12 seeds x 6 rotated cold READ rounds. Arms:
A/A2 original S11, B1/R/B2 individual fusion, F all without metadata, FM all
with metadata, RAW new producer bodies with original glue, BG generic rather
than native decode-1 O. Scratch owners remain alive for graph lifetime.

`INC` compares completion-frontier increments for `[O+b1,pool+route,unpool+b2]`.
`EPI` is a diagnostic final-epilogue-store minus last-partial-store stamp; it is
not a kernel-completion event. Compressed output preserves every A/FM CUPTI node,
stream, name, start/end, MAD, and replay span. `harvest.py JOB` archives this
evidence and the paired results after `tools/h200_lab.py read JOB`.

### H200 receipts

Job `f21c60b0b58c4fd5`, prefill, payload `5b93acdb`: row-owner prototype.
All eight fixtures and paired gates completed. Lower variant/A is faster.

| Load | A:A | b1 | route | b2 | all, no meta | all + meta | raw producers |
|---|---:|---:|---:|---:|---:|---:|---:|
| decode1 | .99615 | 1.06986 | 1.14780 | 1.06483 | 1.21171 | 1.28528 | 1.02470 |
| decode16 | 1.00751 | 1.04885 | 1.10194 | 1.05169 | 1.13415 | 1.22677 | 1.02457 |
| decode32 | 1.00166 | 1.05308 | 1.10364 | 1.02649 | 1.11679 | 1.19701 | 1.01015 |
| verify1 | 1.00559 | 1.11369 | 1.11811 | 1.04128 | 1.17303 | 1.26348 | 1.03160 |

Decode1 sliding increments: original `[4.06,24.22,6.78]` us, fused
`[13.09,34.24,11.97]`; profiled wall 63.30/84.38 us. Its epilogue store tails
are b1 7.936, route 13.792, b2 6.944 us. Raw producer-only regressions are much
smaller than full fusion. This supports addressing serialized row rereads,
metadata, and producer fidelity, rather than promoting kernel-count reduction.

Job `f83f9f40e89d4b47`, prefill, payload `22d25aec`: hybrid tile-statistics
revision: **compile failure, no usable whole-layer measurements**. Adds compact metadata, distributed apply owners, exact native
decode-1 O, and BG attribution. Admission: hybrid payload eight fixtures clean;
normal/forced local_eval 13 tests, 12 graph shapes, 24 static checks each.
Candidate SHA256 `61ae7886`; native CUDA `f65773d5`. Native raw/fused/traced O
all use 104 registers, 2048 shared bytes, zero spills on sm90a. Triton statistics
kernel: 48 sm90 compile cells, no local-memory accesses/spills, <=64 registers.
The H200 wrapper discarded the innermost compilation cause. `build_diag.py`
creates a small synthetic payload retaining the full exception chain;
`check_embedded.py` checks the exact embedded source with the Hopper target.
The embedded-source Hopper check passes four cells locally as well. Diagnostic
job `ebe45a997c6445ae` on MoE, payload `b13968d2`, found the cause: native O+b1
passes, but the judge's Triton rejects a `triton.next_power_of_2` call inside the
JIT epilogue (torch 2.14.1+cu130, Triton 3.8.0, sm90). This is a compiler
compatibility failure, not a performance result.

The fix computes the statistics extent in Python and passes a constexpr; the
metadata extent uses only constexpr integer branches. It changes no arithmetic
or publication protocol. Fixed local admission: eight whole-layer fixtures and
timelines clean, 120 changed-path poison/zero/wrap replays, four embedded-source
sm90 compile cells. `admission.json` separates these checks from the earlier
normal/forced screens. Fixed trial `3395fd7e`; paired payload `294fe739`.
Job `15abb9c1e1c74b0b` on prefill is pending. The lab now retains completed arms
if a different arm fails and records the innermost exception cause.

## Integration

No winning patch yet. Reproduce the experimental candidate with:

```sh
python3 problems/muse_bf16_proj/dev/producer_glue/build_fixed_o.py
python3 problems/muse_bf16_proj/dev/producer_glue/patch.py \
  problems/muse_decode_layer/cand_r3_best7_s11_final.py /tmp/producer_trial.py --trial --hybrid
python3 problems/muse_bf16_proj/dev/producer_glue/build.py
```

Next: use the hybrid H200 paired ratios and combined timeline increments to
choose a concrete bottleneck. A passed correctness gate does not admit a
slower role. If any role wins, freeze only its winning shape/role rows as a patch
onto acda6c62 and retest the resulting whole-layer candidate before handoff.
