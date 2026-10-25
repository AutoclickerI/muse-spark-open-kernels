# P11: small-M layer kernel decision

**Do not build the layer kernel or send the first increment yet.** The explicit
fusion estimate is **+1.74% on the six-load score**, below the owner's +3% gate.
Even deleting the estimated synchronization/resource tax gives +2.22%. The
one-read ceiling is much larger (+19.0% overall), but it also deletes intrinsic
attention/dequant/mainloop work; that is not a fusion estimate. This conclusion
is conditional on the present measured inner loops. A competitive bulk-ring
receipt or new H200 boundary/NT1 phase evidence can change it.

The arithmetic, source hashes and assumptions live in [scope.json](scope.json).
[accounting.md](accounting.md) contains every visible kernel row for decode1
and verify1, sliding and full. The source is **c0e74bedf7b74be5**; S6 is still
unmeasured here. `current.json` is the reset profile plus the measured scale-bank
and M256 deltas, not a landed S6 profile. No S6 latency is called measured.

## What the profile establishes

| Load/layer | Calls | Kernel sum us | 3.6TB/s one-read us | Stage excess us | Current modeled adjusted us | Critical-path ceiling us |
|---|---:|---:|---:|---:|---:|---:|
| decode1/sliding | 10 | 105.00 | 58.26 | 46.74 | 105.18 | 46.92 |
| decode1/full | 10 | 120.00 | 63.89 | 56.11 | 115.22 | 51.33 |
| verify1/sliding | 15 | 160.90 | 97.58 | 63.32 | 149.99 | 52.41 |
| verify1/full | 15 | 181.80 | 103.21 | 78.59 | 172.58 | 69.37 |

The official rounded decode1/sliding profile total is104.9us; its printed rows
sum to105.0us. Raw graph minus row sum is +3.27/-2.85/+6.12/+1.94us in the
table's order. Kernel durations include PDL waits and can overlap; these
residuals are not four independent measurements of launch cost. The adjusted
graph and measured per-load deltas, rather than a sum of profiler rows, drive
the score calculation. There are10/15 calls, not uniformly12.

Bytes are one useful read of each original bank and logical KV work. MoE bytes
are inferred from the official floor's active-expert/route model, not actual
transaction counters. Glue rows use their logical IO; library partial-reduction
traffic has only a lower bound because the profile omits its split count. A
zero size is never substituted for missing metadata traffic. Repeated names
merge calls: M1 `_gemv_kernel` combines pool/router and unpool; M8's two large
nvjet calls combine QKV and unpool. There is no defensible measured split of
those individual durations in this receipt. The grouped rows are retained.

Do not mix these kernel spans with95ab's21.1us raw QKV graph time. The latter
includes a different graph envelope/control and does not expose cuBLAS's inner
loop. Subtracting a guessed fixed number would manufacture a phase measurement.

## Why87dc lost

The accepted receipt **87dc944bfd7944c7** measures this exact recipe:
O/boundary/pool =57.67/62.68/65.29/71.23/198.87/365.55us at M1/8/16/32/128/256,
versus best7's38.29/39.17/39.39/39.76/43.56/56.30us. Its source is best7_reset
SHA `b4cc76dc5302c4572951cf56163cb1bd1c8f1db5d3c133c667fbbe208ba6c98a`.
The exact device source hash is in `prior_sm90a.json` and `prior_local.json`.

1. `_dense_chain_call` hard-codes `nt=4` for every M. The device computes a
   32-token tile, including duplicated padded activations, at M1 and M8. NT1
   would use8 tokens. This multiplies activation loads, MMA work and accumulator
   state by4; it does not multiply the immutable weight bytes by4.
2. The grid has132 CTAs,256 threads each:8 active warps per SM. O has512
   row16 tasks (four serial trips for most CTAs), pool has288 (three trips for
   24 CTAs, two for the others). The final pool trip occupies only18.2% of SMs.
   Every tile restarts/drains the register pipeline and performs shared partial
   reduction/barriers. There is no shared stage ring spanning task boundaries.
3. Three grid barriers wait for every CTA: after O, after the boundary and after
   pool. At M1, the boundary has only `M*(16/NT)=4` working CTAs;128 wait. These
   are real serial norm dependencies, though waiting CTAs could preload work.
4. The16MiB prefetch targets L2, not a retained shared-memory operand tile. It
   happens before and between stages, adds traffic and still requires later
   loads. It is not Hazy's next-instruction operand pipeline.
5. At M128/256, token groups reread both dense banks4/8 times. Library GEMMs
   exploit reuse across a wider M. This explains the loss growing with M; a
   constant grid-barrier penalty alone cannot explain that scaling.

`analyze_prior.py` compares the unchanged body with NT1/NT4 and prefetch0/16MiB
on local sm120, grid132, cold256MiB. It checks the boundary and pool outputs and
averages12 phase observations after the existing grid barriers. Those marks
include inner work plus barrier waiting; they cannot isolate barrier latency.
The local results and Hopper compiler resources below are supporting evidence,
not a new H200 timing decomposition. The existing H200 receipt has no barrier
stamps, so an exact H200 barrier-versus-inner-loop percentage remains unmeasured.

## Prior-art mechanism and the architectural difference

[Hazy](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles) uses a
host-built instruction stream per SM, dependency counters, and shared-memory
pages that let the next instruction fetch operands while the previous one
finishes. Intermediate MLP chunks release their own consumers. Its own
breakdown still assigns substantial time to activation consistency, atomics
and warp synchronization; eliminating launch boundaries does not eliminate
those dependencies.

[Cohere](https://cohere.com/blog/megakernels) combines a common warp-specialized
task ABI with static ordering and dynamic attention/expert queues. Its published
BF16 H100 gains are1.58x at batch1, with parallel attention and MoE branches that
can backfill each other's tails. It uses fixed shared layouts, keeps stages
alive across same-op tiles, and fetches weights before activation waits. Paging
was rejected for overhead. Muse's MoE depends on the attention residual, so
that large parallel branch is absent. NVFP4 decode is also additional work.

[MPK](https://arxiv.org/html/2512.22219v2) represents SM-level tasks and readiness
events, uses decentralized worker queues and hybrid early/ready dispatch, and
preloads tasks into released shared pages. Its1.0–1.7x result is a range across
models/hardware, not an expected Muse gain. Reserving scheduler SMs and
integrating host iteration management are not assumed beneficial for this
single-layer graph workload.

## Proposed one-kernel layer, M<=8

Use one resident256-thread CTA per SM, a shape-only task list in prepare, and a
small input-dependent expert queue created by the router inside the call. Keep
the original8-warp NT1 consumer arithmetic; thread0 issues bulk loads. Do not
silently force our256-thread NT1 into Cohere's384-thread ABI. The first design
has fixed scratch layouts per opcode, two reusable weight rings, and no general
compiler or dedicated scheduler SMs. It must preserve the existing allowed
precision, BF16 rounding points, FP8 KV and all keys.

The real dependency chain is:

```text
QKV tiles -> QK prep/RoPE/KV write per KV group -> attention slices/combine
  -> O partials -> round BF16 -> branch RMS -> residual -> hidden RMS/round BF16
  -> pool+router partials
       -> pooled BF16 + expert-input RMS
       -> FP32 hi+lo router + sigmoid/top8
  -> expert/token readiness -> NT1 W13 + BF16 SiLU product
  -> per-expert W2 -> weighted expert RMS/combine
  -> unpooler slice -> BF16 -> final boundary -> next attention input/stream
```

QKV/attention/O tasks can release per KV group. M8 tokens may progress
independently where doing so preserves weight reuse. W13/W2 readiness is per
expert and token block. The two8192-wide residual norms and4096-wide pooled
norm are genuine reduction joins; replacing a grid barrier with a counter does
not make a pool K-slice's normalized activations available before those joins.
All256 router scores participate in top8. Expert weight addresses become known
only after selection; speculative reads cannot be credited as free bandwidth.

Every task has an opcode, output tile, K slice, input event+target and output
event. A producer publishes every output store before its leader performs a
release increment; a consumer acquires that count before loading activations.
Async-copy completion and ordinary global-store visibility are separate
invariants. No CTA may spin behind a task whose producer is later on that same
CTA's static list. Runtime tasks enter an eligible queue only after their
metadata and input events are published. Empty/remote experts publish no work;
terminal counts account for the actual, not maximum, task count.

Use per-invocation epochs. Each CTA retains its own completed epoch in scratch;
CTA0 initializes the next counter bank and releases its init epoch. Other CTAs
can fetch immutable weights while waiting for initialization. Ordinary graph
dependency orders calls, and the kernel cannot finish until all task stores
and terminal bookkeeping finish. Test scratch poisoning and repeated
none/some/none local routing; a prior invocation's count may never satisfy a
new wait. Allocate scratch once in prepare, never cache input-derived results.

The inner-loop sources are the best measured GEMV choice per role, admitted
q1 attention/FA3 q8 arithmetic, one-LOP3 NT1 decoder/MMA, and the correct
boundary/route arithmetic. cuBLAS and FA3 host entry points cannot run inside
another kernel. Their device loops must be ported and separately match the
incumbent before fusion gets any performance credit. The O6 cell is directly
portable; the pending bulk-ring cell is unmeasured and gets no assumed gain.

M16/32 extends the task graph structurally, but is outside this estimate and
dispatch scope. It changes token tiling, expert packing/reuse and attention
parallelism; no T<=8 unpool shortcut is allowed there. An eight-token internal
chunk does not change the call's total T. Do not claim gains on the other four
loads without separate measurements.

## Op estimates and the score gate

The following are **design assumptions**, not H200 observations of the proposed
kernel. They retain the existing loop times and give fusion credit only to
glue/joins and boundary transitions. Individual Q/O/U times marked~ are the
lever model's resource-proportional allocation of merged names.

| Op, sliding | M1 current -> proposed us | M8 current -> proposed us | Rationale |
|---|---:|---:|---|
| QKV incl library reduction | ~15.09 ->15.09 | ~17.22 ->17.22 | Port parity assumed; no copy-floor gain |
| Attention incl prep/combine/gate | 9.90 ->9.90 | 20.30 ->18.30 | M1 already fused; M8 saves2 in glue, not core attention |
| O | ~6.71 ->6.71 | ~7.65 ->7.65 | Port parity assumed |
| Boundary1 | ~3.75 ->2.25 | ~3.40 ->2.15 | ~40% saving; keeps two norms/rounding |
| Pool+router | ~23.85 ->23.85 | 22.70 ->22.70 | Port parity assumed |
| Route+expert norm | 4.50 ->3.00 | 4.20 ->2.70 | 1.5us saving; top8 and full norm remain |
| NT1 W13/W2/combine | 34.80 ->34.80 | 70.29 ->70.29 | Preserve dequant and MMA work; no speculative speedup |
| Unpooler | ~2.65 ->2.65 | ~3.83 ->3.83 | Port parity assumed |
| Boundary2 | ~3.75 ->2.25 | ~3.40 ->2.15 | Same reduction work as boundary1 |
| Tail/dispatch saving across layer | -3.00 | -3.00 | Budget3us, not Hazy's5us times every boundary |
| Counter/activation/resource tax | +1.50 | +2.00 | Explicit uncertain cost, not zero |
| Net saving | **6.00** | **7.00** | Same deltas at full attention; its core work remains |

Apply these deltas to adjusted whole-layer times, preserving each profile's
overlap/replay residual. With46 sliding+16 full layers per load and the other
four loads unchanged, the score factor is
`product_load[Tload/(Tload-46*dS-16*dF)]**(1/6)`.
It gives1.01736x, or1.02218x with zero tax. A uniform10.796us saving in all four
small-M cases is required for1.03x. The point estimate misses by4.8/3.8us per
decode/verify layer. A copy-bandwidth model would claim much more by deleting
real inner-loop work, which the proposed scheduler has not shown it can do.

The riskiest piece is **preserving standalone inner-loop throughput at one
resident CTA/SM while maintaining deadlock-free counter accounting through
the two global norms and dynamic NVFP4 W13/W2 work**. Port parity itself is
optimistic:95ab lost15/16 register-MMA comparisons. Resource-sharing regression
could erase the entire estimated gain. This supports the no-build decision.

## First increment, if the estimate is reopened

One persistent `O -> boundary -> pool+router -> route -> NT1 W13` prefix, M1/8,
against frozen best7_scale on identical inputs in one prefill-host lab (that
host provides dense weights, norms and packed experts). Project-host execution
would require explicit fixed-seed packed expert synthesis. No send before the
owner's09:00KST gate. This is a design, not a queued lab.

Use the same W13 tile routine in both arms. Extracting a standalone W13 prefix
from NT1 changes its launch envelope; never compare persistent W13 alone with
the incumbent's whole W13+W2 call or subtract a guessed W2 time. Return BF16
intermediates indexed by `(expert, sorted pair, column)` plus route metadata.
For correctness, compose the existing W2/unpool/final-boundary suffix outside
the prefix timing and run TASK.md's complete route/cache/update checks against
the reference with the **candidate's chosen experts**. Exact IDs and0.003 are
logged diagnostics, never cell-admission gates. Keep timing cells that pass
TASK.md despite failing either diagnostic, and keep other cells running after
a numerical rejection. Native launch/synchronization failures remain fatal.

Batch these mechanism arms: best7 prefix; exact87dc control; NT1-token repair;
new counter schedule with prefetch off/on; same scheduler with phase-wide
barriers; W13 standalone inner-loop parity; one/two resident CTA geometries.
Stamp task wait, operand fetch, compute, store/publication and worker tail in
separate diagnostic launches. Performance graphs contain no stamps. Use cold
256MiB AB/BA and A:A, at least12 fresh seeds spanning soft, outlier, none-local,
skewed and changed routing. Report raw pairs, per-seed/load effects, TASK gates
and strict-diagnostic failures last. Promote only from a subsequent identical-
input six-load whole-layer comparison with correctness margin across seeds.

There is **no first-increment build, new lab send or projection handoff** from
this scope. The existing bulk-ring send remains held under `ef2048bf49b3fdf4`.
Reopen when port-parity/transport or join evidence supplies the missing gain;
do not wait on that receipt in this turn.

```bash
python problems/muse_decode_layer/dev/megakernel/scope.py
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_decode_layer/dev/megakernel/analyze_prior.py
tools/img python problems/muse_decode_layer/dev/megakernel/audit_prior.py
# After S6 lands, refresh current.json using dev/levers/build.py, then pass its
# saved profile to scope.py --profile. Read a landed receipt once; never poll.
```
