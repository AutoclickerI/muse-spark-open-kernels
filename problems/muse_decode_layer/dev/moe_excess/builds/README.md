# P3b: prebuilt outcomes for the morning S19 ladder

Base: newest P4 build at construction, **S22 `4b8c803d`**, shipment `d1ccdc35`.
[base.json](base.json) pins its committed source so P4's morning selection cannot
silently rebase these already screened arms.
The diagnostic remains exact S19; S22 has the same large-M MoE body. No H200
performance result exists for this portfolio yet. Public full-port recipes remain
closed. These arms replace only the heavy expert branch inside S22's original
production routing/three-stream MoE graph, plus EPI's finalizer choice.

| Arm | Diagnostic outcome targeted | Concrete change |
|---|---|---|
| REG | L0 vs unique STREAM: operand issue/reuse or occupancy dominates | Original packed Q/scale words load directly into registers. Only BF16 activations use the shared pipeline; one warpgroup, 64-row tiles. Shared reservation 100 KiB →64 KiB. No repacking or widened weight values. |
| DECODE | L1−L0: decode dependencies/register lifetimes dominate | One decoded16-register fragment instead of two; complete WGMMA's asynchronous register read before reuse. Exact bit decode and scale arithmetic stay intact. Trades overlap for register lifetime; native99/101regs instead of128. |
| PAD | L2−L1 / MATH: MMA work or padded token columns dominate | Heavy tails choose exact16/32/48/64-row WGMMA tiles instead of32/64. Same K reduction, output and weight precision; no token or expert skipped. |
| OCC | L0−STREAM: insufficient residency dominates | One128-thread warpgroup, 32-row tiles and three stages. Dynamic shared38,400B and88/89regs; targets residency at the cost of more repeated weight reads for larger experts. |
| BALANCE | CTA/SM histograms: wave quantization or expert-size tail dominates | Flatten `(expert,row64,channel)` into uniform grid-stride work. Device-side prefix uses actual counts and ORDER; no host reads, atomics, extra route launch or offline schedule. |
| EPI | L3−L2 or readiness dominates | Hoist global/norm scale loads and use S22's existing warp-only PDL finalizer14, removing the finalizer's CTA exchange/barrier. All FP32 Y and square sums are retained. |

These are hypotheses, not a promise that the largest conditional contrast is
independently removable. In particular, DECODE changes register dependencies and
completion, not NVFP4 arithmetic; OCC may lose to extra HBM demand, and BALANCE
addresses the heavy-branch tail rather than an unobserved synchronization cost.
Native instructions/resources and production whole-load time decide them.

## Exact artifacts and checks

[candidates.py](candidates.py) generates all six source patches on the pinned S22
and a common native extension. `REG.py`, `DECODE.py`, `PAD.py`, `OCC.py`,
`BALANCE.py`, `EPI.py` are exact single-file production candidates, preserving
S22's embedded MIT, Apache and NOTICE texts. [candidates.json](candidates.json)
and [P4_HANDOFF.json](P4_HANDOFF.json) bind expanded/packed source identities and
the four changed scored cells. The generator's asserted replacements are the
reviewable patch; generated expanded sources and CUDA/SASS are below ignored
`out/`. P4 receives this portfolio for later selection, with **no winner named**.

Packed files trim trailing spaces on outer MIT notice comment lines.
`packed_notice_equivalence.json` binds the pre-normalization packed hashes in
the queued manifest to the current files; their encoded executable source is
byte-identical. Native/source identities and every timing/check remain unchanged.

Local component screen: all six full MoE bodies pass eager and graph TASK at
M33/128/256/257, plus M128/256 fixtures containing counts17/31/33/48/49/65/97/M
so every padded width and multi-row-block case executes. **72 TASK comparisons**;
metadata, ACT, normalized FP32 Y,64square-sum slots and finalize ABI retained.

Native sm90a compilation and [audit.py](audit.py) inspect22 actual heavy kernels
(both phases and depth variants). Every arm emits HGMMA and BF16 products;
PAD emits the48-row instruction. REG depth4 has144/140regs, no spills; DECODE
99/101, PAD100/104, OCC88/89, BALANCE128/128, EPI128/128. All new native depth4
arms have zero spills except EPI W2, which retains the original4/4 store/load
bytes. Actual Hopper occupancy remains pending. [native_validation.json](native_validation.json)
contains source/SASS identities and static counts; no local timing is used to
pick a candidate.

Full production screens use S22's explicit normal/forced Hopper stand-ins.
The exact whole-graph payload is rehearsed with64fresh seeds×16symmetric rounds,
**3,840 TASK plus1,040 nearby checks**, every candidate on every changed scored
cell and all nearby checks. The forced run asserts new heavy launches in eager
and captured Python glue for every arm/phase. The unchanged P4 independent TASK
checker supplies all gates. Validation JSONs record exact wrapper identities,
wire size, counts and the stand-in limitation.
[wrapper_equivalence.json](wrapper_equivalence.json) binds the rehearsed wrapper
7275744d to queuedaa449a80: the CLI changes only the two budget literals850 to
850.0. Exact payload/candidate/check/sampler bytes are identical; the numerical
deadline is identical, and the queued wrapper passes transport smoke.

The first nonblocking normal rehearsal faulted asynchronously with CUDA error700;
its originating launch was not recovered. A fresh launch-blocking normal run
passed all4,880checks. This remains an unassigned local event, not a diagnosed
kernel defect or a claimed repair; the final nonblocking/forced validations and
`rehearsal_issues.json` retain its status. Native faults invalidate their arms.

## One production cohort after the diagnostic

`build_lab.py` uses P4 commit `ec754e5`'s sampler and source/protocol admission:
64fixed fresh seeds,16palindromic rotating rounds, arithmetic mean of every round,
no selective retiming. Input/router/weight seeds are disjoint from the parent
diagnostic and prior P4 selection/confirmation seeds. A/A2/INC are independent
S22 graphs; each candidate runs on v16S/F and v32S/F, inside full `kernel_fn`.
Other scored rows have identical code and borrow the independent A2 measurement,
as P4's restricted-arm protocol does. Each load is46sliding+16full, followed by
the six-load geomean. Per-kernel wins cannot nominate a whole-graph winner.

The protocol keeps all-round arithmetic means and counts for every seed/arm,
and the worst four complete primary round/enqueue traces with telemetry around
each symmetric block, exactly as `ec754e5`. It does **not** archive every raw round
for every candidate; the parent ladder archives all of its rounds. No trace
selection alters score samples, controls or seed counts. The public transport
keeps a checksummed compressed archive within the16,000-character tail, with
4,000trailing characters tested. Every dropped/incomplete/device-fault arm stays
unmeasured by `tools/promotion.py`.

[queue.json](queue.json) identifies production sender **384352**, wrapper
`aa449a800d62484a`, payload `fbbc47741b749b89`, same host and PLAN rank6 as
diagnostic `bad55569`/sender3645567, queued later so it follows the diagnostic.
Both wait for the10-27 09:01KST reset. No candidate is scored or promoted by this send. `read.py`
uses `promotion.evaluate`'s native/source/protocol/TASK health and reports
**S22/candidate whole-load R** with paired95% CIs, plus A:A. Selection from this
portfolio requires new disjoint confirmation before a G0 winner is handed to P4.

`follow.py` is a finite local event worker, identified in `follow_queue.json`.
Worker **408337** replaces our older duplicate diagnostic reader. It waits for
the exact diagnostic receipt, then drops an arm only if its
predeclared hypothesis is refuted with matched control/resource evidence. A
positive term is retained even if another term is larger. REG/OCC/DECODE need a
nonpositive paired contrast at **both** M, matching residency/no local traffic,
and allocation/clone CIs covering1. PAD drops only if the observed expert counts
never use a changed16/48-row tail. Finalizer and scheduling hypotheses stay when
the diagnostic cannot refute them. Failed controls do not justify a drop.

If the production snapshot is still local, the worker stops only its own waiter
under `runs/.dispatch.lock`, rebuilds the subset, screens packaging/transport,
and replaces that waiter via `tools/h200_lab.py send --lane P3b`. This consumes
one admitted production run. If already admitted, it preserves the original
cohort and records that pruning was too late. After that receipt it writes the
whole-load report and exits. No polling, model invocation, scoring, promotion or
shared-doc edits occur in this worker.

Rebuild and screen:

```sh
python problems/muse_decode_layer/dev/moe_excess/builds/candidates.py
python problems/muse_decode_layer/dev/moe_excess/builds/build_lab.py
CUDA_VISIBLE_DEVICES=0 python problems/muse_decode_layer/dev/moe_excess/builds/screen_components.py
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_decode_layer/dev/moe_excess/builds/screen_lab.py
CUDA_VISIBLE_DEVICES=1 tools/img python problems/muse_decode_layer/dev/moe_excess/builds/screen_lab.py --forced
```

Pending: native diagnostic terms/CIs, native production transfer, and any fresh
confirmation. P3b stays active; nothing here closes the large-M ceiling.
