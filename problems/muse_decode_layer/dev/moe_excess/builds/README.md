# P3b: S22 outcome builds for the S19 ladder

**Measured `054e28849b424be2`: every arm loses whole-load six-load R vs S22 `4b8c803d`** (64 fresh seeds
x 16 rounds, 3840 TASK + 1040 nearby pass, A:A 1.0001 [0.9997,1.0004]): REG 0.9555 [0.9543,0.9567],
OCC 0.9848 [0.9836,0.9861], PAD 0.9975 [0.9968,0.9982], BALANCE 0.9964 [0.9958,0.9969], EPI 0.9979
[0.9974,0.9983], DECODE 0.9998 [0.9993,1.0004]. Per load in `result/summary.json`; per shape in
`result/raw.json`. No arm goes to P4. [base.json](base.json) pins the committed S22 source.

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

## Production cohort

`build_lab.py` uses P4 `ec754e5`'s sampler: 64 fixed fresh seeds, 16 palindromic rounds, arithmetic
mean of every round. A/A2/INC are independent S22 graphs; each candidate runs on v16S/F and v32S/F
inside full `kernel_fn`; other rows borrow A2. `read.py <receipt>` reports S22/candidate whole-load R
with paired 95% CIs through `promotion.evaluate` and matches the receipt by its `.py` source.
