# S22 production-graph outcome portfolio

Frozen P4 base: expanded **4b8c803d**, shipment **d1ccdc35**. Every arm is
an exact append-only patch to that expanded source; each patch carries the
MIT/Apache notices in the reviewed public kernels it embeds. S22 already
restores S19 at d16S and consumes direct partials at d32S/v1S. The diagnostic
still uses its original S21+9b control; its loss at d16S cannot justify
reintroducing that port into S22.

| Arm / source SHA | Diagnostic outcome targeted | Built change / cells |
|---|---|---|
| CHAIN / 92367a26 | Lost overlap or ordinary prep/main launches dominate | PDL entry release; consumers wait after independent CTA setup, history producer proceeds without QKV. Six remaining public cells: d32S,v1S/F,v16S/F,v32S. |
| INC (DIRECT) / 9eedddf9 | Scalar duplicate projection-partial reads or prep launch dominate | Ordered FP32 planes through aligned 16-byte loads; once-per-CTA query/own-KV sharing, guarded at request boundaries; direct BF16 consumer rounding. PDL composed. Six public cells plus actual d1S last-CTA and d1F prep readers. |
| STAGE / 5d3d6f89 | Query/own-KV rereads, live preload registers or consumer prologue dominate | Cooperatively reduce the first 1280 BF16 Q/K/V columns once into 20 KB of dead V-widening shared storage; consume staged vectors, release storage before PV. No extra allocation or launch. v1F,v16S/F,v32S. |
| LANE1 / d0c044f1 | Gather latency, insufficient resident CTAs or split waves dominate | STAGE with one request lane, 384 threads/CTA, launch-bound80 registers; producer gives registers to consumers. Same total logical splits and merger. Targets two CTAs/SM, subject to the native occupancy receipt. Same four cells. |
| SOFTMAX / f2116fdc | Softmax exponent issue dominates | STAGE plus 50% degree-3 FP32 FMA exponentials and 50% MUFU on the fast path; existing tensor-core row sum and rescale path retained. No extra K pass or skipped keys. Same four cells. |
| OCHAIN / b92d9094 | Attention-to-library output boundary dominates | Replace the O128/O256 library plans with existing notice-preserved native NF BF16 WGMMA kernels with PDL. Attention unchanged. v16S/F,v32S/F. |

`manifest.json` records full source, shipment, module and patch hashes.
`build.py` regenerates the candidates/patches from the exact base and the
reviewed `public_vector` sources. Generated files, including the `bundle.json`
`build_lab.py` reads, are ignored in `out/`.
No candidate is nominated before a native receipt.

## Verification

- Six exact candidate files, normal **and** forced Hopper glue, through
  `tools/local_eval.py` in the judge image on GPU0: **156 TASK checks and
  144 production-graph checks**, all pass. All 13 public TASK cases and
  all 12 scored cells run for every exact candidate/mode.
- Six-arm packed batch (`10c8f076`'s payload), normal and forced, context4096: **116 TASK +260
  nearby checks each**, all pass. Nearby means the audit's 65 cases,
  two seeds, A and primary arm. Context4096 exercises multiple K tiles;
  the native payload retains the exact scored context80000.
- Split labs (below), each in normal and forced mode, context 257: core **92 TASK + 260
  nearby**, stage **88 + 260**, all pass. Exact ec754e5 full-protocol rehearsal (64 fresh seeds x 16
  symmetric rounds): core **2,944 TASK + 260 nearby**, 8,694-byte wire; stage **2,816 + 260**,
  9,103 bytes; both complete with no failure, each packet read back after a 4,000-character
  suffix. Counts and wire sizes are in `p5b_<lab>_<mode>_screen.json`.
  Local times are never used to rank candidates.
- Native sm90a compile: staged six-kernel family plus O128/O256, all
  pass; dependency release/wait instructions present. Stage NL2 uses96
  initial registers, NL1 uses80; **both still spill**. O kernels use90/154
  registers with zero spills. Native `stage_resources` records actual
  registers, local bytes, dynamic smem and resident CTAs in the receipt.
- Exact two-lane staged-copy/read/lifetime GPU probe: 18 BF16/FP32
  cases, **106 cross-request CTAs and2 active/idle peers**, bit-exact,
  including gate reads after the temporary storage is overwritten.
- SASS verifies aligned LDG128/LDS128 and native HGMMA. On both verify
  specializations the mixed exponent arm changes static MUFU.EX2
  **151→135**, FFMA **437→485**. These are static instruction counts,
  not dynamic cost measurements.

sm120 cannot execute the native Hopper WGMMA/DSM/PDL paths. Stand-ins
exercise their arithmetic, Python dispatch, allocation, capture and pointer
glue; native TASK, two-CTA residency and speed remain unmeasured.

## Paired checks

`10c8f0762f7246de` (all six arms, 840 s import budget, sent 10-27) is **unmeasured**: the judge killed
it at the 900 s cap (`timeout`, returncode -9) and returns no log on a timeout, so neither the packet
nor the attention replay fix in `tools/h200_lab.py` was exercised. A run that progresses needs about
400-550 s: P3a's S22 build lab `f6970526` (7 arms x 12 cells, the same 64 x 16 protocol, 5,376 TASK)
finished its import in 405 s, and this one had 3,712 TASK. Timing stops at the deadline, so a stall
outside that loop (setup, compile, or a native hang in kernels never run on Hopper) is likelier than
the budget; the receipt cannot say which. (The in-graph diagnostic `32889c27` that was to prune
CHAIN/DIRECT is also unmeasured: its packet was cut by TASK-stub tracebacks, which the replay fix
addresses.)

Resent 10-27 as two lab-only checks on the attention host with the same seeds, built by
`build_lab.py --keep out/keep_<lab>.json --name p5b_<lab>`; the riskier new staged kernel family is
isolated so a stall there cannot cost the other arms:

| Lab | Arms (primary runs all 12 cells as INC) | TASK | Job |
|---|---|---|---|
| `p5b_core` | DIRECT (INC), CHAIN, OCHAIN | 2,944 + 260 nearby | `5caaec87d4984c85` |
| `p5b_stage` | STAGE (as INC), LANE1, SOFTMAX | 2,816 + 260 nearby | `d97ca0e9571c46a0` |

Each lab prints `P5N_STEP` at every setup phase and cell and keeps them in its packet, declares a
720 s budget (300 s compile + 400 s run), and runs a watchdog thread that prints `P5N_STALL` with
the current phase, cell, arm and seed and exits 50 s after the deadline, inside the cap. On a
simulated GPU hang (sync, `.item()`, graph replay, full launch queue, and both payloads with a
hung host build) it printed its line and exited at the guard time.
The judge rejects sources that start processes, so a main thread blocked while holding the GIL is
not covered; the `P5N_STEP` lines still name the last phase then. `read.py <receipt>` picks the
manifest (`lab_manifest_<lab>.json`) by the sent wrapper's sha and returns the progress lines when
there is no packet.

The paired batch uses P4 **ec754e5**: fixed64 disjoint held-out input seeds,
new weight/router seeds,16 forward/exact-reverse cold rounds, arithmetic
mean of **every** round, unchanged TASK for every active arm/seed. A and
its A2 twin run all12 cells; primary INC also runs all12. Other candidates
run their changed cells, taking A2 on cells with identical dispatch. The
receipt retains all per-seed six-load ratios, per-arm round counts, worst
four primary traces with all rounds/host enqueue, and per-block NVML.
The complete compressed wire packet is tested with a4000-character suffix;
no sample, timing round or failing check is discarded to fit it.

The boundary is the full production `kernel_fn` graph, including prepare
and cast **kernels** that occur in the timed call. Each load is46S+16F;
`read.py` reports S22/new with the paired/A:A larger log SE and G0 CI.
Correct improving points go to P4 as these exact licensed patches; a
selected composition requires fresh disjoint G0 confirmation. The lab's
compile pool is not a measurement of a single candidate's300-second
prepare cap; P4 checks that cap on its final selection.
