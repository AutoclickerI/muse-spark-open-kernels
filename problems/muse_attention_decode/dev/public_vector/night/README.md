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
reviewed `public_vector` sources. Generated files are ignored in `out/`;
the sealed live bundle remains there while the queued worker needs it.
No candidate is nominated before a native receipt.

## Verification

- Six exact candidate files, normal **and** forced Hopper glue, through
  `tools/local_eval.py` in the judge image on GPU0: **156 TASK checks and
  144 production-graph checks**, all pass. All 13 public TASK cases and
  all 12 scored cells run for every exact candidate/mode.
- Final packed batch, normal and forced, context4096: **116 TASK +260
  nearby checks each**, all pass. Nearby means the audit's 65 cases,
  two seeds, A and primary arm. Context4096 exercises multiple K tiles;
  the native payload retains the exact scored context80000.
- Exact ec754e5 full-protocol local rehearsal: **64 fresh seeds x16
  symmetric rounds**, **3712 TASK +260 nearby**. Full counts, transport
  validation and payload identity are in `normal_protocol_screen.json`.
  Local times are never used to rank candidates.
- Pruned-portfolio rehearsal (DIRECT/CHAIN absent, STAGE renamed to the
  protocol's primary INC): forced, **96 TASK +260 nearby**, all pass.
  Candidate source identities remain unchanged; pruning only removes arms.
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

## One dependent paired check

Diagnostic sender **3831549** stays unchanged: attention/check, PLAN rank2,
not-before **1793059200 /2026-10-27 09:00 KST**. `dispatch.py` waits on that
sender's completion event, reads the exact receipt through `h200_lab.py`,
then sends **one** paired production-graph check on the attention host with
`--lane P5b --decision "S22 outcome candidates after in-graph attention diagnostic"`.
Dependent worker **691620** and sealed admission **0e3e8aa7** are recorded
in `queue.json`; its status is waiting on the real diagnostic-completion event.
It does not consume a quota slot or send a second diagnostic overnight.

Decision rule: remove CHAIN only if every changed cell's diagnostic D
new/control lower95CI is >=1; remove DIRECT only if all six F and both
native d1 V cells meet that condition. A partial/uncertain loss retains the
whole candidate, allowing per-cell choice. STAGE/LANE1/SOFTMAX/OCHAIN are
new implementations and are not refuted by another recipe's loss. Failed
native controls or TASK are unmeasured; the worker stops for inspection.
No arms or measured rounds are selected after this paired check begins.

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
