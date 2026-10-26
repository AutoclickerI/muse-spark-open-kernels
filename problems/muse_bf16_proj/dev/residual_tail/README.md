# P7 residual tail, exact S15

S15 source SHA256 `60254883`. This lab targets the call's total T<=8: decode1 and verify1. It never uses internal chunking to extend the sliced unpooler shortcut.

## Source claim audit

`_best7_run` selects `unpooler_slice` when t<=SLICED_MAX_TOKENS (8). The selected bank has1,024 output columns, and `_best7_boundary` passes its NB=1024 into `glue_cl::boundary_cl<8,256,1>`. That kernel explicitly substitutes zero for j>=NB, then still computes `alpha2[j]*stream1[j]`, squares it into the second norm, and writes the entire8,192-element residual and normalized hidden output. Thus the final7,168 residual elements and their squared sum depend only on b1's stream1 and immutable alpha2. b1 already holds stream1 in FP32 registers at its final output loop. The identity exists exactly where the review described it; it is not a wait-removal claim.

The branch RMS and next hidden RMS denominators stay8,192, as do the epsilons1e-8 and1e-5. All8,192 residual/hidden elements are written. The shortcut is enabled only for actual CL8/NT256/one-round b1+b2 with totalT<=8; other original dispatch rows execute unchanged. Captured target rows must prove the new kernels actually launch.

## Arms

A/A2: exact S15, independent captures. V: b1 writes alpha2*stream1 on the tail directly into b2's final residual storage; b2 loads it for normalization/output and writes only the first1,024 residual elements. V still reduces all8,192 norm elements. N: b1 writes seven1024-element squared-sum partials; b2 retains its residual computation/output but replaces tail norm reductions with those partials. C: the combined review recipe, with V and N retained as one-change ablations. In N/C b2 reduces only the first1,024 and adds all seven cached partials before dividing by8,192. Its tail CTAs still write every BF16 hidden element.

Only current-call b1 data enters the workspace, allocated inside that call/capture. There are no cross-call counters, cached outputs, input-dependent host branches or inter-CTA polling. All b1 branch inputs, stream1/tail/statistics in b2 use explicit ordered PTX loads after the grid dependency wait. Every graph replay overwrites all seven statistics and every tail element used. The native tests poison workspace/outputs and change the inputs between replays.

## Decision and P4 hand-off

Every changed arm/shape with candidate/control point estimate below1 is handed to P4 with SE and held-out seed IDs; A:A is retained as a noise diagnostic, not a per-shape veto. P4 composes shape choices and confirms six-load R=exp(mean(log(T_old/T_new))), T=46t_sliding+16t_full, with lower95% bound>1 on disjoint seeds. Correctness is TASK.md's checker only. A component numerical difference or a routing near tie is not an extra rejection rule. Unchanged duplicate captures are identified as controls, not optimizations.

## Screens and dispatch

All five A/A2/V/N/C arms pass normal and forced tools/local_eval.py checks:130 official test cases and120 captured shapes total. The native replay test covers512 mode replays, totalT1/2/4/8, BF16/FP32 split inputs, changing stream/branch/alpha2 and poisoned workspace/output. The final source is bit-equal to the original on those native fixtures; exact tail identity and cached squared sum are checked separately. sm90a compiles all six native specializations without local-memory SASS or stack spills; V b1/b2 use78/76 registers, N96, C111. This register growth is a measured resource cost, not an H200 latency prediction.

The exact local packed rehearsal passed640 copied public TASK checker calls (four shapes x32 seeds xfive arms),32 balanced AB/BA x8 seed pairs per arm, identical A:A capture signatures and proof that both target boundary kernels changed. It used20.1s after prepare. No baseline-relative error threshold or routing-membership veto is applied. The diagnostic TASK namespace memoizes common oracle subexpressions only within one fresh fixture; every arm still runs the unchanged public check_implementation body on its own collected output/cache, and production kernels consume none of that data. Source/hash audit: claim_audit.json, native_audit.json, admission.json. Full raw local observations: local_raw.json.xz. Normal/forced receipts: screens.json.xz.

Payload `2fe023f2`, wrapper `3de4daa1`, sender1294249 completed as H200 decode check **e841e2e8b1d04f0c**. Host passed; all640 public TASK checker calls passed on32 seeds, all A:A schedule/resource/DAG fingerprints identical. Lab16.4s, host prepare including lab196.2s. `h200_result.json` retains ratios/SE/seeds, exact receipt hash and sampled boundary timings. Ratios below are candidate/control latency; all are selection data, not held-out G0 promotion.

| Shape | A:A (SE) | V (SE) | N (SE) | C (SE) |
|---|---|---|---|---|
| d1S |1.011209(.003787)|1.013104(.003155)|1.007888(.003012)|1.012938(.002364)|
| d1F |.993489(.001628)|.996049(.001739)|.985986(.002102)|.990038(.001966)|
| v1S |.997094(.002187)|1.033689(.002733)|.997537(.002102)|1.004485(.002453)|
| v1F |1.009460(.002267)|1.038322(.001787)|1.001321(.002010)|1.008552(.002053)|

Four exact S15 point patches are handed to P4 in `point_nominations.json`: V/N/C at d1F and N at v1S. Each carries32 seed IDs, SE, candidate/patch SHA256 and base60254883. `handoff.py` reproduces them under out/point_handoffs; every patch passes git apply --check. The selected N(d1F,v1S) composition is candidate `d162253e`, patch `63568db1`, with unchanged S15 dispatch everywhere else. P4 owns held-out six-load confirmation; no standalone full-six-load tail-only confirmation is sent, following the owner's priority to rank mechanisms before production-scale small-gain tests.

In the initial receipt, stats-only N is the point leader. V is specifically adverse at verify1 (+3.37/+3.83%), and C does not recover those shapes. Sampled b1 residence rises with all arms; N d1F b1/b2 A5.280/5.824us ->5.472/5.567us, v1S3.456/6.113 ->3.616/5.984us. These residences include predecessor waits, so they do not prove a critical-path saving. Registers rise57->96 for N,111 for C. A CPU sm90a probe moving tail production after output stores leaves N/C register counts unchanged, so that source-order change is not sent. Follow-up `rebuild/` (payload2947227d, wrapperd210d9bc, decode check11d531dcb81d4b7f, sender1663403) specializes N's seven tail CTAs to omit prefix-only branch arithmetic/loads, with explicit ordered dependent loads, and keeps A:A plus the measured N arm on32 disjoint seeds. All104 normal/forced TASK tests,96 captured shapes,384 native replays and512 full-batch TASK checks pass; exact SASS keeps the two immutable tail weights before the wait (98regs versus114 after it) and dependent stream/statistics after it. The archive includes full per-role intervals and conserved critical increments. It leaves register/overlap causality and integration interactions open.

Rebuild with build.py then build_lab.py. Screen GPU0 in tools/img with screen.py [--forced], check_native.py, audit.py, check_lab.py; check_archive.py and admit.py validate exact identities and receipt round trips. Read the accepted job with tools/h200_lab.py read, then report.py RECEIPT --out out/h200.json. Keep every sub-1 shape point for P4 and confirm the composed score on disjoint seeds.

P4 nominations from earlier receipts are separate selection data: `../d1_chain/point_nominations.json` (ten changed d1 pieces) and `../moe_tail/point_nominations.json` (eight MoE shell pieces). No composition is declared promoted before P4 confirmation.

Selected-composition local verification: `selection_screens.json.xz` records26 public TASK test passes and24 captured shape passes in normal/forced modes; `handoff_validation.json` checks every patch and exact candidate SHA. This is nomination evidence; P4 still owns held-out G0 promotion.

Follow-up completed as11d531dcb81d4b7f: all512 TASK checks pass. N's earlier d1F/v1S points do not repeat (1.004723/1.002925); new v1F N/T points .997095(SE.001811)/.994935(SE.002209) go to P4. T v1S loses1.038238 despite a smaller sampled b2 increment; extra pool/router/MoE time precedes b2 launch, leaving capture/allocation/clock effects unisolated. Details, controls, critical intervals and exact S15 patches are in rebuild/h200_result.json, rebuild/h200_mechanism.json and rebuild/point_nominations.json. The original four points remain selection nominations; none has cleared a held-out full-six-load G0 gate. No third tail-only lab is sent before larger levers are fixed.
