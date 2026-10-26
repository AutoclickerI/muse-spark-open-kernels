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

Payload `2fe023f2`, wrapper `3de4daa1`, decode check sender **1294249**, priority2. Queue snapshot: queue.json; progress runs/detached/3de4daa1469726c1.log. The wrapper was built through h200_lab.py against the exact S15 host and sent unchanged via okc_run.py from this lane's out directory; it cannot be silently merged with another lab's output budget. No H200 result is claimed yet.

Rebuild with build.py then build_lab.py. Screen GPU0 in tools/img with screen.py [--forced], check_native.py, audit.py, check_lab.py; check_archive.py and admit.py validate exact identities and receipt round trips. Read the accepted job with tools/h200_lab.py read, then report.py RECEIPT --out out/h200.json. Keep every sub-1 shape point for P4 and confirm the composed score on disjoint seeds.

P4 nominations from earlier receipts are separate selection data: `../d1_chain/point_nominations.json` (ten changed d1 pieces) and `../moe_tail/point_nominations.json` (eight MoE shell pieces). No composition is declared promoted before P4 confirmation.
