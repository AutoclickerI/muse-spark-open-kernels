# P3a correct S22 outcome portfolio

Active. Native H200 results pending.
Frozen base S22 expanded **4b8c803db080f4e0f2a69e4c666ee91573fc6ffde238d2e63db4223f83f84d27**. Paired sender **984504**, wrapper **ab36c11e7cb7ea6cb1f7fdddde9e822b0a53550b66db80ca161a8f38b5c07a25**; `--lane P3a`, PLAN rank4,MoE check after09:00KST10-27. Older diagnostic sender3637718/wrappere1710fb3 precedes it. No scored submission.

| Candidate | Mechanism | Diagnostic outcome targeted |
|---|---|---|
| CACHE1 | Real64KiB BF16 activation bank,two vector planes,one raw TMA slot | Fragment placement/Bf lifetime;real fill/LDS/one-slot costs paid |
| LAZY | Load original BF16 fragments per K-group/tile instead of keeping64 registers live | Bf lifetime/activation scheduling;register-lifetime control for LAZY2 |
| LAZY2 | LAZY plus two alternating FP32 accumulator chains,final sum | Accumulator dependency latency with freed register capacity |
| N1SIMT | FP32 SIMT matvec only for actual singleton token blocks;HMMA otherwise | Useful-N padding;all K,experts,outputs retained |
| EPI4 | Four epilogue warps,scalar W2 channel ownership;remove duplicate norm flush | Epilogue serialization;original reduction/readiness retained |

| Candidate | Registers M1 / M8+ | Shared B M1 / M8 / M16 / M32 | CTAs/SM | Stack/spills |
|---|---:|---|---:|---|
| CACHE1 | 94 / 60 | 112688 / 112960 / 113280 / 113920 | 2 | 0 |
| LAZY | 94 / 96 | 83904 / 84176 / 84496 / 85136 | 2 | 0 |
| LAZY2 | 96 / 100 | 83904 / 84176 / 84496 / 85136 | 2 | 0 |
| N1SIMT | 106 / 88 | 83904 / 84176 / 84496 / 85136 | 2 | 0 |
| EPI4 | 115 / 126 | 83904 / 84176 / 84496 / 85136 | 2 | 0 |

| Candidate | Whole six-load speedup R [95%CI] | Per-shape point nominees | Leaves open |
|---|---|---|---|
| CACHE1 | unmeasured;await native receipt | none yet | Other schedules/layouts and P4 composition;only this exact recipe can close |
| LAZY | unmeasured;await native receipt | none yet | Other schedules/layouts and P4 composition;only this exact recipe can close |
| LAZY2 | unmeasured;await native receipt | none yet | Other schedules/layouts and P4 composition;only this exact recipe can close |
| N1SIMT | unmeasured;await native receipt | none yet | Other schedules/layouts and P4 composition;only this exact recipe can close |
| EPI4 | unmeasured;await native receipt | none yet | Other schedules/layouts and P4 composition;only this exact recipe can close |

**Correctness and local scope.** All five exact candidate math bodies pass270 leaf TASK gates each: M1/8/9/16/24/32,real/all/empty/singleton/dense routes,local_start0/96/224,three changed inputs per captured graph. All five shipped candidates pass normal and forced Hopper-glue screens:13TASK cases and12captured shapes per mode (130TASK,120graphs total). All three small-M native launch paths are observed eagerly and captured. Fifteen exact sm90a translation units compile;20shape/layout occupancy rows retain two CTAs and zero spills. CACHE1 needs113KB shared memory unavailable on sm120:only its activation-bank address space/LDS is replaced with preallocated global storage locally;math,one-slot raw TMA,ordering,allocation and native launch remain. The H200 payload contains the actual shared bank. Other Hopper-stage stand-ins are the existing S22 screen hooks. Native H200 correctness and performance remain unmeasured until the receipt.

**Transport rehearsal.** The exact packed payload passes 168TASK+24nearby checks on all12graphs, with an explicit local two-seed/two-round plan and forced stand-ins. This is not a64-seed H200 result. Lossless archive round-trip passes; synthetic64×16 packet stress is11645B within11744B,including four complete retained timing traces with NVML. Synthetic values never enter results. The codec losslessly packs the existing six-decimal sample integers and regenerates deterministic round orders; all six loads,seeds and round counts remain. Receipts use checksums and immutable source audits.

**Native measurement plan.** ec754e5/S21 protocol:64 disjoint held-out seeds×16 rotated forward/reverse rounds, arithmetic means of every round,cold-cache flushes,all12whole-production-graph shapes for every arm. A and A2 are separately captured exact S22 graphs using the common baseline preparation;five candidates give5376TASK+780nearby checks. Actual per-seed six-load times use46sliding+16full;no off-shape imputation or component-time score transfer. Every primary six-load seed sample and round count is kept,plus bounded worst4 per-round GPU/enqueue traces and NVML before/after timing blocks. Targeted eight small-M shape moments retain paired log-SE;four heavy shape moments are redundant and omitted,but their actual six-load samples remain. Prepare is reported as shared host preparation plus each arm delta,a conservative bound rather than a cold candidate measurement.

**Gate and handoff.** `read.py` audits wrapper,payload,bundle,checker identities and calls `promotion.evaluate`. Every valid whole-layer shape point win goes to P4 with ratio,SE,seed IDs and exact S22 patch. A:A sets uncertainty without vetoing individual shapes. P4 composes and confirms on new held-out seeds; six-load R lower95%bound>1 plus TASK/nearby establish the lab gate. Official TASK and the300s prepare limit are still required for final submission. No measured winner exists before the native receipt.

**Finite follow-through.** `follow.py` waits on receipt/file and sender-exit events,reads the diagnostic, and removes an arm only if the predeclared matched upper saving bound is nonpositive at every M (LAZY requires a measured loss at every M). A merely non-largest positive term is retained. Failed controls/implementations are unmeasured and do not close a recipe. Only the owned unsubmitted paired waiter can be replaced under the dispatch lock;an accepted job is never cancelled. If dispatch outruns the diagnostic,all arms stay. The worker reads the one eventual paired receipt, exports P4_HANDOFF/point_nominations and ends;no automatic scoring or shared-doc writes.

**Exact rebuild.** `base_snapshot.json` pins the original packed S22 Git blob,so P4 edits cannot silently change the base. Run `build.py`,then `count.py` inside tools/img,`check.py`,`run_screens.py`,`build_lab.py` and `rehearse.py` on GPU0. `manifest.json` pins expanded/shipped/patch/cell hashes. Each patch applies to the expanded `dev/s22/generated/s22.py`,not the compressed shipment. Only the three small-M CUDA literals change;all notices and remaining graph code are retained. Native resource checks repeat on H200.

One EPI4 M1 local failure was traced to SLOT metadata never initializing expert-wide n/off. Keeping the actual segment off/ne fixed it;every leaf,normal/forced and packed check reran on the final source. It is an implementation fix,not an H200 loss. No closed mixed_wgmma/public-port design is reused.
