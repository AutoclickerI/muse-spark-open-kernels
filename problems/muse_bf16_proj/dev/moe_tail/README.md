# P7 MoE route / finish / tail, S15

Base: exact S15 `60254883`. This lane owns the route metadata, output combine/finish and hand-off boundaries at M128/256. The round3 16.0/17.1 us shell is an allocation model; S15's measured internal route+finish is about 6–7 us, with other tail time overlapping expert activity. No modeled shell saving is claimed as measured.

## Coordination with P3b

P3b owns the heavy and light expert kernels, weight layouts, grids and launch order in `problems/muse_moe_w4a16/dev/mixed_wgmma/branches/round3`, round4 and round5. This lane reads those files and edits none of them. The rebuild keeps their S15 producer pointer ABI and native handles. Route metadata preserves C/OFF/ORDER/TASK/SLOTS/POS/T32/T8. Finish consumes unchanged FP32 Y/SS and emits the same BF16 output. The optional dependent-launch arm changes only the large-M <=8-token W2 entry release, compiled separately; heavy/light code is untouched. No winner can be assumed to compose with a new P3b producer until paired again.

## Rebuild

R aggregates same-expert routes within each warp before the shared histogram atomic. F8 uses128 threads per512-column CTA; F4 uses256 per1024. F4/F8 distribute the eight norms over four-lane groups with warp-private metadata, removing the CTA shared-memory hand-off; output accumulation keeps top-8 order and BF16 materialization. C combines R+F4. E tests only the small-expert W2 entry release. P uses E plus the original finish's immutable route metadata loaded by inline PTX before griddepcontrol.wait. All Y/SS remain after that wait. PF uses E plus the warp-private finish with the same ordering. A/A2 are the unchanged S15 control on independent captures.

Decision: require exact TASK gates, clean same-capture A:A, balanced AB/BA pairs and gains exceeding A:A uncertainty across fresh seeds. Compare E/P/PF to isolate producer tax and consumer overlap. Freeze shape/role rows only from whole-layer wins; a failed control or an unproven branch is unmeasured. Other route orderings, producer readiness and heavy-branch epilogues remain open after a loss of these recipes.

## Verification and ordering mechanism

Every A/A2/R/F4/F8/C/E/P/PF arm passes normal and forced `tools/local_eval.py` gates with one shared prepare per mode: 13 official tests and all12 captured shapes per arm. New route/finish math and poisoned-buffer graph replays pass360 native checks across M33/127/128/255/256/257 and local_start0/64, including changing IDs, Y, SS and weights; worst relativeL2 is recorded in native_validation.json. All7 full candidate files pass undefined-name and call-signature checks. Actual new CUDA compiles for sm90a without local-memory SASS or stack spills.

The initial P arm failed locally (verify16 sliding update relativeL2 .1154, worst-row .7063 above its .65198 allowance). sm90a SASS identified two dependent SS loads at0x280/0x290 before ACQBULK at0x310 despite source order. That prototype was replaced. The retained P/PF use volatile `ld.global.cg` for SS/Y after the dependency wait; SASS admits only four immutable metadata loads before ACQBULK. The corrected normal/forced screens pass. Native W2 has an entry release; no new polling flags, global counters or graph epoch state exist. Readiness follows [CUDA's programmatic dependency contract](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html); concurrency is an opportunity rather than a progress requirement.

The exact packed local rehearsal is four verify shapes x32 fresh input seeds x9 arms, with balanced AB/BA x8 per candidate/control pair. A/A2 use the same capture method and must have identical kernel/resource/stream/DAG signatures. Heavy/light kernel names, geometry and resources are compared as a multiset (their relative observed start times may vary). Both programmatic finishes have captured typed dependencies from all three W2 branches. Only declared R/F/P kernels may change. Full raw local observations are in local_raw.json.xz; check_archive.py reconstructs every retained paired seed within1e-5 ratio and .00501us reference-time error. Raw per-round times are not retained; exact per-arm mean A/B event times and per-seed ratios/reference times are retained.

The lab consumes the S15 host's actual weights/state and selects the busiest shard from its own router; all arms see identical fresh inputs. Local SM120 times are not used to pick H200 winners. MoE shell profiles retain route/finish intervals/resources, producer W2 ends and finish dependency types. The lab is one full-budget decode check. Its exact h200_lab wrapper is sent from this lane's ignored out directory, outside the shared lab directory, so another lane cannot automatically merge into its full output/time budget.

Profiles use three replays of the final fresh seed; paired ratios retain all32 seeds. Only archive encoding was compacted after the successful GPU rehearsal; local_receipt.txt is explicitly labeled as a repack of its actual observations.

## Dispatch

Queued decode check, sender **1090184**, priority **2**, wrapper SHA256 `696d82e6`, payload `df8357b5`. The first unsent sender1007502 was verified under the dispatch lock and replaced; no duplicate H200 job was spent. queue.json names the final snapshot; replaced_queue.json preserves the earlier send. Progress: `runs/detached/696d82e66013ee55.log`; queue.json pins the actual snapshot. No H200 shell win is claimed before its receipt.

Read the accepted job through `python3 tools/h200_lab.py read JOB`, then run `python3 problems/muse_bf16_proj/dev/moe_tail/report.py RECEIPT --out problems/muse_bf16_proj/dev/moe_tail/out/h200.json`. Check each shape's A:A and gates before comparing the weighted load. P4 receives only a measured winning shape/role patch onto S15; every unchanged load must remain neutral.

Reproduce: `python3 .../moe_tail/build.py`, `python3 .../moe_tail/build_lab.py`; GPU0 judge-image `screen.py [--forced]`, `check_native.py`, `audit.py`, `check_lab.py`; then host `check_archive.py` and `admit.py`. Complete normal/forced official and Python static receipts are compressed in screens.json.xz.
