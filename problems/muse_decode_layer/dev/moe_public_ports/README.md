# MoE public-port lab (P3a / P3b)

Compare reviewed public kernels with frozen S19 `62ea5bf4` at M1/8/16/32/128/256. Whole-MoE graphs retain S19's original dispatch, overlap, route, and finalize. The independent component TASK reference checks every arm, eager and graph, on the official nearby tests and on every timing seed.

Donors were cloned with hooks disabled into separate new directories under `/data2/jihoon/muse_kernel/review/moe-public-Ury55uMA/`, outside this repository. No donor installer, build driver, test, notebook, diagnostic, or tuner was run. `review.json` proves identical operator device ASTs and Lee fused math ranges. Reviewed kernels, owned launch wrappers, complete MIT/Apache texts, and the operatorstack NOTICE are in this directory. Lee has no NOTICE at its pin; the source identifies our modifications.

- `operator.py.inc`: operatorstack `6d9b77656b6c295e421be9e1424b0f8f99299b0a`; all 22 device functions unchanged. The host is a plan of route/W13/W2/finish closures with the donor's exact shape choices. M1's W13 reduction is counted in W13; W2 partial reduction is in finish.
- `lee_x.cu.inc`, `lee_repack.py.inc`: Lee `601c65413496cf63f942b4f15ca087839ef2f5cc`, MoE v43d expert-wide RS-WGMMA. X1 is RPW2/NCW8, X2 is RPW1/NCW16, both TPX64. Original fused math/scheduling is an arm at every M. Explicit projection specializations implement a split ABI too; W2 includes its necessary scratch clearing. Standalone finalize uses the original vector loop. Full split pipelines also pass TASK.
- `lee_f.cu.inc`: v43d's two published small-M fused choices, MT2 with/without LASTA at 196KB. On-chip W13/W2 remain inseparable; they are whole-MoE arms at M<=32.
- `runtime.py.inc`: exact S19 per-M dispatch plus isolated phase envelopes over all its expert branches. A projection barrier is used only in isolated measurements; whole S19 retains its original graph.

Protocol: eight new input/router seeds, one new weight bank, four cold rounds in two rotated forward/reverse cohorts, identical work and all rounds for every arm. Mean all rounds within seed, paired port/S19 ratios and 95% t7 intervals across seeds. A2 is an independently allocated exact S19 control. Small S19 fuses W13/W2/finish, so those denominators are N/A. Phase timings use refreshed ready fixtures and are diagnostic, not additive whole-layer savings or G0 promotion.

Build with `python3 problems/muse_decode_layer/dev/moe_public_ports/build.py` after placing the pinned reviewed clones at the builder's explicit path. Generated payloads and local logs live in ignored `out/`. `screen.py` checks actual operator kernels and Lee's mma.sync emulation; `compile.py` compiles all four dispatched configurations for sm90a; `glue.py` forces S19/F launch glue with stand-ins. `tools/local_eval.py` covers static admission and the lab-only transport.

Send only with:

```sh
python3 tools/h200_lab.py send problems/muse_decode_layer/dev/moe_public_ports/out/public_moe.py --host muse_prefill_layer --lab-only --import-budget 840 --check --lane P3b --decision "port public MoE kernels per M" --note "reviewed public ports against exact S19"
```

`analyze.py <runs/...json>` decodes every integer-ns round from the checksummed binary packet and writes `H200.md`, `result.json`, and `receipt_data.json` without dropping samples. `manifest.json` pins the complete sources/checker/protocol. `P4_HANDOFF.json` and exact patches are the integration interface when measurements complete.

Receipt `5e45775a66a9416f` is unmeasured: the first official M4 TASK gate exposed an omitted generic S19 counter binding in the reduced prepare. The original `smallm_bind` was restored and M4 added to the local forced-glue check; the primary cohort was rerun unchanged.

The native cohort completed as `550f0aa88785489f`: 598 TASK checks, all six A:A intervals contain one, 8 seeds × 4 rounds retained, 41.1 s including setup. All whole-port cells lose. See `H200.md` for all configurations and native phase times.

A contract audit found the native W2 and finalize ratios assigned norm/SS work differently. `build_abi.py`, `abi_helpers.py.inc`, `operator_abi.py.inc`, and `abi_finalize.cu.inc` therefore adapt the candidate phases to **S19's sorted BF16 ACT, FP32 Y and 64-part FP32 SS**, with metadata cost timed. Local `screen_abi.py` passes eager+graphs on M33/128/256/257 and skewed M128; the changed finalizer compiles sm90a. Follow-up `17b91ef9e95841e7` passes 132 TASK checks on eight new seeds, all rounds retained, 36.6 s. `ABI_H200.md` has the matched phase and actual hybrid whole-MoE ratios.

**Decision: retain S19 at every M.** No winner patch or G0 nominee is sent to P4. `P4_HANDOFF.json` includes the negative results and M1–32 phase times; `MECHANISMS.md` explains every initial sub-kernel point below one and why none establishes a transferable gain. O W13 M128 remains unresolved (.9814 [.9045,1.0583]); the native O W2 / Lee finalize advantages disappear on the actual S19 ABI. The unchanged S19 finalizer's control points are not donor gains.

`source_formatting.json` documents blank license-comment space cleanup in the readable artifacts. It reconstructs exact measured source bytes; both regenerated payload SHA256 values still match their native receipts.
