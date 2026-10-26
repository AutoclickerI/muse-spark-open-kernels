# P10 exact dense coding and fixed costs

The current handoff is **exact12 fused gate at T1 sliding only**. Against frozen S19 it passes the held-out six-load gate: **R1.004405 [1.003003,1.005808]**,16freshseeds/576public TASK checks, receipt **e364381789e8445e**. The added prepare cost is9.192027s, including3.525ms GPU packing/metadata; total170.454230s. `patch_gate.py BASE.py OUTPUT.py` produces the independently screened integration patch. P4 must confirm its new composition and complete its official decode run after quota resets.

`P4_HANDOFF.json` pins the base, patch, candidate, receipts, intervals and prepare costs. `winner_validation.json` records normal/forced13TASK/12captured-shape validation; `whole_results.json` retains all paired H200 samples. The candidate keeps gate computation concurrent with attention and waits for attention output only before sigmoid/multiply output. The full projection result and consumer ABI stay at their original precision.

[Every projection interval and bank preparation cost](shape_ratios.md), plus [all per-shape layer-change intervals](layer_ratios.md), covers qkv/qkv3/gate/O/pool_router/unpooler at T1/8/16/32/128/256. `ratios.csv` and `results.json` retain every arm, including losses and inactive diagnostics. Large-shape follow-ups are in `pipeline_results.json`, `grouped_results.json` and [wide-change intervals](wide_ratios.md).

| H200 receipt | Decision |
|---|---|
| dc6e811d5cc04f89 | Unmeasured; import diagnostics were lost before host weight setup. |
| 2b2f5c80673444a2 | Unmeasured; eager preflight identifies T256 early wide invalid launch. |
| cc06d34c38284ff2 | Complete36cells,4small/7wide arms,8seeds8cold,A:A,public projection checks including original FP32 router. Gate1 .919767 [.914989,.924571]; QKV8 .933678 [.925420,.942009]; best active wide paths lose1.272–2.123 with actual neighbours. |
| e7a553de66604682 | Frozen S19,8newseeds/576layer TASK. Broad code R1.000014 [.997643,1.002390], early .999654 [.997462,1.001851], merge .989549 [.986697,.992409], composition .989779 [.986745,.992823]. Rejected by G0. |
| e7b0d5786d174c26 | Wide raw/code oracle and decoder/MMA overlap,8cells26arms8seeds. Overlap saves1.3–6.4% within that schedule; best active code still loses1.212–1.947. Raw versions also lose most of that time. |
| e364381789e8445e | Fixed gate1/sliding candidate,16disjoint seeds, fresh weight/router seeds,576public layer TASK. G0 winner above. |
| ff32499a47704733 | Complete8cells26arms8seeds: K128 groups and adjacent token grid save5–19% inside matched register schedules, but best new code still loses1.190–2.092. No large-shape handoff. |

The old **1.36125×** leader advantage is a frozen component aggregate from08cc9405. This experiment measures the layer ABI, including fused N4608 pool/router and split gate concurrency. There is no new same-ABI paired measurement of that component residual. Dividing1.36125 by the layer R would mix workloads. The leader's code and per-shape times remain private.

## What changed and what the measurements isolate

Small code dispatches S4/WK4/ST1, S2/WK4/ST1 or S1/WK8/ST1. The fused gate performs its BF16-input dot before waiting for Y, then sigmoid/multiply directly into BF16 output. The early-weight arm uses volatile initial vector loads: SASS places3–6 vectors before the dependency wait; merely moving ordinary loads in source did not.

Wide code removes decoded-weight shared staging: exact BF16 weight fragments go into WGMMA A registers and are reused over128/256tokens. K splits write separate FP32 partials for existing norm/route consumers. A single GPU-coded body is shared across split1/2/4/8; GPU metadata repartitioning is exact. Router hi/lo groups retain separate exponent origins. Expanding the escape table to256 entries handles the measured worst88. The exact codec comparison covers469,762,048weight patterns.

`pipeline_results.json` compares raw/coded weights with identical register math, then overlaps decoding the next tile with current WGMMA. Matched code/raw ratios at the selected points are1.0015–1.183 for T128 and0.9965–1.0182 for T256. Therefore the decoder alone does not explain the gap. The raw register schedule itself is slow: K64 synchronization, activation transport and token tile order remain candidates. Grouping K128 reduces synchronization and schedules neighbouring token tiles together when they share weights.

Async A registers remain live until their matching wait; X shared slots are replenished only after their previous MMA drains, following [NVIDIA's WGMMA register ordering rules](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-instructions-wgmma-wait-group). Compilation confirms no spills/local stack in the submitted grouped body. An earlier dynamic register-array index produced256bytes of local stack; the submitted unrolled phase index removes it.

The T256 launch failure was a host linkage error: two extensions shared a GNU-unique template-local `ready` flag. The second library skipped its own dynamic shared-memory opt-in; T256 requires69,632bytes. Distinct body/launcher symbols isolate the flags. `readelf` proves the symbol separation; cc06 completes every entry. The first receipt lost diagnostics because host weight generation failed before prepare; import evidence now prints immediately through the shared lab transport.

## Reproduce

Run commands from the repository root; GPU commands use `CUDA_VISIBLE_DEVICES=0 tools/img python ...`.

- `build.py`, then `screen.py` and `screen.py --forced`: published projection TASK plus all dispatch arms; `compile.py`, `host_glue.py`, `test_codec.py` separate native resources, launch ABI and exact encoding.
- `test_lab.py`: complete packed matrix rehearsal, explicit Hopper arithmetic stand-ins; its local timings make no H200 claim.
- `read.py cc06d34c38284ff2`, then `report.py`: reproduce all matrix intervals and prepare costs from the canonical receipt.
- `whole_read.py e7a553de66604682`: selection verdict, preserved in `first_whole/`. `whole_read.py e364381789e8445e`: held-out confirmation with full six-load covariance. These commands overwrite `whole_results.json`.
- `whole_build.py --selection .../selection_confirm.json --confirm`: exact confirmation recipe. `whole_screen.py --lab` rehearses it. `whole_screen.py CANDIDATE --out DIR [--forced]` runs local_eval on the production patch.
- `pipeline_build.py`, `pipeline_compile.py`, `pipeline_screen.py`; `pipeline_read.py e7b0d5786d174c26`: first wide factorial.
- `grouped_build.py`, `grouped_compile.py`, `grouped_screen.py`: grouped factorial. All arms and seeds share one host run.

Only `tools/h200_lab.py send ... --host ... --lab-only --import-budget700 --lane P10 ...` launches remote work. Each canonical `runs/` source contains its complete payload. The lab-only host deliberately fails after evidence emission; its host `passed:false` is not a recipe correctness failure. Readers require the complete inner matrix, every published TASK gate, valid checksummed transport, and no transport device error. A failed/incomplete batch is unmeasured.
