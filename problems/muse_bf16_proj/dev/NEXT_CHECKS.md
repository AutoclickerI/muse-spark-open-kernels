# Projection next checks

Target: an H200 scored run at **1.26x or better**. Work alone; GPU0 only. The orchestrator
sends and commits; this thread does neither. All distinct ready candidates are eligible,
including diagnostics. There is no check-then-promote or diagnostic hold in this round.
The order below recommends h200, native, max, then all; existing queued files retain their dispatch state.

## Ready sources, in recommended order

Four new broad portfolios are ready. The preferred **`cand_portfolio_h200.py`** retains
the latest fused-router and 32 MB NN/Lt winners from `f3786341611f4f1f`, as well as the
previous `dd989ef2d76147e7` winners, and keeps both sets in the final paired comparison.
It passes 9/9 tests and 30/30 graph shapes at **1.102951x sm_120**, **39.81 s prepare
with the extension cache warm**, **12,286** forced Hopper cases and an sm_90a build.
Its GPU bodies match native/max. The parent `cand_portfolio_native.py` explicitly
enables native BF16 DeepGEMM and also retains the SDK cuBLASLt path. It passes 9/9 tests,
30/30 graph shapes, **12,281** forced Hopper cases and the sm_90a build. Its local score is
**1.111248x**, prepare **63.91 s with the extension cache warm**. GPU profiling confirms
`deep_gemm::sm120_bf16_gemm_impl` actually runs; this is not an H200 measurement.

The other broad portfolios remain ready. `cand_portfolio_max.py` adds shared-X tiles, active
multicast WGMMA and the default DeepGEMM SDK call (which routes to cuBLAS) to `cand_portfolio_all.py`, which packs the measured H200
router winners, fused hi+lo/merge kernels, all returned Lt heuristics and checked split-K,
NN/TN weight layouts, Stream-K, DSMEM, skinny/GEMV/TC/scalar-m16/DSV3 alternatives.
Its frozen source passes 9/9 tests and 30/30 graph shapes at **1.117166x locally (sm_120)**.
Cold prepare is **122.66 s**, including **65.53 s** compiling/repacking. All 135 skinny,
paired and legacy specializations compile for sm_90a. Forced Hopper dispatch checks pass
**9,912 cases** across every family. These are not H200 speedups.

The following files already in the orchestrator queue must not be resent or edited.
The max superset passes 9/9 tests and 30/30 graph shapes at **1.120767x sm_120**,
**71.77 s prepare with a cold extension build** (38.92 s compile/repack; shared image-library
JIT caches may already be warm). It passes **12,265** forced Hopper cases, **72** native
wide-kernel TMA/MMA fresh-input graph cases and an sm_90a build. No new H200 claim yet.

| Order | Candidate | Frozen SHA256 | Local score (sm_120) | Dispatch state |
|---:|---|---|---:|---|
| 1 | cand_portfolio_h200.py | `37b8c8070ad812db8c545c49756f1a971cc76501eb9e7eb63c238a3515510d8b` | 1.102951x | ready, not sent |
| 2 | cand_portfolio_native.py | `8018615509eada252cd105526b6d81c5cdfbe181eb620d23fea0cd578ed00eb6` | 1.111248x | ready, not sent |
| 3 | cand_portfolio_max.py | `f11614551f939216c0dfbc4f0e9d8402e56cb1ea96ac27e8fcec9f8ec2ff9ead` | 1.120767x | ready, not sent |
| 4 | cand_portfolio_all.py | `65ea3171f44e26fd1b1614b2d0d8c7a5da87cf7b8787338d25888d2bc1e612cd` | 1.117166x | ready, not sent |
| 5 | cand_reduce_hier.py | `051b1eeb33f0bd1cc38a19ff8273eaad2a9254fba92529e8be869b464ebbef33` | 1.091486x | local queue 3916761 |
| 6 | cand_reduce_packed.py | `d19dcd2bc9c0d036297bc70474318aac6e465d83267d88dff9add5ca78d5211e` | 1.091603x | local queue 3916762 |
| 7 | cand_reduce.py | `280822ad5f91aae8e20e9440032f3f3ce22043119179375c4087059ee0c887f6` | 1.093592x | local queue 3916759 |
| 8 | cand_reduce_fast.py | `b28e00958daef31379f8e3df4698e43be3ae05487c0e09c7d36204248132413e` | 1.078570x | local queue 3916760 |
| 9 | cand_lt_layout.py | `6c65aff44f84301c8eb9b772a1a2ac2ab3cece9429e44a2c6d0adb207d2c0381` | 1.088505x | local queue 3916758 |
| 10 | cand_dsv3_select.py | `0453c6ae3dd3b05a46846f556da94290a0b20b31a0c158e354dae72d1523b4c2` | previous screen; see receipt queue | local queue 3890089 |
| 11 | cand_dsv3.py | `8d9d827cd15b00b22a898d851f1a49e04450fb15cbad494a6d1cf19ca011a624` | previous screen; see receipt queue | local queue 3890092 |
| 12 | cand_gemv3.py | `16ba69bb100b069ca53d03096057f20261561feeb689a2139a7215f39178c9ec` | previous screen; see receipt queue | local queue 3890094 |
| 13 | cand_gemv_static.py | `f50425383a9c75e206fbf5d13202ac7ff6c7883087c6900acb9a6a3ab7e7a76a` | previous screen; see receipt queue | local queue 3890182 |
| 14 | cand_m16.py | `48c5397b78df800e9cc843bdd0659c5475bf745ca65c0184c81d8fea5bff017e` | previous screen; see receipt queue | local queue 3890098 |
| 15 | cand_skinny.py | `89bb6387a6d43ebf53cae10faee99464b30b6375a3daf45d56b99b1b64438ee8` | previous screen; see receipt queue | local queue 3890111 |
| 16 | cand_tc_k256_qkv.py | `0036aeba8960ce0ea0585968ae726a4464339f5db4eea16f3233bde7a8093120` | previous screen; see receipt queue | local queue 3890174 |
| 17 | cand_tc_k256.py | `accd2952ae08baf952e9cced1c2418e94094a1af3ccbb919479d8077f2ae7779` | previous screen; see receipt queue | local queue 3890190 |
| 18 | cand_tc.py | `c0553a0bb0b060f3f2b74b8d42b938e7ad154a748250d1c43ad9f865e7504cf3` | previous screen; see receipt queue | local queue 3890144 |
| 19 | cand_tile.py | `4f8b1aa3aa9974b203b85ba68189f3c57ccf2c890f31fb9a789f6b486aa20cb7` | previous screen; see receipt queue | local queue 3890132 |
| 20 | cand_tuned2.py | `40648053bc6269f2010db7f38d2b92e458498faa94334b31482613c4a525883b` | previous screen; see receipt queue | local queue 3890185 |
| 21 | cand_wg2.py | `b02135be25cb7918d76252387df63b2bb30e0070a1d2d7006727bf535196fcd7` | previous screen; see receipt queue | local queue 3890192 |
| 22 | cand_wg.py | `c224d1a5bae0fb7a0164d06e82d185c2271e6c4778b5a70c7d34d6562f6fbe8e` | previous screen; see receipt queue | local queue 3890194 |

Completed/covered sources are excluded from the send list:

| Source | Frozen SHA256 | Evidence / queue |
|---|---|---|
| cand_fused_lean.py | `2d42d4e56886d905a0e278a25ed1fef0578bbc2c924230c9d93c930d37f50f69` | local queue 3916757; same GPU bodies/configs measured by fused-select |
| cand_fused.py | `f454863e2ed72f9e6b380d2e4c86528fb3dfdc83fff73a1404ba348d312a4d48` | local queue 3916756; same GPU bodies/configs measured by fused-select |
| cand_fused_select.py | `10d02fa32b572b1649f5e0c63e5f9841567835dc3b83ffe17cd7d843429117e9` | completed H200 1.181998x, `f3786341611f4f1f` |

The lean/CPP fusion files have the same CUDA arithmetic and configuration lists as the
completed fused-select receipt. The orchestrator can retire those covered queued copies;
this thread did not cancel or modify their waiting processes. The other diagnostics still
add unmeasured configurations and remain listed.

All prior diagnostic sources remain frozen; their standalone H200 receipts can expose
a per-shape win even if the complete file loses overall. Do not list another source once
a completed receipt already measures its same GPU paths/configurations. No submissions
are launched here. `okc_run.py send` selects scored/check/reset handling for the orchestrator.

## Received evidence and remaining work

- Read **14** clean projection receipts through **`f3786341611f4f1f`**, scored **1.181998x**.
  `submission.py` is now its exact source, SHA256
  `10d02fa32b572b1649f5e0c63e5f9841567835dc3b83ffe17cd7d843429117e9`.
- Frozen `cand_portfolio.py` is completed `dd989ef2d76147e7` (1.172813x); do not resend it.
- Latest H200 fused router: m128 `(64,64,4,DSMEM,8,160)` **10.133 us / 1.637x**;
  m256 `(64,64,4,DSMEM,8,256)` **11.792 us / 1.509x**. Profiles confirm one kernel.
  Earlier router `(128,128,4,DSMEM,16/8)` **10.327/13.434 us** remains eligible.
  Qkv m128/m256 picked Lt ordinals 3/4; pooler 0/0; o m128 0. Unpooler and o m256
  retained cuBLAS. Qkv m256 picked a prepare-time winner but judged at **0.982x**:
  synthetic selection can regress; the broad portfolio pairs finalists on three fresh sets.
- FINDINGS owns all 30 SGLang/best shape times and receipt IDs. Best-ratio stitching is
  an optimistic **1.197544x**, not a new measured run. If only m128/m256 improve, each
  needs another **1.165x** speedup to reach 1.26x. Starting from the actual board needs
  **1.211x** each if only verify improves. Large BF16 projections remain the gap.
- The inherited and unmeasured fusion/layout sources remain pending/local queued as above.
- Latest fused-select picks: router configs above; qkv retains cuBLAS; pooler NN1/NN0;
  unpooler NN0/Lt0; o NN0/cuBLAS. The h200 portfolio retains their exact 32 MB choices.
  Read every receipt as it lands; fold new winners into a new source. Queued hashes stay frozen.
- The queued wg/wg2 sources never set `wg_ok`; they cannot measure their WGMMA kernels.
  The max superset activates those branches explicitly; portable and native ABI checks pass.
- The all-shape screen: `runs/screen/proj_portfolio_all/muse_bf16_proj.cand_portfolio_all.json`.
  Native/forced glue: `dev/portfolio_all_glue.log`; architecture build: `dev/portfolio_all_sm90a.log`.
- Reproduce: `CUDA_VISIBLE_DEVICES=0 tools/img python tools/local_eval.py problems/muse_bf16_proj/cand_portfolio_all.py --json runs/screen/proj_portfolio_all/muse_bf16_proj.cand_portfolio_all.json`.
- H200-incumbent evidence: `runs/screen/proj_portfolio_h200/`, `dev/portfolio_h200_glue.log`,
  `dev/portfolio_h200_sm90a.log`.
- Native evidence: `runs/screen/proj_portfolio_native/`, `dev/portfolio_native_glue.log`,
  `dev/portfolio_native_sm90a.log`, `dev/portfolio_native_dg_kernels_local.json`. The stock
  projection dispatcher was source-checked: it uses cuBLAS/its own kernels, not DeepGEMM;
  the documented DeepGEMM mode does not alter that baseline.
- Max evidence: `runs/screen/proj_portfolio_max/`, `dev/portfolio_max_glue.log`,
  `dev/portfolio_max_wide_native.log`, `dev/portfolio_max_sm90a.log`.
- Admission failures retained in FINDINGS: nvcc vector-path ICE fixed by owned C buffers;
  serial cold prepare 348.89 s fixed with 16-way nvcc/ptxas optimization. No variants dropped.

FOCUS stays at most 30 lines. FINDINGS owns results; this file owns ready hashes and queue state.
No cold-read agent test: this round is solo.
