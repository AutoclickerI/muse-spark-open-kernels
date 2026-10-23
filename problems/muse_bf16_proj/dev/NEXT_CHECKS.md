# Projection next checks

`cand_portfolio.py` is frozen at SHA256
`57086918cce13065f97757c2471b7b79072e2bc45e8c589899723bf1d70d748e`.
The orchestrator sent it as **scored job `dd989ef2d76147e7`, priority 0**; do not resend or edit it.
Its pending job/receipt will appear under `runs/pending/` / `runs/muse_bf16_proj/`.

## Ready for the orchestrator

Next send: the pruned `cand_portfolio_v2.py` once the frozen portfolio receipt lands and
its exact-source screen passes. The builder refuses a mismatched or unclean receipt.
While that receipt is pending, the ready latency-first experiment is **`cand_fused_select.py`**.
It copies the lean fused kernels and permits the current incumbent to win router/qkv tuning.
The fusion-only diagnostic remains `cand_fused_lean.py`; do not send both separately.
Both keep the GPU math and replace heavy Torch C++ wrappers with a checked raw-pointer ABI.
Cold local prepare fell from 294.36 s to 60.21 s; warm retuning is about 3 s.
The lean port will also wrap any legacy H200 winners in v2.
Do not send the simpler diagnostics separately. The receipt-derived v2 will retain
measured H200 winners and compare only a few new fused/layout alternatives.
Build it with `python3 problems/muse_bf16_proj/dev/build_portfolio_v2.py <actual-receipt>`;
then screen it on GPU 0 and run `tools/img python problems/muse_bf16_proj/dev/validate_portfolio_v2.py problems/muse_bf16_proj/cand_portfolio_v2.py`.
The builder refuses to overwrite an existing candidate; use a new `--output` filename.

| Candidate | Frozen SHA256 | Local screen (9/9 tests, 30/30 graph shapes each) | Role |
|---|---|---|---|
| cand_fused_select.py | `10d02fa32b572b1649f5e0c63e5f9841567835dc3b83ffe17cd7d843429117e9` | 1.091522x | latency-first selector; ready until receipt-derived v2 replaces it |
| cand_fused_lean.py | `2d42d4e56886d905a0e278a25ed1fef0578bbc2c924230c9d93c930d37f50f69` | 1.093657x | compact 11-specialization forced fusion; diagnostic |
| cand_fused.py | `f454863e2ed72f9e6b380d2e4c86528fb3dfdc83fff73a1404ba348d312a4d48` | 1.089587x | compact fusion diagnostic; 294 s cold prepare, superseded by lean |
| cand_reduce_hier.py | `051b1eeb33f0bd1cc38a19ff8273eaad2a9254fba92529e8be869b464ebbef33` | 1.091486x | complete single-kernel fusion + hierarchical K32/K64 |
| cand_reduce_packed.py | `d19dcd2bc9c0d036297bc70474318aac6e465d83267d88dff9add5ca78d5211e` | 1.091603x | 128/160/256-thread pipeline diagnostic |
| cand_reduce.py | `280822ad5f91aae8e20e9440032f3f3ce22043119179375c4087059ee0c887f6` | 1.093592x | 256-thread fusion diagnostic |
| cand_reduce_fast.py | `b28e00958daef31379f8e3df4698e43be3ae05487c0e09c7d36204248132413e` | 1.078570x | 128-thread cooperative diagnostic |
| cand_lt_layout.py | `6c65aff44f84301c8eb9b772a1a2ac2ab3cece9429e44a2c6d0adb207d2c0381` | 1.088505x | lossless weight-transpose / cuBLASLt NN diagnostic |

All numbers above are sm_120, not H200. The fused kernels stage router hi+lo together
for each real-K tile and update the same FP32 accumulators. Fusion-only candidates always
select direct/last-CTA/DSMEM/hierarchical single-kernel paths for router/qkv m128/256.
Kineto confirms baseline router 5 kernels -> 1, qkv 2 -> 1, at both m values.
The latency-first selector may retain cuBLAS. Its latest local screen selects fused routers
(1.357/1.227x at m128/256) and cuBLAS qkv (1.008/1.000x); warm prepare is 1.22 s.

Admission evidence:
- Basic fusion: 770 kernel/replay cases; 1,075 forced Hopper glue cases.
- Packed fusion: 2,310 kernel/replay cases; 3,129 forced Hopper glue cases.
- Hierarchical fusion: 2,850 kernel/replay cases; 3,243 forced Hopper glue cases.
- All fused variants cross-build for sm_90a. Packed/hierarchical expose 66 specializations.
- Compact fused: 11 specializations, 210 pruned fresh-input graph cases. Lean port: 338
  native kernel/Lt cases and 110 forced Hopper glue cases; its selected graphs remain
  one kernel for router and qkv. The future legacy-winner raw ABI passes 180 native cases
  and cross-builds for sm_90a. No CUDA math was moved outside timed calls.
- Latency-first selector: 338 native kernel/Lt cases and 110 forced Hopper dispatch cases
  pass; `RAW_CUDA` is identical to the sm_90a-verified lean diagnostic. Screen and validation
  are in `runs/screen/proj_fused_select/` and `dev/fused_select_*.log`.
- Full v2 assembly passes 48 native fresh-input graph cases, 100 forced Hopper dispatch
  cases and an sm_90a build using explicit local control configurations. This is pipeline
  validation, not a fabricated H200 receipt or a claim about H200 variant winners.
- Screens: `runs/screen/proj_reduce{,_fast,_packed,_hier}/muse_bf16_proj.cand_*.json` and
  `runs/screen/proj_lt_layout/muse_bf16_proj.cand_lt_layout.json`.
- Detailed logs: `dev/reduce_*.log`; kernel counts: `dev/cand_reduce*_kernel_counts_local.json`.
- New methods are local-only until the orchestrator's H200 receipt measures them.

The builder and pruner are development tools; frozen files above must not be regenerated
in place after the orchestrator queues them. Screens retain their exact source hashes.

## Follow-through

- No okc_run launch or commit from this thread; the orchestrator sends passing sources.
- Read-only quota showed 11/20 checks used and 9 left; those 9 are allocated to the 9
  inherited queued checks. There are no unallocated checks for this round; the orchestrator
  chooses a scored submission, reorders its allocations, or waits for the 09:00 KST reset.
- Read the portfolio's selected configs and per-shape H200 table, prune losing variants into
  `cand_portfolio_v2.py`, then screen/freeze it. The measured H200 incumbent remains eligible if a fused experiment
  loses: the decision is latency/G3 first. Fusion-only sources separately verify the kernel-count
  reduction. Do not claim synthetic timings as judge timings.
- Event-driven receipt reader: `python3 problems/muse_bf16_proj/dev/receipt_digest.py --watch`.
  It reads receipts and pending markers; it does not submit, collect or poll.
- Reproduce fusion screen:
  `CUDA_VISIBLE_DEVICES=0 tools/img python tools/local_eval.py problems/muse_bf16_proj/cand_reduce.py --json runs/screen/proj_reduce/muse_bf16_proj.cand_reduce.json`.
  Extra checks: `tools/img python problems/muse_bf16_proj/dev/validate_reduce.py kernels` / `glue`,
  and `tools/img python problems/muse_bf16_proj/dev/profile_reduce.py` (GPU 0).

## Receipts and inherited queue

Read the original 11 completed projection receipts through `122997f357d04fa2`. The scored
`68e2f34b1d4744cb` stays at 1.142642x. GEMV checks `a3b29a26372c4b35` (1.112106x) and
`122997f357d04fa2` (1.114312x) lose overall; their pooler M32 cuBLAS branch beats the scored
skinny branch (0.991/0.996x versus 0.930x), now folded into the portfolio. Other same-algorithm
large-M differences and o M1's ~1% difference are inside noise, not per-shape discoveries.

The 13 inherited dispatch sources remain queued; no completed receipt has landed for them.
Do not resend or edit them: checks dsv3, dsv3_select, gemv3, gemv_static, m16, skinny, tc,
tc_k256, tc_k256_qkv; scored tile, tuned2, wg, wg2. P8 hardware probe `849bd4b650e54c7f`
landed clean at 1.136951x; its projection dispatch is old, and its diagnostic measurements
are summarized in FINDINGS. Their identities/snapshots live in `runs/waiting/` and
`runs/pending/`; fold new per-shape winners into a new frozen portfolio revision.
The wg/wg2 files never set `wg_ok`, so their queued scores do not exercise WGMMA.

Projection docs: STATUS.md was folded into FINDINGS and FOCUS.md (at most 30 lines).
The project-docs checker reports zero errors and zero warnings. Cold-read agent test was not
run because this round is solo. Skill revision: `5205a93`.
