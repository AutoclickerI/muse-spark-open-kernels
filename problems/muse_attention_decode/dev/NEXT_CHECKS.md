# Attention handoff / 09:00 KST order (2026-10-24)

Only the orchestrator sends. Do not edit or resend accepted sources.
No okc_run or multi-agent work was launched in this attention turn.

| Order | File | Frozen SHA256 | Action |
|---|---|---|---|
| 0 | `submission.py` | `610a590120ae79c446248cce5b6a857e283b425290aba67af29e18283c96c810` | Exact promoted check `32c5b4b287c24c38`,1.390910x >1.388x. Orchestrator may score. Verify is FA3 throughout; gain vs old check is within noise. |
| 1 | `cand_xqa_next.py` | `fee88fd0a7e9de09639228d67a5300ba760ddd0c946e998b33550466650933c8` | New packed asynchronous-producer MMA/WGMMA diagnostic. Check first among new attention work; full B16/B32 are first in prepare. Prints XQA_NEXT_DIAG/PAIR/AGREE. |
| collect | `cand_fa3_tune.py` | `bdddb8813ccecaebee3d13dfd18f0a215d6afb53cd537d39cc9ccdcc6b5eeefd` | Accepted check `dd927c6c46264afe`; read FA3_TUNE on arrival and promote only the exact passing source under owner's threshold. |
| existing | `cand_xqa_pipeline.py` | `b56b70c299139071298edb1bfac7ef3768c5ac8d73bce312822410184b01b43a` | Existing sender `runs/waiting/3126112.json` when inspected; no H200 receipt yet. Collect it; don't duplicate. |
| existing | `cand_verify_sweep.py` | `2f0205924f1b6ea5f16d27a9b0a5465b6374144987f01d00ad3344da69d6605e` | Prior screen sender; inspect queue/receipt before any action. |

New design: one KV head per CTA;64 query rows on N. K128 has eight consumer
warps doing E4M3 mma.sync m16n8k32 QK and FP16 PV; K64 has one consumer
warpgroup doing E4M3 wgmma m64n64k32 QK and FP16 wgmma PV. Both have128
separate producer threads gathering every page row with cp.async and full/empty
mbarriers. K128 stages3/4 hold48/64KiB raw KV; K64 stages5/7 hold40/56KiB.
Splits scale from the GPU SM count and1/2/4/8 scheduling waves; last CTA merges
FP32(max,sum,output), gates/norms, resets counters. Exactly2048 sliding tokens
and exact within-block causality. q_len=1 source is inherited unchanged.

The packed prepare uses synthetic random page tables and spread lengths only,
validates every variant's output against FA3 (<3% relative L2), then times16
variants+FA3 and selects custom only at >1.015 paired ratio. No data-derived
state survives a call. A Hopper variant failure raises; FA3 selection cannot
hide it. Untuned nearby shapes use the group kernel.

Admission for this exact candidate:
- `runs/screen/attention_next/muse_attention_decode.cand_xqa_next.json`:16/16
  tests and12/12 graph shapes,3 replays and2 fresh input sets, CUDA_VISIBLE_DEVICES=0.
- `dev/xqa_next_variants.log`:32 forced family/stage/split cases, five replays
  each, including1M context, window edges, page1/64, DP heads, scale1/64;
  eight FA3 tensor/tuple glue cases; all16 tuner choices/paired dispatch through
  portable stand-ins. Actual WGMMA is unexecuted locally.
- `dev/xqa_group_sm90_compile.log`, `xqa_warp_sm90_compile.log`:eight sm_90a
  specializations,zero spills. Group180–181registers; producer MMA153–156.
  SASS in `xqa_group_sm90_sass.log` contains QGMMA,HGMMA,LDGSTS.
- `dev/check_wgmma_layout.cpp` / `wgmma_layout.log`:4096coordinates per K/V/P
  layout agree with installed CUTLASS canonical atoms; descriptor fields checked.
- Local producer MMA full B16/B32 about355/676us vs synchronous339/660us;
  sliding B16/B32 about40/54us vs42/56us. Group MMA stand-in is slower locally
  (~585/1041us at full B16/B32 best split); this cannot rank Hopper WGMMA.

Landed evidence:
- `32c5b4b287c24c38`:1.390910x,CI[1.37005,1.41519],16/16H200 tests.
  All verify choices FA3. Best old K64 full B16/B32 675/1183us vs207/380us.
- `a0511c0001464dbe`:1.382808x,CI[1.36207,1.40730],16/16tests,all verify FA3.
  K128 paired full B16/B32 616.74/1173.81us vs207.36/391.36us;
  sliding B1/B16/B32 custom/FA3 ratios0.683/0.565/0.508.

Do not call the verify goal complete from an overall score. It remains open
until an actual H200 custom/tuned verify path beats FA3. No H200 ID exists for
cand_xqa_next yet. Sent diagnostic/K128/pipeline/FA3 files remain frozen.
