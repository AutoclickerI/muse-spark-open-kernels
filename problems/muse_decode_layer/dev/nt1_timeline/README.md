# P3a small-M MoE: H200 decisions

Exact control: S14 `f1c77ce6`, flags `{1:95,8:69,16:69,32:69}`.
**PAIR is nominated to P4 at M8/16/32; no new whole-layer kernel is frozen.**
**The four-arm follow-up is measured: ILP also nominates M16/M32 versus S14.** It has not yet beaten
PAIR directly. FLIGHT, K16 and LUT lose; M8 ILP does not clear the stated uncertainty margin.
The three reset runs are complete and audited. `read_reset.py` reproduces the receipt/hash,
correctness, seed-count and A:A checks and writes `h200_results.json` and `h200_pipe.txt`.
Ratios below are candidate time / control time; smaller wins. A:A is mean absolute drift,
not the largest per-seed excursion. The receipts retain summary statistics, not raw per-seed timings.

## Reset decisions

| Run | Measured result at M8 / 16 / 32 | Controls | Decision |
|---|---|---|---|
| Epilogue `d650c04ead114119` | PAIR/S14 **.97493 / .97262 / .97932**, savings 1.787 / 2.085 / 2.292 us | 16 seeds, PAIR wins 16/16 at each M; A:A .415 / .235 / .225 us; EPBASE/S14 .99656 / .99979 / 1.00208 | Nominate PAIR. VEC loses 1.01332 / 1.01457 / 1.01898. PROD wins .98056 / .97831 / .98517 but PAIR is better. NORM .98142 / .98340 / .99000 adds no advantage over PAIR. ONCE recipes still lose 1.235–1.331. |
| Epilogue-next `29ad388de976432c` | PAIR/S14 **.97830 / .97462 / .97778** confirms first run. TREE/PAIR .99904 / .99811 / 1.00007 | 32 seeds; PAIR wins 31 / 31 / 32; A:A .442 / .363 / .322 us | Keep PAIR. TREE is tied. FOUR/PAIR 1.02215 / 1.02842 / 1.03432; LOCAL 1.01213 / 1.00951 / 1.00939; FOUR_LOCAL 1.03406 / 1.04155 / 1.04617. Close these exact additions to PAIR, not epilogue optimization generally. |
| Pipe `16c180784f414576` | Wrong-math PDIA/S14 .90404 / .89884 / .82615. Shared ILP2/W8 .98991 / .98145 / .98272 by node time | 16 routed seeds; S14 A:A .443 / .250 / .336 us. 4 seeds for 96 pipe / 66 work / 6 latency rows; microprobes have no independent A:A | PDIA is diagnostic only. A small independent-chain contrast supports testing ILP in the full kernel. Send the prepared four-arm batch; no rebuilt kernel is measured yet. |

All three official hosts pass, with no lab failure, truncation or incomplete seed count. Every correct
lab arm passes the native TASK gates before/after timing; maximum reported total/row error is below
.00247, against .015/.075 limits. PDIA deliberately computes wrong math and is never eligible.
M1's first-run PAIR/S14 is .99890 with .243 us A:A: neutral; retain original flags95.
Both PAIR runs clear `ratio + 1.96*SE + A:A/control_us < 1` at all three affected M values.
This is a stage nomination, not evidence of a six-load score gain.

## P4 handoff

Use existing admitted S16 arm **`epi_PAIR`**, restricted to the six shapes
`(batch,q_len,window) = (1,8,*) / (16,1,*) / (32,1,*)`. Leave M1 on its original flags95.
The S15 arm has normal/forced 13-test, 12-shape, 26-static-check receipts in
`dev/s16/admission.json`, exact SHA `af86073f101381a365affa7e66f0cebe4505d6b2fb45ea9fbfb38a56eb9bf601`.
S14 patch: `epilogue_s14_pair.patch` (`786a5ecb`), cell `epilogue_s14_pair.cell.py`.
`patch_s14.py --kind pair -o <candidate.py>` reproduces it against S14 `f1c77ce6`.
P4 still must run its many-seed whole-layer pair and official correctness gates before freezing.
TREE offers no measured incremental gain and is not preferred over the admitted PAIR implementation.

The original P4 parser rejected the entire epilogue batch because ONCE has no M1 row. During this audit
P4 independently repaired that handling. `read_reset.py` verifies the current real-receipt selector
chooses exactly `epi_PAIR` at the six shapes above; this lane did not edit P4's source.
`h200_results.json` records its source hash and exact selection.

## Four-arm follow-up: measured

`prof/lab/moe_nt1_pipe_rebuild.py`, exact **`caa59593`**, was sent through `h200_lab.py` on
`muse_moe_w4a16` at priority1.2. Sender **4124752**, merged wrapper **`de47736b9f588673`**,
judge **`dd539517bc8344dd`**, receipt
`runs/muse_moe_w4a16/20261026T010122Z_submit_dd539517bc8344dd.json`; metadata is in `followup_queue.json`.
It shares the run with P3b's already-waiting round2 payload. All four FLIGHT / ILP / K16 / LUT arms
remain packed together with exact S14, 32 seeds per M, ABBA read flush, A:A, before/after TASK gates,
consumer nodes and unique-byte rates. The payload bytes match the existing local screen.
The screened local output is3,014 characters, within the merged per-payload cap of4,500; the merge
preserved its complete controls and summaries. All32 H200 seeds complete and the official host passes,
without flags. Every arm passes native TASK gates at .00238/.00246 total/worst row. S14 A:A mean
absolute drift is **.487/.316/.314 us** at M8/16/32. `read_rebuild.py <receipt>` reproduces the
hash/gate/completeness audit and `rebuild_results.json`.

| Recipe | Ratio/S14 at M8 / 16 / 32 | Decision | Leaves open |
|---|---|---|---|
| FLIGHT | 1.07662 / 1.05285 / 1.03119;0/32 wins each | Close this bulk-L2-prefetch schedule | Other lookahead distances/issuing agents without competing with useful transfers |
| ILP | .99055 / **.99041 / .99046**;26/30/30 wins of32 | M16/M32 nominate to P4; M8 does not clear margin | M8's small gain, other chain/register schedules, PAIR combination and whole-layer gain |
| K16/r64 | 1.20394 / 1.21153 / 1.23270;0/32 wins each | Close this16-warp/shared-B-cache partition | Other K partitions/occupancy layouts without its cache and epilogue costs |
| LUT | 1.59959 / 1.62315 / 1.67379;0/32 wins each | Close this1KiB FP4-pair-table layout | Other lossless encodings and decoder reductions without its shared-service cost |

For ILP, `ratio + 1.96*SE + A:A/control_us` is **1.000762/.996981/.996081**.
Consumer nodes are67.083/72.868/104.667 us versus S14's67.652/73.791/105.861, consistent with
the small gain. ILP's128regs retain2 CTAs/SM; K16 also achieves2 CTAs with64regs/16-byte local stack.
These are valid measured recipe comparisons, not a proof of which physical port causes each loss.
M8 ILP is inconclusive for nomination, rather than proof of zero gain.

Exact P4 handoff: **`pipe_s14_ilp.patch`**, full SHA256
`b55c2970a7737c3708b0e5f251ab24b31ef1888bdae55f812c8e155d9eac230d`, onto S14 `f1c77ce6`.
Candidate `f71a0206`, cell `809bef8e`; `p4_ilp_handoff.json` pins full hashes and existing normal/forced
13-case/12-shape/25-static admission. Restrict the measured nomination to **M16/M32 sliding and full**;
the screened source has a general small-M driver, so applying it to every M would overstate the evidence.
Retain original M1 flags95. PAIR remains the stronger already-measured stage result at M8.

The trigger is the small shared-math ILP contrast at identical 2-CTA/16-active-warp residency.
Work-normalized ratios are .98973 / .98177 / .98275. No saturated operational pipe is established.
There is no microprobe A:A, so the contrast is a hypothesis requiring the full paired lab, not a
proven bottleneck or a predicted kernel gain. The other three arms travel in the already-prepared batch;
the diagnostic does not nominate them as fixes. More copy depth and 32 active warps do not improve
matched work. Cold/hot math differs, but depth and Little's-law contrasts do not isolate insufficient
flight. Integer/PRMT saturation is not demonstrated. See [PIPE.md](PIPE.md) for the measurements.

## Next comparison sent

`prof/lab/moe_nt1_ilp_vs_pair.py` **`d1e40f35`**, sender **479906**, wrapper **`af7b4526022c8d75`**,
on MoE at priority1.2, compares exact ILP directly with admitted PAIR plus S14 back.32 fresh
weight/router seeds at M8/16/32, ABBA read flush, before/after TASK gates, A:A mean/max and consumer nodes.
`ilp_vs_pair_queue.json` tracks its judge job; log `runs/detached/af7b4526022c8d75.log`.
The exact payload passed all32 seeds locally on GPU0; `ilp_vs_pair_validation.json` pins its SHA,
gates and1,725-character output. Native cells are byte-identical to their existing admitted screens.
`build_pair_comparison.py ILP -o <payload>` reproduces it. No new kernel or widened weight format is added.

Select from that receipt only with complete seeds, valid native gates and A:A controls. A stage winner
still needs P4's held-out six-load whole-layer pair. ILP+PAIR composition is unmeasured; their gains
must not be added by assumption. A loss closes only the exact tested recipe.

## Implementation evidence and reproduction

- [ISSUE.md](ISSUE.md): per-16-weight issue model, epilogue address/dependency accounting and measured decision.
- [PIPE.md](PIPE.md): operational pipe limits, matched phases, TMA latency and classification limits.
- `epi_next_validation.json`, `epi_next_screens.json`, `epi_next_lab_validation.json`: all four later
  epilogues passed528 component/replay checks plus6 forced ABI calls each, normal/forced whole-layer
  13 correctness cases,12 dispatch shapes and24 static checks; complete32-seed local payload passed.
- `pipe_ready_manifest.json`, `pipe_ready_screens.json`, `pipe_ready_validation.json`: exact candidate,
  cell and patch hashes; all four responses passed528 component checks plus6 ABI calls, and normal/forced
  13-case/12-shape/25-static screens. Worst TASK L2 .0024375.
- `pipe_rebuild_validation.json`: unchanged `caa59593` payload passed all32 local seeds and192 K16
  forced ABI calls in51.5s. It is pre-send validation, not a live queue record.
- K16's real two-stage Hopper kernel uses110,736 shared bytes atM32,64 registers,16-byte stack,
  zero spills and two CTAs/SM. Local sm120 uses an explicit one-stage native-math stand-in only for ABI
  coverage. No stand-in is shipped. LUT's independent proof covers all1,610,612,736 weights and scale codes
  (`pipe_lut_proof.json`); prepare retains packed codes, not widened weights.
- `pipe_lab_validation.json`, `pipe_native_audit.json`: final diagnostic passed16 local seeds,
  eight phase seeds per M, all96 probes/66 work/6 latency rows. D4 uses a local D2 ABI stand-in for36 calls;
  original D4 compiles sm90a. PDIA retains activation/weight loads with120 regs,zero spills,zero MMA/HMUL2.
- Local checks use GPU0 in the judge image through `tools/local_eval.py`; Hopper-only attention glue
  uses explicit stand-ins. H200 claims above come only from the listed judge receipts.
