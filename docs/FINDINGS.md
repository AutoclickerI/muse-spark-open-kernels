# Findings

Each row is one scoped quantity. Class: **measured** (on the judge, by our runs), **static** (counted
from code, config or spec), **model** (an estimate under stated assumptions; numbers from `python -m sim`, definitions in
[ceiling](reference/ceiling.md)), **official** (published by the organizers). Per-shape tables:
`python -m sim.calibrate`.

## Decode integration: saved-board dispatch audit

**Static + compiled:** board v9 `0e94268` always uses `moe_fused<1>` (8-token blocks, two stages,
256 threads, grid264 on H200) at M32 and M128; it has no MoE cost portfolio. best5 introduced NT1/NT2
selection; best6 `1b80ffc0` chose NT2/grid132 at M32 but **still NT1/grid264 at M128**. NT1's device-source
diff from v9 is stored-E4M3 conversion and two call sites; S1 now reinstates lossless recoding in a
separate NT1 bank. sm90a ptxas gives board NT1/best6 raw NT1/NT2 **124/121/194 registers**, no spills.
[Source/dispatch evidence](../problems/muse_decode_layer/dev/best7/DISPATCH.md).

**Measured + model:** v9 reruns are 1.428/1.422/1.432x; best6 is 1.380x. Reconstructing all four scores
from saved benchmark medians and 46/16 layer weights gives mean v9 per-load speedups
1.530/1.545/1.318/1.727/1.328/1.182 and best6 1.592/1.497/1.211/1.665/1.190/1.207.
The stronger ratio per load models **1.4415x**, not a candidate prediction. M128's gap is now
isolated by the paired scale-bank experiment below; it is not an NT choice change.
S1 fixes NT1 below M256, Humming at M256 and best6 decode1 attention; S2 replaces M256 with
CTA1g2112. Their hashes, admission and
run allocation are owned by [decode FOCUS](../problems/muse_decode_layer/FOCUS.md).

**Scale bank (H200, `58ae2e2cf8df45ef`):** v9's prepare-recoded scale bank (E4M3 scales re-encoded to uint8 exponent/mantissa, sign/zero folded into the FP4 codes: exact, so allowed by llms.txt "repack the weights and compress them losslessly") makes NT1 faster than best7's raw-E4M3 reader on identical routes at every M, 6/6 fixtures each, A:A noise <1 us, gate 0.0017: M8 92.0 -> 84.1 us (0.914), M16 99.4 -> 90.1 (0.908), M32 140.0 -> 125.5 (0.897), M128 322.2 -> 283.6 (0.880). This is the v9 vs best6-line gap. Any MoE reader that converts raw E4M3 scales per call should be measured against a recoded-bank variant.

**S1 integration (queued for 10-25 09:01 KST scored send):** `problems/muse_decode_layer/cand_r3_best7_scale.py`, SHA-256 `42f543a0e25f98058099e417ad2f8b95a71e1ab33c7371b3af6a4499905cbad7`, uses a separate recoded NT1 reader/bank for M1/8/16/32/128, keeps raw storage for NT2 and Humming's native M256 bank. Normal screen: 13/13 tests, 12/12 graphs, 18/18 static, prepare 56.7 s. Forced Hopper default: same gates, prepare 87.7 s. Forced table exercise: 34/34 gates (24 stream, DG and Humming), same 13/12/18 checks, prepare 45.4 s. The final normal tests-only rerun prepared in 100.5 s; scale-recode phases were 0.60/1.53/0.76/1.17 s, and every prepare was below the 300 s limit. M1 stream receipt `963515d3f0c04d91` has landed with no win (raw NT1 44.17 us; best P2g264 stream 57.07 us), so no M1 streaming path is promoted.

**S2 integration (local):** `cand_r3_best7_scale_w2fix.py` SHA-256 `b080544708544fd67ee23b82aca111bf0b9bb82c30a5eb86315044cd9bf92337` adds the exact raw-bank CTA1g2112 kernel/runtime from `97be11d4631b44d6` to S1, changing only the two M256 expert cells. Normal and forced screens each pass 13 tests/12 graphs/19 static, with both M256 layer types capturing CTA1g2112. Local prepare14.1 s normal/39.6 s forced; an initial cold-attention build took82.3 s. Raw-plane repacks were0.33/0.32/0.45 s. The integration gate reaches L2/row0.00380/0.00426. Three independent FP32 fixtures and six changed-route graph replays gate the new expert in prepare; failed admission raises instead of silently selecting another engine. H200 standalone385.2 vs Humming447.6 us is measured; whole-layer composition and cold H200 prepare are unmeasured. [Provenance and reproduction](../problems/muse_decode_layer/dev/s2/README.md).

**M128 fused producer recode (H200 pending, `471f6d6216804dda`, send `def9bff54e82c1b0`):** the repaired packed-WGMMA producer now has an exact v9-format scale reader: sign/zero folded into E2M1 in prepare; four byte permutations replace scale lookup loads, and the per-CTA E4M3 conversion/table disappears. The paired MoE-host lab compares raw/recoded cfg3/4/5 at grid528 with actual board NT1-recoded, on12 public-router weight seeds x3 identical fresh fixtures, six cold128MiB graph orders and NT1 A:A samples. Local36 fixtures/216 producer graphs pass independent FP32 gates (L2<=0.004132,row<=0.004851), all4064 finite scale/code combinations are exact; wrapper5 tests/6 graph shapes/6 static checks pass. All12 native specializations have no spills/C7513, and20 K64 paths keep disjoint register-A banks. H200 ranking and any best7 promotion remain receipt-gated. [Frozen sources and evidence](../problems/muse_moe_w4a16/dev/mixed_wgmma/m128/scale_bank/manifest.json).

**M128 instruction limit (SASS measured; H200 pending `dc3fea0a20a349bf`, send `a1762556e26931ef`):** actual sm90a NT1-recoded K512 reader: 492 instructions/256 weights per lane, cut to476 by one explicit LOP3 blend (17 ->16 unpack instructions per16 weights). Its aggregate four-warp issue estimate is9.99TB/s before route rereads, so the count does not establish aggregate issue as NT1's sole binding limit. On36 local public-router fixtures, NT8/16/32/128 decode rereads are2.176/1.478/1.157/1.000 per touched expert; NT1's logical972.8MB reads would cost243.2us at4TB/s without L2 tile reuse, versus97.4us ideal aggregate issue. These are models on this lab's fixtures, not timings of the earlier full-layer283.6us draw. F128's N32 W13 static K loop is84.75 instructions/16 weights (unpack17,scale16,MMA2,load6.5,address37,control6.25): address work is44%. A reversible u16 exponent/mantissa scale word, single LOP3 and wider copies cut it to81.25; fixed D32/D16 loops take65/59 and reuse the free lane's distributed W2 finalizer. Word metadata costs one extra byte/block (+11.1% bank bytes), with no expert-weight expansion. Static loop counts include conditional paths and imply model estimates, not measured hardware ceilings. New MoE-host lab compares all paths on12 weight/router seeds x3 identical real routes, six cold128MiB orders/A:A, plus separate W13/W2 graph phases and NT1 per-CTA globaltimer distributions; automatically batched with the small-M instruction lab. Local252 graph checks +42 signed/zero/subnormal/route-transition checks,4064 exact values,17 native entries without spills/C7513 and16 safe K64 paths pass; wrapper5 tests/6 shapes/6 static checks,26.4s within78s. Prior471f6d62 had no receipt at this turn's one read and no phase timers in its payload. No H200 win or best7 dispatch change is claimed. [Full counts, limits and source hashes](../problems/muse_moe_w4a16/dev/mixed_wgmma/m128/issue_limit/EVIDENCE.md).


## V9 tile reuse and router-seed sensitivity

- **Current result and handoff:** `problems/muse_moe_w4a16/dev/v9_reuse/dropin.py` now contains the native V9 portfolio, with raw NVFP4/E4M3 storage, three-fixture independent FP32 agreement gate, and eight alternating cold graph pairs in prepare. It compares native NT2 with hot-first fused NT1/NT2, one/two weight stages and channel caps. Promotion also requires no paired sample slower than1.05x incumbent. Independent-tile register/shared reuse, separate-phase work queues and W2 split-K are excluded after losing. `moe_reuse_prepare(full_decode_weights)` / `moe_reuse_experts(x,scores,local_ids,state)` are unchanged. The complete-layer integration is `problems/muse_decode_layer/cand_r3_v9_reuse.py`. The owner sends candidates and commits; this lane sent labs only.
- **Local matched experiment:** twelve weight seeds5353+7919*i, each passed to public `reference.generate_weights`, with `weights_fields` called using104729+100003*i. The published benchmark cases have no router_load targets, so fields={} for every draw; no hidden target or seed was inferred. Inputs use actual public `generate_input → ref_front → FP64 route`, including attention/residual/norm and busiest EP8 range. Every variant within a seed sees identical weights, activations and IDs. Timings are graph medians, eight rotated/reversed orders,128MiB write before each replay. GPU0 sm120 absolute microseconds are not H200 claims. Every seed and router byte hash is in `dev/v9_reuse/seed_sweep.json`; complete tables/commands are in `dev/v9_reuse/REPORT.md`.

| Variant | M128 mean / max us | M128 max/min | M256 mean / max us | M256 max/min | Class |
|---|---:|---:|---:|---:|---|
| V9 NT1 |376.5 /441.4|1.258|560.7 /627.2|1.282|local,12 seeds|
| V9 NT2 |375.9 /460.8|1.327|514.8 /568.4|1.274|local,12 seeds|
| V9 NT4 |503.5 /582.7|1.398|731.1 /838.4|1.380|local,12 seeds|
| Hot-first NT1,2 stages |373.4 /440.4|1.291|547.4 /609.3|1.291|local,12 seeds|
| Hot-first NT2,1 stage |359.0 /424.0|1.294|520.9 /579.6|1.306|local,12 seeds|
| Hot-first NT2,2 stages |372.5 /457.7|1.371|522.8 /580.1|1.292|local,12 seeds|

- **Decision and unmet target:** hot-first NT2/one stage lowers the M128 mean4.65% versus NT1; native NT2 lowers M256 by8.18%. Max/mean remains1.181/1.104 for those choices: **the requested worst seed within5% of the mean is not achieved**. A cap which merely slows all draws into a narrow band was rejected. Results do not justify making the hot expert's total cost independent of row count. Scheduling splits its channels into CTA work proportional to row blocks, but the arithmetic and unique expert bank traffic also change across seeds.
- **Work versus tail evidence:** M128 has171–228 local pairs and15–21 touched experts; M256 has316–467 pairs and16–26 experts. In a matched exploratory twelve-seed sweep, M128 NT1 time correlates0.959 with distinct-expert count and0.126 with hottest-expert rows; M256 correlations are0.563/0.774. These correlations do not establish causality. The measured seed sensitivity in decode receipts remains real, but assigning all of it to a hot-expert tail is not established. Compare non-MoE stages or kernels on matched weights/routes; full-layer board scores across different routers cannot isolate a kernel change.
- **Device scheduling mechanism:** the fused variant sorts expert ranks by current counts inside CUDA, remaps only scheduling metadata, preserves original expert weight/GS identity, and recomputes its active CTA count from actual row-block work each call. Physical launch capacity is fixed for CUDA graphs; excess CTAs return before entering the grid barrier. Channel chunk caps16/32/64 and hot-first/non-hot dynamic queue controls were measured. The shared-memory and register-resident grid limit is respected. W2 split-K2/4/8/16 stores disjoint FP32 partial outputs, then reduces before norm; it passes local arithmetic but does not beat fused NT2's twelve-seed mean. Source/summaries remain experimental, outside dispatch. The free lane retains CUTLASS/TileLang ownership.
- **H200 V1 closed:** receipt `a029bd95b80a461f`, payloadSHA6fef39b39321b17d,52 timing rows, all FP32 gates pass. R0=exact26d7e05 fixture0 IDs: M128209pairs/18experts/hot55; M256414/21/119. Fastest scalar-load experimental row r16g2 is1742.7/2833.4us, while same-run V9 is~333/~577us. Full-K shared BF16 reuse passes H200 arithmetic but costs2741.0/4526.4us. **V9 wins**; the payload's experimental-only WIN lines omitted the incumbent and are not dispatch winners. Complete table is REPORT.md. R0 keeps the exact exported IDs but refreshes activations/route weights; its original router seed/fields were never exported. The G track is a separately seed-reproduced public generator, not a claim to reconstruct R0's original weight matrix.
- **H200 seed lab:** `c30223b03cc34829` (payload `prof/lab/moe_v9_seed.py`, 12 public seeds + R0, all FP32 gates pass). Means over 12 seeds, us: M128 NT1 341.5, NT2 339.8, hotNT1 334.7, NT4 439.7; M256 NT1 600.3, NT2 544.1 (-9.4%), hotNT2s2 549.3, NT4 644.5. Time follows routed pairs, not touched experts: M256 NT1 runs 485 us at 342 pairs and 801 us at 653 (~1.0 us per pair, intercept ~140 us) across 14-25 experts, so V9 pays per 8-row block (each block re-reads and re-dequantizes its expert's tile) against a ~106/124 us unique-weight floor. NT4 loses on the H200 too (+29%/+7%). Final places are medians of the organizers' reruns (CONSTRAINTS 2), so variants are compared by the seed mean.
- **V9 NT3 row-block probe:** `dev/v9_reuse/nt3_dropin.py` adds a legal 24-row hot-first specialization (no expanded weight bank) and `cand_r3_v9_nt3.py` ports it into the decode layer. On sm120, the 12 public seed means are 370.7/507.6 us at M128/M256 versus the native controls 377.7/515.8; the result is a routing-tuner lead, not an H200 claim. Independent validation covers 144 captured configurations and 576 fresh route/input replays, max relative L2 `0.001333`. sm90a ptxas uses 125/219/255/255 registers for NT1/2/3/4; NT3 has an 8-byte frame (20/12 spill stores/loads), while NT4 has a 288-byte frame (500/344), so the larger-row limit is register pressure/spilling rather than an MMA-shape change. The H200 lab is queued as `711370966a58ac2b`; no receipt yet.
- **Raw64 WGMMA decode probe:** `dev/v9_wgmma/raw64_dropin.py` specializes raw17's register-A path from `m64n128k16` to `m64n64k16` and changes the route tile/gather/combine width to64. W13 SwiGLU remains fused before W2, and both E2M1 codes and E4M3 bytes remain packed until in-kernel conversion. The local portable SM120 screen passes all five challenge tests; the adapter screen passes156 fresh graph gates (raw64 whole-output L2 about0.0048, max row L2 below0.0056). SM90a ptxas reports zero stack/spills for raw64's four MODE/STAGES instantiations. The comparison lab included NT1/NT2/raw17/raw64/masked-Humming/DG on exact R0 plus12 seeds; receipt `63c980e800194764` passed arithmetic but raw64 measured534/679 us at M=128/256 and is closed.
- **Raw64 H200 closure and Humming follow-up:** receipt `63c980e800194764` passes all FP32 gates but gives raw64 seed means `534/679 us` at M=128/256, versus V9 NT2 `342/547` and masked Humming `342/450`; raw64 is closed for decode. Humming's two grouped-masked GEMMs account for nearly all of its time, so profile payload `prof/lab/moe_humming_profile.py` (source key `e858855cac1b5ad0`) records default/block-M/SM variants, W13/activation/W2/gather/combine spans and unique-bank throughput. Variant payload `prof/lab/moe_humming_variants.py` (source key `053c794fc0aba59a`) compares masked, masked-TMA and indexed plans on exact R0 plus12 seeds. Both are queued; no new H200 win is claimed.
- **Precision and validation:** no prepared BF16/FP8 weight bank; stored E4M3 bytes undergo layout repacking only and are converted inside CUDA (fast normal path, exact signed/subnormal/zero cases). Global scales stay FP32. Independent gates dequantize scratch oracle weights only inside prepare and do not return/cache them. Complete-layer13/13 tests,12/12 scored graph cases and7 static checks pass. The standalone component is tested separately to catch missing globals that the layer might supply. Twelve-seed independent FP32 checks cover120 captures/480 fresh routing/activation replays including sparse/single-pair/all-remote transitions (worst relative L2=0.0013331). Four scored verify layer cases also pass with hot-first one/two-stage forced,20 replays/3 fresh inputs. Current drop-inSHA4e6fff436bbfaf34e1620b97206e516760d6d8e1a17244ceeb1819a6935aeda8; complete-layerSHA590b8026aa7a00da3644aa11392cbc3b972764cc234354a78c4c796fc14c971d. Serving's earlier raw-scale migration is unchanged and remains locally verified through16384 rows; this turn does not promote the new portfolio into serving.
- **Humming at decode shapes (H200):** lab `de6a56fc38e9477d` (payload `prof/lab/moe_decode_adapters.py`, R0 + 12 public seeds, distinct W13 gate/up global scales, all rows FP32 PASS, nothing crashed). Seed means, us: M=128 NT1 343.6, NT2 340.9, humming_masked 341.4, raw17 647.0, dg_w8 1047.8; M=256 NT1 604.1, NT2 545.5, **humming_masked 448.0 (-17.9% vs NT2; wins all 13 fixtures)**, raw17 771.1, dg_w8 1259.7. So 26d7e05's Humming mismatch was that lab's adapter; the prefill lane's adapter is correct on decode weights. raw17 and dg_w8 lose at decode M.


## Grouped NVFP4 RS WGMMA (free lane)

- **H200 result, M256:** `e8968e36ea98422a moe_fused_m256` passes all FP32 gates. Fastest fused12-seed mean is F7=1201.8us versus masked Humming448.5; unfused P128503.2, P256585.3. R0 W13 takes266–426us but W2+norm+combine1376–2644us. Main configurations allow3–5 native CTAs/SM with zero spills, so occupancy alone does not explain that phase. Older scalar/plane losses and pipeline R0 wins are retained in the receipts and “MoE real-routing labs.” No candidate/serving producer switched by free lane.
- **W2 mechanism isolated locally:** `check_tail.py` uses exact R0 IDs (414pairs/21experts/hot119), fresh Y and independent FP32 output gates. Old complete tail costs510/950us at NT16/32; metadata hoisting lowers it to334/595; norm+completion tickets without combine takes9/12us (timing-only), while a vector CTA per token takes14.3/14.3us with agreement. The old tail runs on just40/28 group CTAs, serializes several complete4096-channel tokens per warp, and revisits metadata inside its channel loop. This isolates combine, rather than global atomics/fences, as the large local tail cost. H200 attribution awaits the phase-clock probe.
- **Repair built:** `dev/mixed_wgmma/w2fix.cu` uses GPU route/reset, W13+SwiGLU, W2+weighted Y/tile sum-of-squares, then256threads/token for norm+vector combine. Metadata is hoisted; float4 Y loads and packed BF16 stores distribute the work across256CTAs. Corrected W2 and finalizer contain no global atomics or publication fences. The extra launch replaces the last-warp tail. N16/32, K128/four-stage cp.async/wait1 and grids528/1056/2112 are fixed contenders. W2 reads144KiB Q+SF/CTA; R0 requested bytes are377.5/264.2MB at N16/32 versus198.2MB unique. Throughput labels requested bytes, not measured DRAM.
- **Precision and compiler boundaries:** E2M1/raw E4M3 stay packed until timed conversion; FP32 expert GS applies in epilogues, including distinct gate/up values. No expanded NVFP4 bank in prepare. The previous B128 alignment repair now passes H200 agreement: old local absolute-address emulation reproduces1.416 L2, aligned0.0004215; current B patterns and dynamic bases align1024. Non-unrolled K64 pairs avoid the earlier ptxas C7513/per-MMA wait0 issue. CUTLASS numeric GEMM remains quarantined; no further slot or omni integration.
- **Local admission:** repaired raw-pointer graph glue passes78 independent changing-route/extreme-scale checks (L2<=0.004225,row<=0.004546), including Hopper-address emulation and local_start64. All five actual sm90a GEMMs plus route/finalizer audit with zero local-memory instructions; GEMMs retain wait1/wait1/finalwait0. `native_abi.py` removes PyTorch headers from CUDA and the pybind unit; Python resolves the current CUDA stream at each launch. This addresses cold compile overhead as well as respecting the payload's per-compile/per-seed deadline checks. The92s cold-build prototype that breached78s was not sent. Lean cold build14.5s; complete12-seed/nine-variant/R0 lab36.0s within78s locally.
- **M256 lab protocol:** `prof/lab/moe_w2fix_m256.py` compares six fixed repairs against masked Humming/P16/P32 on12public weight/router seeds,3fresh fixtures each and8rotated/reversed cold128MiB graphs. Gates remain finite,L2<=1%,row<=5%; partial seeds never qualify. R0 diagnostics additionally gate full output and print phase times, native occupancy,144KiB/CTA requests and main/YSS/group-ticket/norm/token-ticket/combine cycles. MoE-host send key`602e2630309eb17c` is queued, batched with M128 K64 repair; immediate detached log says waiting for a slot. Final host5/5,static6/6,directstatic5/5; all12seed/nine-variant and3R0 gates pass locally; report2429chars. New H200 timing and the448.5us seed-mean target remain unverified. Current hashes, budget evidence and queue identity: [admission.json](../problems/muse_moe_w4a16/dev/mixed_wgmma/admission.json); [source/protocol](../problems/muse_moe_w4a16/dev/mixed_wgmma/README.md).

- **Lossless recoded M256 variant built:** owner-approved joint recoding folds E4M3 sign/zero into packed FP4 and stores exponent/mantissa bytes. `recode/recoded.py` applies this directly to the existing bitplanes, then timed PRMT/shift/add constructs the BF16 multipliers. Raw and recoded share the same four-launch fixed-W2 producer and six N/grid choices. SM90a recoded bodies remove16 shared-load sites per K128 stage, with zero spills and wait1/wait1/finalwait0 intact. This is lossless weight representation, with no prepared expanded weight/scale bank.
- **Recoded lab admitted and accepted:** `moe_recoded_m256` job`9e8a6a15f95748db`, send key`fdac893179c9f8ad`, MoE host. Main12public seeds use the actual shared-W13-GS contract so stock NT1-recoded is an exact control; each of14fixed variants sees the same3fresh real-router fixtures and8rotated/reversed cold128MiB graph replays. Independent gate/up GS remains a component stress. Exact65024 encoding cases and111 graph/scale checks pass (L2<=0.004132,row<=0.004714), including Hopper-address emulation, allocator churn and nonzero expert offset. Host5/5,static6/6,directstatic3/3; all12seed/14variant plus4R0 gates pass locally. Cache-disabled fresh build3.5s/full lab15.6s within78s; expired-deadline guards pass; report2934chars fits a batched share. New H200 means against masked448.5us and a best7 cell remain pending. [Protocol and hashes](../problems/muse_moe_w4a16/dev/mixed_wgmma/recode/admission.json).

## MoE real-routing labs

Full kernel × M × routing tables, phase probes, errors, and source receipts: [MoE H200 report](../prof/lab/moe_real_routing_h200.md). R is the decode-style fixture; B is the standalone MoE board's EP8 routing. M = 8/16/32/128/256/512/753.

- **Measured, loaded router and experts:** `26d7e05ab97d4f96` produced 252 timings in 174.9 s (prepare 211.7 s). In that matrix, the valid core leader is V9: R 73.8/76.9/113.4/279.0/478.5 us at M8/16/32/128/256. V19/phase minima differ by less than 1%; this batch keeps V9 instead of duplicating a persistent engine for those differences. Marlin wins M512/753 at 563.3/715.9 us versus V9's 930.5/1331.6 us. Grouped and all 12 TMA/WGMMA choices lose.
- **Correctness rejection:** V11 and original Humming fail every loaded-weight M/routing, typically ~28% relative L2 and 40–65% per-row. Generated-bank run `c8c89f9f111845af` passed them with constant per-expert GS (M256 V11=401.5 us); that result does not establish loaded-weight correctness. Local decode-weight tests pass 30 refreshed Humming-vs-Marlin graphs, and all seven extra variants pass 210 SM120 graphs with varied FP32 GS. The new H200 evidence isolates the passing family to H7, which forces `MmaType.MMA` and 16-row alignment. Default Hopper WGMMA variants fail; H4 tile changes and H8 Stream-K-off do not repair them. This points to the WGMMA/scale path rather than generic GS conversion or Stream-K alone; the exact defect is unresolved. Upstream [issue31](https://github.com/vllm-project/humming/issues/31)/[PR52](https://github.com/vllm-project/humming/pull/52) describe a narrower MXFP4A8/H20 Stream-K rounding issue; it is a diagnostic lead, not our established cause.
- **Captured-route H200 result:** `907004cbcc7c4b82` completes all 140 core cases in 210.4 s (prepare 220.0 s):60 timings and 80 agreement rejections, with no budget omissions. R uses loaded job 26d7e05 fixture0 IDs and three refreshed activations; weights use independent log-normal FP32 GS (seed555). H7 alone passes among H2/H4/H5/H6/H6k/H7/H8 plus original Humming/V11; H3 aliases V11. Core R minima are V9 63.6 / V9_phase75.0 / V9_phase104.5 / V9 278.7 / H7 462.8 us. Same-run V9 at M256 is 476.4 us, so H7 gains 2.9%; on B it loses 465.7 vs 448.2 us. The dispatch targets R. Native differences <1% coalesce to V9 for this batch. No 200/120 us core result was found.
- **Dispatch artifact:** `prof/lab/build_moe_lab_dispatch.py` and the report share `select_dispatch`. They choose V9 at M8/16/32/128, H7 at M256 from captured-route job 907004cbcc7c4b82, and Marlin at M512/753 from loaded-weight job 26d7e05ab97d4f96. `cand_r3_moe_lab.py` uses nearest-M, shape-only selection; native calls add no GPU conversion/launch. Library calls map negative local IDs to remote ID 255. Validation: 13/13 complete-layer tests, 12/12 scored-shape graphs, 6 static checks,and 21 fresh graphs across M192/193/256/383/385/512/753 against FP32 expert math. Complete-layer H200, H7 phases/prefill, and serving remain unverified; H7 was timed with varied synthetic expert scales, not the original loaded bank. This lane sends lab payloads only; no candidate submission or commit.
- **Router and fixture correction:** proj supplies an iid FP32 router at std `1/sqrt(8192)`, without decode's bias/norms; decode uses `.045` noise, `.07` mean alignment, `.055` bias. One decode-host lab slot was therefore necessary. `moe_tune.make_fixture` now applies draft sharing to rows 1–7 of every M8/128/256 verify block, matching `reference.generate_input`; M16/32 retain q_len 1. Other shapes accept explicit `q_len`. Older embedded copies need rebuilding. The compressed export preserves all 3,520 core route slots for further proj-host runs.
- **Measurement protocol:** six alternating paired graph repeats over three refreshed fixtures, 128 MiB L2 flush, median paired ratio anchored by V9's median. Values include the roughly 4 us graph cost. Phase-only graphs start with cold L2, so their sums need not equal E2E. `CY` clock64 counters in the reference supplement are single instrumented eager launches without an explicit prelaunch flush; they diagnose per-CTA phase balance, not paired cold-graph wall time. The generated-model `c8c89f9f111845af` has 280 rows, but predates corrected verify sharing; it is retained as a separately labeled router sample. Its V11 phase AttributeError was repaired; E2E rows are unaffected.
- **Library references:** `897a1606858d444a` produced 28 timings in 61.6 s (prepare 69.4 s), using the generated router and corrected sharing. Local-only 32-group BF16 `torch._grouped_mm` takes R 723.8/934.3 us at M128/256; FlashInfer MXFP4/no-postnorm takes 368.4/559.2 us. Our BF16 reference folds FP32 GS into BF16 weights; it is not a lossless NVFP4 dispatch bank. FlashInfer's E8M0/group32 scales cannot encode arbitrary E4M3/group16 scales losslessly. Earlier `bf16ref_remote` rows compute an additional zero-weight remote expert; the local-only reference supersedes them. SM120's CPU-assisted grouped fallback blocks graph capture ([image's PyTorch source](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/cuda/GroupedBlas.cpp#L748)); H200 capture succeeds. Neighboring `f8e977320c3c4556` finds essentially identical hot-subset times with cuBLASLt preference on/off (M128/3 experts288.064/287.952 us; M256/8 experts538.560/539.312 us); those subsets are not mixed into our full-path table.
- **Inadmissible hybrid evidence only:** neighboring `b37b93deb1b74f3a` measures M256 DeepGEMM psum plus V9 cold experts at337.2us versus its own V9 control396.1us (threshold16, SM share0.25, rel L2~0.37%); M128 retains V9. That generated-router fixture has341 local M256 pairs. Its BF16-resident design in `dev/bf16moe/` violates rule5 and cannot be promoted; it is not a row in this packed-NVFP4 sweep or evidence for our loaded-router sample.
- **Local payload checks:** main/supplement host 9/9 tests and 168 forced launch/phase paths; extra Humming adds35 forced Hopper graph paths and 210 real SM120 arithmetic comparisons. Main/supplement static checks 12/12; extra-family checks 26/26. The current supplement includes shared phase/V19 helper repairs and an explicit graph-closure bank binding. The immutable reference receipt predates those glue/lint repairs; its measured reference functions are unchanged.
- **Bound, conditional on the current packed bank:** loaded R fixture 0 touches 18/21 experts at M128/256, with 209/414 pairs. One bank read per touched expert is 509.6/594.5 MB, or106/124 us at 4.8 TB/s; work is 21.0/41.7 GFLOP. These fixture 0 estimates do not bound a median over three different fixtures. The earlier~70 us estimate used8 touched experts and does not apply to every router sample.
- **V19 static hazards:** only native `permute=0` with a resident grid was measured. Optional `grid2` doubles the occupancy-limited grid while a global barrier waits for every CTA, so it can deadlock. `(u*17+3)%total` repeats/drops work when total is divisible by 17: only 512 of 8,704 tasks are distinct at total 512×17. Source: `problems/muse_moe_w4a16/dev/parked/cand_v19.py`; proof: `runs/screen/moe_v19_static_audit.json`. This is a static proof, not an observed H200 hang.
- **Image inventory:** Marlin and Humming expose BF16×NVFP4 on H200; vLLM/Machete are absent. FlashInfer BF16/FP4 GEMMs and Prims-TS target SM100+; its Hopper MonoMoE has FP8 weights and fixed K2048/N512/M≤8. SGLang Triton W4A16 decodes affine integer4; its ROCm Muse file is an aligner shim. DeepGEMM FP4 APIs take FP8/FP4 activations, and its BF16 grouped APIs take BF16 weights. No additional eligible BF16×NVFP4 H200 path was found. Inventory: `runs/screen/moe_image_kernel_files.txt`.

- **V9 variants at M=128/256, 12 seeds (`3a312224908a416e`):** at M=128 every variant is within 2% of the others (NT1 344.1, NT2 341.7, hotNT1 336.4, hotNT3s2 337.1 us mean; NT4 448.6). At M=256 hotNT3s2 is best at 520.9 (NT2 548.8, NT1 605.9), still behind humming_masked's 448 (de6a56fc). Seed spread is 1.5x max/min (180-306 local pairs at M=128), so a single run can't separate variants. The V9 family plateaus near 3x the ~108 us M=128 floor; the lever is a different mainloop, not another V9 tile.

- **Recoded scales on the fused M=256 path (`9e8a6a15f95748db`, 12 seeds, 8 rotated cold graphs):** C1g2112 (exact exponent/mantissa scale recode, FP4 sign/zero fold) takes 365.2 us mean (346.4 median) against the same path on raw scales 386.1 (C:R 0.946), masked Humming 448.2 and NT1-recoded 500.4; gates pass. The recode gains 2.4-6.2% across grids, most at grid 2112.
- **Fused M=256 path beats masked Humming (`97be11d4631b44d6`, 12 public seeds, 3 inputs, 8 rotated cold graphs, raw bank):** with the W2 combine moved from one serialized last warp to a vector CTA per token, CTA1g2112 takes 385.2 us mean (366.5 median) and CTA1g1056 391.0, against masked Humming 447.6 (438.9) and P32 396.7; gates pass (L2 0.0041). At R0 the fixed W2 phase drops 866.4 -> 161.6 us (0.44 -> 2.34 TB/s requested) and W13 runs 301 us at 2.51 TB/s. The old serialized combine (CTA0) is 463-492 us. Saving ~62 us x 62 layers is ~10% of the verify 32 load. The same run's M=128 K64 bank-rotation repair is correct but 1.7-1.9 ms (HOT128 path), not a candidate.
- **M=128 fused WGMMA producer is 5.7-6.5x NT1-recoded (`471f6d6216804dda`, 12 seeds x 3 fixtures, cold, K64 repair kept):** configs 3/4/5 take 1,650-1,862 us (raw) and 1,650-1,851 (recoded; rec:raw 0.988-0.999) against NT1-recoded 291.7 us; gates pass. Its final W2/combine phase is the serialized one the M=256 lane replaced (466 -> 162 us at M=256 above); this path is not a candidate until that fix is ported.
- **Humming at M=128/256 is bandwidth-starved (`896866402dc94112`, R0 + 12 seeds, eager event spans):** W13 streams the touched experts' unique NVFP4 bytes (16 MiB W13 + 8 MiB W2 per expert; 11-25 experts, 264-600 MiB) at 0.70-1.18 TB/s and W2 at 0.63-1.07 TB/s (M=128 R0: W13 261, W2 146 us for 432 MiB, whose read at 4 TB/s is ~113 us). Humming's own tiles (block [16|24,256,128], 4 stages, TMA, warp-specialized) leave ~4x of HBM bandwidth unused, so the M=128/256 lever is bytes in flight per SM, the same limit as the dense GEMVs (b63389f0). Its SM-count and block16 launch variants all failed the FP32 gate (0.37 relative error), a lab defect rather than a finding.

- **V9 variants at M=1-32, seed means (`b68afb72b72542b4`, 7-8 public seeds each; the AGG lines were truncated):** NT1 is fastest at M=1/8/16 (49.0/84.4/94.1 us), hotNT1 at M=32 (136.8 vs NT1 140.8). NT2 costs +16% at M=1 and +28% at M=8, and NT4 +63-96%, so best6's NT1 stays below M=128. At M=1, NT1 spends 42-59 us on 2-4 local pairs, 2.5-4x one read of those experts (24 MiB each). The same run's real-routing payload died on a lab bug (`KeyError: (128, 'R')`); fixed, it ran in `8c73606710d94cde`: with the loaded decode router (R), V9 takes 42.0/65.4/76.3/105.5 us at M=1/8/16/32 (board routing B: 42.8/81.6/114.2/147.6) and every grouped/tile alternative that ran loses 2.5-4x; the rest ran out of time. With the scale encoding fixed (`b741f04aaf53479d`, banks exact), NT4 passes but loses to NT1: M=16 149.2 vs 93.7 us, M=32 141.4 vs 110.8.

- **M=1 streaming expert GEMV loses (`963515d3f0c04d91`, 12 real-route seeds, raw bank, cold graphs):** persistent W13 rows + W2 split-K with fused norm/sum, 14 grid/stage/PDL variants: best P2g264 57.1 us and S2g264 58.1 against NT1 44.2 and v9m1 44.6 (2-3 local experts; one-read floor 11.8-17.7 us). PDL and grid 264-528 barely matter; 8 stages and grid 132 are much worse. NT1 itself runs at 1.4-1.7 TB/s of unique bytes. Together with the scale-bank result (exact recode = 9-12% faster NT1), per-weight decode work, not bytes in flight alone, is a binding limit at small M.
- **Small-M scale-contract failure resolved on H200 (`b741f04aaf53479d`):** the stale lab fed prepare-recoded E4M3 bytes into NT4/v9m1's raw reader (scale128: stored `0x70` became `0xe0`, interpreted as -32). That pairing reproduces locally at1.862656 relative error. The corrected lab checks192 inverse bank tiles and passes every H200 SwiGLU/W2/norm/output stage. NT1 takes47.44/89.49/93.68/110.75us at M1/8/16/32. M1 `newNT1` is `v9_small_nt1_kernel_fn` through the small NT4 module, hence the v9m1 body:48.00us versus47.44, ratio1.0118. It does not approach one read. NT4 loses at M16/M32 (149.18/141.39us); no promotion. Candidate keeps NT1 at M1/8/16 and hotNT1 at M32, with NT4/v9m1 explicit lab choices. `prof/lab/build_moe_small_debug.py` regenerates the raw bank/reader contract from the current candidate. Earlier failure receipts are3920d3f0/948044d4; this cause is distinct from the free lane's B128 addressing repair. Local logs: `runs/screen/moe_small_{debug,lab_entry,static}.*`.
- **M1 scalar streamer closed on H200 (`963515d3f0c04d91`):** every valid stream recipe loses on12 real-router draws. Best P2g264 is57.07us versus NT1-raw44.17us and v9m144.61us (1.292x slower); one-read floors vary11.8–17.7us. PDL and grids264–528 barely change the outcome; split8 regresses to95us. All gates pass (~0.24% FP32 error). No dispatch promotion. The two-launch kernel's source and original720 native/160 forced local checks remain in `problems/muse_decode_layer/dev/small_m_gemv/`; its completed receipt supersedes the old pending state.


- **Deeper packed WGMMA pipeline, exact R0 routing (`7156d8d8b56a40d1`; seed means truncated):** the best whole-MoE config P5 (NT32, K128, 4 stages, overlap) takes 313.4 us at M=128 (NT1 331.3, masked Humming 348.0) and 415.5 at M=256 (masked 437.0, NT1 576.0), ~5% ahead. Its W13 alone peaks at 195 us for M=128 (P1, grid 1056: 1.74 TB/s of unique bytes) and 261.6 at M=256 (P5: 1.51 TB/s), still ~2.5x one read at 4 TB/s. Humming's masked-TMA variant is ~3% faster than masked (337.6/427.4 vs 350.9/437.6 us); its indexed variant fails the gate (0.37). The run's prepare reached 295 of 300 s because the first payload overran its time share.

- **M=256 12-seed means and the fused path (`e8968e36ea98422a`):** masked Humming 448.5 us stays best (P128 503.2, NT2 548.1, VFIX32 556.4, P256 585.3, NT1 606.7). The free lane's fused route/W13+SwiGLU/W2+norm+combine path (F0-F12) takes 1,138-3,159 us, with its W2 phase alone 1.38-2.64 ms at R0 (W13 266-426 us); implementation suspect. Humming at M=128 (R0 only): block-M 32 328.0 us vs default 355.3, TMA 341.3; 66/88 SMs 629/489; 2 stages is invalid (WGMMA needs 3). The M=128 one-read path on the same fused producer (`66133279d19145ad`, 12 seeds) has the same slow final phase: best A0 827.0 us vs NT1 344.3, NT2 340.5, masked 341.0; W2+final 557-1,369 us vs W13+SwiGLU 240-293; A4-A6 and A8 return non-finite output.


- **M128 K64 register lifetime defect (local proof; H200 repair pending):** cfg3/cfg5 in `66133279` reuse af[0] while its WGMMA group remains live after wait1. Native SASS confirms this [PTX lifetime violation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions-wgmma-wait-group). Shared `fused.cu` now rotates two banks across paired K64 stages; W2/split-K epilogues are byte-identical. All10 K64 paths go from63 early overwrites per64-stage loop to zero;393,216 B128 addresses match the writer and remain in bounds. Local234 M128+210 shared gates and12 public seeds plus R0/HOT128 pass (M128 L2<=0.004132,row<=0.004948), with no native spills/C7513. The immediate SM120 reader cannot reproduce asynchronous UB, so the new receipt must confirm removal of the non-finite gates. Repair lab send **0ea82c48f2255889**, pointer-only sm90 build/load12.5s; its detached log says waiting for a slot with no rejection. Evidence: `problems/muse_moe_w4a16/dev/mixed_wgmma/m128/{REPAIR.md,repair_admission.json}`.

- **Small-M recoded NT1 limit audit; new H200 lab queued:** unchanged sm90a NT1 uses124 registers with no spills, normally two36KiB stages/grid264. Per16-weight group, its production decode/scale path has17 integer extraction instructions +8 BF16 multiplies +2 amortized scale-building instructions =27 (isolated probe29 before scale amortization). Line-mapped SASS is in `runs/screen/nt1_decode/`; `prof/lab/audit_moe_nt1.py` reproduces it. Historical recoded H200 phase receiptac9dd461 at1980MHz gives W13/W2 CTA medians M1=8.91/9.08us, M8=53.97/30.57, M16=73.35/36.30, M32=82.20/40.64; these board-routing phases overlap, W2 includes wait/norm, and they are not new real-route cold timings. Confirmed small-M loss so far: the raw-scale conversion tax9–12% (58ae2e2c); the remaining primary limit awaits the new phase/ablation data. Valid variants: single-LOP3 mixed-word select (26 instructions/group;125 registers), lossless16-bit integer scale encoding (26/group, +11.1% bank bytes;128 registers), and earlier stage refill with half the activation cache moved to shared memory (128 registers). All eight native families compile sm90a without spills;432 native graphs/288 forced Hopper calls pass on3 public weight seeds, max L2=0.002416; full local twelve-seed entry39.22s passes. Changed early-stage2 and standard stage3 glue use supported-depth arithmetic stand-ins on SM120. Ddecode/Dmma are explicitly ineligible, non-equivalent lower bounds: SASS shows Dmma drops the unused decoder too, so it measures streaming/control rather than isolated MMA. MoE-host send **0ea5aeba5a3435bf**, immediate log waiting for a slot with no rejection; judge acceptance unverified. Source, drop-in and hashes: `problems/muse_decode_layer/dev/nt1_limit/{README.md,admission.json}`. Best7 is unchanged pending a valid H200 win.

## The judge's H200 (measured, 2026-10-23)

| Quantity | Value | Scope | Class | Receipt |
|---|---|---|---|---|
| Judge GPU state | H200, SM 1980 MHz, memory 3201 MHz, 700 W limit, ~130 W idle draw | node n-qtqz4qekgtq37epqb5xaarah6h | measured | `runs/muse_bf16_proj/*7fb3e93056d34999.json` (`machine`) |
| Graph replay fixed cost the judge subtracts | ~4.2 us (raw median minus reported median) | bf16_proj shapes | measured | same, `replay_overhead`, `median_raw_ns` |
| Historical HBM/L2 sweep | 52-61 MB stream: 12.8-14.6 us = 4.04-4.25 TB/s; best used 132 CTAs x 1024 threads and source unroll 4 x 16 B/thread. Warm 4-16 MB: 8.2-9.6 TB/s; 32 MB: 4.5-5.1 TB/s | CUDA events with the sweep's baseline subtraction, different loads/cache hints and CTA sizes from P8 below; outstanding requests were not counted. The 32 MB drop is specific to this protocol | measured | `problems/muse_bf16_proj/profiles/cand_sweep2.62bd26b45aeb4992.log` |
| V9 MoE time depends on the run's router seed | The same `moe_fused<1>` kernel took 258.9 vs 304.6 us at verify 16 sliding and 449.5 vs 546.6 us at verify 32 full in 0e94268 (1.428x) and 7b2165c8 (cand_r3_best, 1.351x), while SGLang's baseline moved 1-2% (537.7/540.9, 1088.8/1113.1 us). prepare() gets a per-run seed, so routing and the hottest expert's load change per run; V9's tail (one CTA set finishing the hot expert) carries the variance. Decode scores swing a few percent per run from this alone: compare candidates by non-MoE stage times or on the same routing, and balance the hot expert (split its N, hot-first scheduling) to make V9 both faster and steadier | receipts' per-kernel profiles | measured | `problems/muse_decode_layer/profiles/cand_r3_best.7b2165c875bd48de.json`, `cand_r3_v9.0e94268cf5654012.json` |
| best5 vs the board run, per stage | cand_r3_best5 `691bff0bcff44e99` 1.353x vs 0e94268 1.428x: MoE at verify 16/32 ran at the 12-seed mean (341/527-530 us vs 259-280/450-477), attention at decode loads improved (decode 1 full 33.9->18.9, decode 32 full 228.5->197.8 us), and verify-32 full FA3 regressed 377.9->501.5 us: prepare picked num_splits=32, pack_gqa on a synthetic input whose KV lengths don't match the scored 'agentic spread', while SGLang's baseline on that load moved -1.6% | decode_layer | measured | `problems/muse_decode_layer/profiles/cand_r3_best5.691bff0bcff44e99.{txt,log}` |
| best5_glue vs best5, per stage | `7370b18221964412` 1.367x vs `691bff0b` 1.353x: the three PDL glue launches left glue kernel time and total-minus-MoE-minus-attention within -6.8..+2.6 us per case (noise); the score moved with MoE routing (verify 32 527/530 -> 503/490 us) and FA3 at verify-32 full (509 -> 380 us, same tuner code). Single scored runs can't resolve few-us changes; paired labs can | decode_layer | measured | `problems/muse_decode_layer/profiles/cand_r3_best5_glue.7370b182*.txt` |
| Paired decode lab: glue, FA3, experts (H200, 3 public seeds, AB/BA x6, cold) | `9c83cdb852fe4576`, 36/36 fixtures, 0 rejects, 0 flips. Glue: best5_glue's new geometry saves 3-4.6% of the whole layer at decode 1 and verify 1 (ratios 0.954-0.973; A:A noise <1.3 us), is neutral at 16/32 (0.99-1.01) and costs 0.6-1.3% at decode 32 sliding. FA3 at q8: the public 'layer' scheduler is as fast as every split setting (within ~2-4%) and much faster than sp16t/sp8n at verify 1 full (1.65x/2.63x). Experts: NT2 loses to NT1 at every M (1.09-1.34x); at M=128 Humming loses 6-13% (346-353 vs 317-320 us) and raw17/dg_w8 1.6-3.8x; at M=256 Humming wins 19-25% (417-426 vs 537-547 us) | best7_paired locks the measured new geometry except old glue at decode32 sliding, public layer scheduling at q8, NT1 below M256 and Humming M256; bank-recode E:C remains unresolved | decode_layer | measured | `runs/muse_decode_layer/*9c83cdb852fe4576*` |
| v9 vs the best5/best6 line, 3 seeds | The board entry v9 (`0e94268c`) reran as the paired labs' host: 1.428/1.422/1.432x on three router seeds (`7307be78`, `1365af66`). best6 1.380x and best5_glue 1.367x are below it. Per load, v9 mean vs best6: decode 1 1.526/1.606, decode 16 1.553/1.472, decode 32 1.321/1.203, verify 1 1.725/1.670, verify 16 1.316/1.226, verify 32 1.164/1.263. The gaps are MoE: decode 32 sliding moe_fused<1> 101.7 (v9) vs moe_fused<2> 137.6 us (best6); verify 16 sliding 258.9 vs 315.7 us (layer 387-393 us on three seeds vs 423/432). best6 wins at decode 1 full (attention) and verify 32 (Humming). Per-case best of both models 1.455x. v9's 4th seed (`9c83cdb8`) 1.434x. best7_reset (`c0e74bed`, v9's NT1 dispatch to M=128 on best6's raw-scale bank, Humming at 256, best6 attention) checked 1.402x: decode 1 1.624, decode 32 1.336, but verify 16 1.198 (sliding 441.8 us vs v9 387-393) and verify 1 1.627, so the M=128 gap is not the dispatch | decode_layer | measured | `runs/muse_decode_layer/*0e94268c*`, `*7307be78*`, `*1365af66*`, `*1b80ffc0*` |
| best6 vs best5_glue, per stage | `1b80ffc037cc454c` 1.380x: prepare picked humming_masked at M=256 (V9 NT1 at M=128); verify 32 MoE (W13 262 + W2 146 us + swiglu/combine/gather 25 us) vs V9 503/490 us, layer 648->583 (sliding) and 986->884 us (full); FA3 at verify-32 full 376 us. Other loads moved with the router seed | decode_layer | measured | `problems/muse_decode_layer/profiles/cand_r3_best6.1b80ffc0*.{txt,log}` |
| Lab host on a withdrawn run | `tools/h200_lab.py` picked the host as the best scored run, which for the decode layer was the withdrawn, inadmissible bf16moe `07d07d960b814b86` (1.461x): the batched GEMV+overlap lab `b63389f01e4741b9` runs on it (a check, not on the board; its host-function timings are not the incumbent's). Host selection now skips withdrawn runs (decode host: 0e94268) | tooling | measured | `tools/h200_lab.py` best_source |
| Streaming GEMV vs a plain read (H200) | `feb7b287e3464e4a`, cold 256 MiB flush, 20 pairs: a plain one-launch read of each weight takes qkv 18.6 us (37.75 MB, 2.03 TB/s), o 11.3, pool+router 30.8, unpool slice 8.7, unpool full 28.4 us (0.96-2.45 TB/s; ~2.4-3.1 TB/s after a 5.2 us launch/ramp envelope). The streaming GEMV matches that read at M=1 (qkv 18.3, o 12.3, pool+router 29.6, unpool slice 9.3 us; 0.77-1.01x cuBLAS) and loses to cuBLAS at every M>=8 (1.1-1.7x at M=8-32, 2.3-4.9x at M=128-256). Drafter shapes lose too (1.12-1.80x) | Matched same-byte reads show the measured launch/ramp envelope; overlap is the next practical M1 lever, not a proven hardware ceiling. Best6 uses PDL but its small-M custom recipes explicitly disable own-weight prefetch | measured | `problems/muse_prefill_layer/lab/profiles/lab_streaming_gemv_lab.feb7b287e3464e4a.log` |
| Persistent dense chain vs best7 (H200) | `87dc944bfd7944c7`, AB/BA, cold 256 MiB, changed inputs: best7's o -> boundary -> pool+router takes 38.3/39.2/39.4/39.8/43.6/56.3 us at M=1/8/16/32/128/256; the persistent chain 57.7/62.7/65.3/71.2/198.9/365.6 (1.51-6.49x). In the whole best7 layer the chain costs +21 us at M=1 (82.7 -> 104.0) and +24 at M=8. Errors 4-8e-5, no route flips. (The earlier 57.3 vs 76.1 us 'win', `9d223519`, was against torch.mm + an FP32 router, not best7.) | This chain implementation loses at every M; best7's segment moves 88 MB in 38 us (2.3 TB/s), so a persistent design still has room, but not this one | measured | `problems/muse_decode_layer/lab/profiles/lab_dense_chain_best7_compare.87dc944bfd7944c7.log` |
| Small-M native glue fusion vs best7_scale | New CUDA pool prologue + last-CTA route, unpooler + boundary2; alternative o + boundary1 epilogue avoids recomputing RMS across pool CTAs. Frozen c0e74bed projection/attention calls with current best7_scale glue/recoded NT1; synthetic non-expert seed941732 and real experts, all eight decode1/16/32 + verify1 S/F cases. Three changed-input graph draws, AB/BA6 cold256MiB, A:A; independent FP32 route gates. | Local native72 checks and forced Hopper96 whole-layer comparisons pass; max layer L2 `0.004515`, no route flips, host5 tests/6 static pass, SM90a8 kernels compile without spills. All local cells lose, so the table stays empty. H200 accepted, pending. | `dev/fused_glue/{README.md,admission.json}`, `prof/lab/best7_fused_glue.py`; send **e4a1f1e29a7bf810**, job **80b70f7f20fa41c6** |
| Side-stream L2 prefetch inside best6's layer (H200) | `d5a9025890a64141`, paired AB/BA, 3 seeds, decode 1/16/32 sliding+full: the best recipe slows the layer by 0.3-16.6% (decode 1 sliding 123.1 -> 143.5 us). The prefetch branch costs ~34 us and overlaps attention at most (9.5-18.3 us), never MoE, because those kernels hold every SM. With its weights in L2, o drops 8.2 -> 6.0 us and pool+router 26.4 -> 25.2 | A separate prefetch launch can't run beside full-grid kernels; any prefetch must be issued from inside the running kernels, and the measured ceiling is a few us per layer | measured | `problems/muse_prefill_layer/lab/profiles/lab_l2_overlap_lab.d5a9025890a64141.log` |
| Verify dense split-K/stream-K WGMMA (H200) | `19f3a13b17b44b6f`, synthetic BF16 weights of the real shapes, cold 256 MiB, 27-32 configs per role: best7's cuBLAS/Lt takes qkv/o/pool+router/unpooler 21.7/11.8/28.1/23.7 us at M=128 (85.3 total) and 24.1/13.8/33.9/28.6 at M=256 (100.4); our weight-wide TMA/WGMMA's best per role is 1.34-2.06x slower (137.4 / 159.9 total, 0.76-2.12 TB/s), and every split-K>1 setting lost to split 1 | This implementation loses at every role; cuBLAS itself reaches only ~1.7-2.4 TB/s here (~3x the floor), so the verify-dense gap stays open | measured | `problems/muse_moe_w4a16/lab/profiles/lab_verify_streaming_synthetic_moe.19f3a13b17b44b6f.log` |
| In-kernel PDL own-weight prefetch (H200) | `7426e17cd3bc4659`, best7_paired layer, synthetic non-expert weights, cold 128 MiB, 24 AB/BA pairs: decode 1 sliding whole layer 115.78 -> 115.58 us (0.9986); per role qkv 1.0017, o 1.0094, pool+router 0.9951, unpooler 0.9910. The other seven cases stopped on the lab's own whole-layer gate (0.0031-0.0033 against a 0.003 bound on synthetic weights), so they are unmeasured | With the L2-warm oracle above, own-weight prefetch is worth <=1% per role at M=1; the dense lever is the GEMVs' fixed cost | measured | `problems/muse_moe_w4a16/lab/profiles/lab_dense_pdl_prefetch.7426e17cd3bc4659.log` |
| L2-warm oracle for the dense stages (H200) | `09d19813094842b1`, best6 layer, each projection's weights read into L2 before the timed graph, cold 128 MiB flush otherwise, decode 1/16/32 sliding+full: activity time cold -> warm: qkv (36 MiB) 15.0-20.6 -> 9.8-14.9 us (5-6 us saved), o (16 MiB) 6.8-9.3 -> 6.7-8.2, pool+router (72 MiB) 25.5-29.2 -> 24.9-29.6 (0-2.3), unpooler 0-1.3. Whole-layer ratios 0.93-0.97 for qkv, 0.95-1.0 for the rest (A:A spread ~3 us). Warm qkv still takes 9.8-14.9 us (2.4-3.7 TB/s from L2) and warm o 6.7-8.2 us | Prefetch can only help stages after the first; their warm gain is <=~2-4 us per layer. qkv, the one stage that gains, opens the layer, so nothing in the layer can prefetch it. Even L2-resident, the dense kernels are bound by per-launch fixed cost and issue, not HBM: the dense lever is the GEMV kernels' own efficiency (the proj board field is ~30% faster), not overlap | measured | `problems/muse_prefill_layer/lab/profiles/lab_lab_l2_oracle_ceiling.09d19813094842b1.log` |
| Fused glue GEMVs and verify-load WGMMA (H200) | `5535d2f171b34d72`: fusing norms/residual/routing into the streaming GEMV costs 2.3-21x cuBLAS at every M (o at M=1 69 vs 17.8 us; 0.03-0.7 TB/s). The weight-wide WGMMA for the verify loads loses 1.28-2.18x to best6's own Lt/cuBLAS choices: dense total M=8 102.3 vs 64.9 us, M=128 128.0 vs 84.4, M=256 161.3 vs 99.6. best6 already streams the dense weights at 2.0-2.3 TB/s (138-197 MB), near the ~2.5 TB/s one launch reaches at these sizes | Standalone dense kernels are at their per-launch limit; the dense lever at every load is overlap across launches, not a new mainloop | measured | `problems/muse_prefill_layer/lab/profiles/lab_streaming_glue_lab+verify_streaming_lab.5535d2f171b34d72.log` |
| Standalone projection GEMVs on the H200 | Cold-L2 graph replays, decode-layer shapes (`b63389f01e4741b9`): qkv 37.75 MB in 20.7-21.9 us (1.73-1.82 TB/s), o 16.78 MB in 11.9-13.9 us (1.2-1.4), pool+router 75.5 MB in 32.3-34.4 us (2.2-2.3), unpooler 67.1 MB in 27.6 us (2.4), unpooler slice 8.4 MB in 11.2 us (0.75), at M=1/8/16 alike. cuBLAS, every Lt heuristic, our GEMV and a 4x-CTA split all land within a few percent, while a plain 52-61 MB read stream reaches 4.04-4.25 TB/s. The initial comparison used a different read size/protocol; the matched-read feb7b287 result above supersedes its unused-bandwidth inference. The same run's overlap payload hit a sticky `cudaErrorInvalidAddressSpace` (on the withdrawn bf16moe host) and failed the run | decode-layer projections | measured | `tools/h200_lab.py read b63389f01e4741b9 gemv_standalone_lab` |
| Decode projection side-stream L2 prefetch | `l2_overlap_lab.py`, payload `e62b6b087adeb0c5`, prefill send key **9367398c624b7796**. Thirteen completed-load/bulk-hint recipes, bounded40MiB front/24MiB tail, captured forks/joins during attention/o/MoE. Exact best6 arithmetic and receipt dispatch at M1/16/32 sliding/full; missing historical M1 glue geometry retuned by its original selector. Cold128MiB per replay, three refreshed public streams,24 finalist AB/BA rounds, A/A noise and native activity profiles. | Local: all78 coarse recipe/cases,12 finalists,6 profiles,5 host tests/3 static checks pass; cached payload12.0s of80s. FP32 raw/router1e-4 plus independent attention/V9/norm gates; final mixed-BF16 A/B0.3%, exact IDs. Native sm90a128-bit loads and UBLKPF.L2, zero spills. Frozen best7 transport test08bbc928:13 official/12 graphs/16 static,6 real prefetch instances. Switch off pending paired H200 whole-layer gain; no prepared expert expansion or cache persistence. | local admission; H200 pending | `problems/muse_bf16_proj/dev/l2_overlap/{PROFILE_MAP.md,README.md,validation.json,best7_admission.json}` |
| Deterministic persistent dense chain lab | `lab_persistent_dense_chain.py`, payload SHA `3e32bde3c605bbe5`, detached/send key **86a149e7b115d57d**, queued on `muse_prefill_layer` at priority0.6. One grid-resident BF16 `o -> boundary -> pool+router` chain uses a phase-numbered grid barrier; each CTA prefetches the dependent pool/router rows before that barrier. Cold-flush A/B and three changed-input graph replays cover M1/8/128/256; chain stops before routing to isolate the boundary. | Portable SM120 screen: hidden/router relative L2 ~1e-4, pooler ~0.2-0.3%; chain/base ratios 0.72/0.67/1.60/2.02 at M1/8/128/256 (local only). H200 receipt is required before any table enable; no prepared dequantized bank. | local admission; H200 pending | `problems/muse_prefill_layer/lab/lab_persistent_dense_chain.py`, `runs/detached/86a149e7b115d57d.log` |
| Deterministic persistent dense chain vs best7 incumbent | Replacement payload `dense_chain_best7_compare.py`, SHA `dd6f0b41c891390ba78ad136de72ca23995416ab31e47b9a82a0d098f528a432`, detached/send key **5279d3e3903a57b5** (prefill quota reset wait). It embeds frozen `best7_reset` SHA `b4cc76dc`; A calls best7's own `project(o) -> boundary -> project(pool_router)`, B calls the phase-numbered chain. Six alternating cold-flush graph replays cover dense M1/8/16/32/128/256; full best7 v9nt1 graphs cover decode1/verify1 with changed-input checks. | Local sm120: chain/base stage ratios 1.04/1.01/1.00/1.06/3.14/3.89; full-layer ratios 1.01 (decode1) and 1.01 (verify1), with hidden relative L2 <=0.0013 and zero route flips. The send is queued until the prefill reset; all `BEST7_DENSE_CHAIN` cells remain false pending H200 whole-layer evidence. | local admission; H200 pending | `prof/lab/dense_chain_best7_compare.py`, `runs/detached/5279d3e3903a57b5.log` |
| Our own WGMMA MoE kernels at decode M | 12 seeds + R0, H200: raw64 (raw17-derived m64n64k16, MoE lane) 534/679 us and vector TileLang (free lane) 471-501/660-763 us and REJECTED by the FP32 gate, against V9 NT2 342/547 and humming_masked 342/450 at M=128/256 (`63c980e800194764`, `4ee35d3dd5374c40`). Humming's 450 us at M=256 reproduces | moe_w4a16 at decode shapes | measured | `tools/h200_lab.py read 63c980e800194764`, `4ee35d3dd5374c40` |
| Judge queue | Our runs start one at a time, a median 399 s apart, each running a median 125 s (95 runs to 2026-10-24 03:40). The spacing doesn't depend on our previous run's duration (correlation 0.06, slope 0.07 over 94 pairs), so the judge gives us runs, not minutes: a longer run (compile, prepare autotuning) costs no slot up to the 15-min cap, and packing variants into one run multiplies what each slot measures. Wait = queue position x ~6.6 min. No cancel exists (`okc` has withdraw only; a withdrawn queued run stays queued, 07d07d96). Under load the spacing stretches: 3 of our runs finished between 13:35 and 15:02 on 10-24 while `GET /api/v1/queue` showed a backlog of 1,709 jobs on 64 GPUs (bearer token, client user-agent), so read `/queue` before planning on ~6.6 min. | receipts' created_at/started_at/finished_at | measured | `runs/*/*.json` |
| Run-to-run noise | the starter's speedup CI was 0.892-0.920 (+-1.5%) | one check, 30 shapes | measured | same, `speedup_ci` |
| How a case is timed | Two eager `kernel_fn` calls (untimed: JIT, lazy tuning), then capture on a side stream, one untimed replay; each timed replay: fresh inputs copied into the graph's tensors, L2 flushed, ~1 us GPU spin inside the window, replay; fixed replay cost calibrated (graphs of one vs two tiny kernels) and subtracted. Launch on the current stream; extra streams must fork/join inside the capture | every kernel problem | source | `harness/worker.py` recovered in github.com/msaroufim/openkernelchallenge-audit (0a68efe), hash-matched to our receipts' `evaluator.files` |
| What prepare's state becomes | `prepare(weights)` runs once with this run's weights (seed and `fields` per run: the router, hence routing, changes run to run). Its tensors in plain dict/list/tuple are cloned into up to 16 copies behind random pads (decode 1-3, MoE 4-16, proj 10-16 in our receipts, fewer for big states) and cases rotate over them; other objects are shared and the originals stay alive, so cached pointers stay valid but address-dependent tricks don't hold | every kernel problem | source + receipts `records[setup].copies` | `harness/worker.py` recovered in github.com/msaroufim/openkernelchallenge-audit (0a68efe), hash-matched to our receipts' `evaluator.files` |
| Rejections and integrity | Server static checks reject source that reads harness frames (`inspect.stack`); a rebound CUDA event/graph-replay method fails the run; exit code 112 is any validation failure, compile errors included; `allow_tf32=False`; the profile tables come from one eager call, not the graph | every kernel problem | source + probes | `harness/worker.py` recovered in github.com/msaroufim/openkernelchallenge-audit (0a68efe), hash-matched to our receipts' `evaluator.files` |

## P8 hardware probes on H200 (2026-10-24 KST)

One check, [849bd4b650e54c7f](../runs/muse_bf16_proj/20261023T135730Z_check_849bd4b650e54c7f.json),
passed 9/9 projection tests and all 30 scored shapes; the copied projection achieved 1.137x.
Diagnostic prepare took 31.4 s; total judge execution was 128.3 s. The GPU reported 132 SMs,
1980 MHz SM / 3201 MHz memory before and after the run, on node `n-3tovvmp4azgxjrdbhkqxjjl254`.
These are endpoint clocks, not sampled probe clocks.

[Source](../problems/muse_bf16_proj/cand_probe_hw.py) and
[full labelled log](../problems/muse_bf16_proj/profiles/cand_probe_hw.849bd4b650e54c7f.log).
Times below are raw CUDA event intervals inside captured graphs, without baseline subtraction or
host tensor-map setup. Rates count requested useful bytes, not memory-controller counters. Each
entry is the median of three trials, except the graph-boundary experiment's five. Local screening
compiled both extensions for sm_90a with CUDA 13; sm_120 portable probes, Hopper Python glue with
an MMA stand-in, and all nine projection tests passed. Compute Sanitizer memcheck reported zero
errors on the portable run. SASS confirmed LDG groups of 1/2/4 loads before consumption and no
local/stack use in the inspected probe kernels; WGMMA N8/16/32 used 42/58/90 registers.
Local logs, SASS and both sm_90a build flags are preserved in `runs/screen/p8_hw/`.

### HBM concurrency

256 threads/CTA. LDG uses 16-byte `.cg` loads; cp.async uses 16-byte `.cg` copies, followed by shared
consumption; bulk uses independent 4 KiB copies and an mbarrier, then shared consumption. Each
trial reads a distinct 64 MiB region, truncated to complete grid iterations (60.6-67.0 MB).
Flight is the nominal request-group footprint averaged over 132 SMs, not measured outstanding bytes.
The bulk experiment is 1D `cp.async.bulk`; tensor-map TMA is exercised separately in WGMMA.

| Quantity | Value (TB/s in increasing U order) | Scope | Class | Receipt |
|---|---|---|---|---|
| LDG128, 66 CTAs | U1/2/4: 0.64 / 1.04 / 1.61 | nominal flight 2 / 4 / 8 KiB/SM | measured | `849bd4b650e54c7f`, HBM log |
| LDG128, 132 CTAs | 1.16 / 1.80 / 2.49 | 4 / 8 / 16 KiB/SM | measured | same job |
| LDG128, 264 CTAs | 1.95 / 2.61 / 3.01 | 8 / 16 / 32 KiB/SM; best 64,880,640 B / 21.57 us | measured | same job |
| cp.async, 66 CTAs | U1/2/4/8: 0.55 / 0.89 / 1.39 / 1.89 | 2 / 4 / 8 / 16 KiB/SM | measured | same job |
| cp.async, 132 CTAs | 1.01 / 1.57 / 2.17 / 2.64 | 4 / 8 / 16 / 32 KiB/SM | measured | same job |
| cp.async, 264 CTAs | 1.72 / 2.39 / 2.88 / 3.15 | 8 / 16 / 32 / 64 KiB/SM; best 60,555,264 B / 19.20 us | measured | same job |
| cp.async.bulk, 66 CTAs | U1/2/4/8: 0.44 / 0.76 / 1.24 / 1.77 | 2 / 4 / 8 / 16 KiB/SM | measured | same job |
| cp.async.bulk, 132 CTAs | 0.85 / 1.38 / 2.06 / 2.46 | 4 / 8 / 16 / 32 KiB/SM | measured | same job |
| cp.async.bulk, 264 CTAs | 1.46 / 2.24 / 2.79 / 2.88 | 8 / 16 / 32 / 64 KiB/SM; best 60,555,264 B / 21.06 us | measured | same job |

These kernels have not saturated the historical 4.25 TB/s result: this check's separate 528-CTA
weight stream itself reaches 3.80 TB/s. Different CTA sizes, cache instructions, scheduling, sinks
and timing protocols prevent treating 3.15 TB/s as a new H200 roof or a regression.

### Random page-1 KV gather and L2

Gather reads a random permutation of 1,048,576 unique 128-byte rows (128 MiB), one coalesced warp
per row, four independent rows per warp. It includes page-index loads and a checksum sink, without
FP8 conversion or attention math. Warmup precedes timing; the working set exceeds L2, but this is
not a separately flushed cold-cache bound for every attention workload.

| Quantity | Value | Scope | Class | Receipt |
|---|---|---|---|---|
| Page-1 gather, 132 / 528 / 1056 CTAs | 292.67 / 83.10 / 52.48 us; 0.459 / 1.615 / 2.558 TB/s; 3.583 / 12.62 / 19.98 billion rows/s | 128 threads/CTA, same random rows | measured | `849bd4b650e54c7f`, gather log |
| Warm L2, 4 MiB window | ordinary / evict_first / evict_last / evict_normal hint: 7.69 / 7.80 / 7.78 / 7.42 TB/s | 268,173,312 B requested; `.cg` bypasses L1; 528 CTAs, U4 | measured | same job, L2 log |
| Warm L2, 16 MiB window | 7.58 / 7.74 / 7.72 / 7.56 TB/s | same policy order and read count | measured | same job |
| Warm L2, 32 MiB window | 7.31 / 7.32 / 7.31 / 7.35 TB/s | same; no abrupt 32 MiB threshold in this protocol | measured | same job |
| APW reservation | requested 4 MiB; driver returned 7,864,320 B, previous 11,796,480 B | Previous reservation restored afterward; captured kernel APW verified locally | measured / local verification | same job, APW log |

Retention warms a 4 MiB activation, streams 129,761,280 B of weights, then rereads 25,952,256 B
from that activation. The printed "8x" target is truncated to complete grid iterations. Each case
resets persisting lines before configuring APW; APW uses hitRatio=1, persisting hits and streaming
misses. Weight stream and activation intervals are timed separately.

| Weight policy | APW off: weight us / activation us | APW on: weight us / activation us | Class | Receipt |
|---|---|---|---|---|
| ordinary | 34.85 / 8.96 (3.72 / 2.90 TB/s) | 35.10 / 8.51 (3.70 / 3.05 TB/s) | measured | `849bd4b650e54c7f`, retention log |
| evict_first | 34.11 / 8.42 (3.80 / 3.08 TB/s) | 34.98 / 8.45 (3.71 / 3.07 TB/s) | measured | same job |
| evict_last | 35.68 / 9.06 (3.64 / 2.87 TB/s) | 34.82 / 9.15 (3.73 / 2.84 TB/s) | measured | same job |

### Graph boundary, BF16 tensor math and NVFP4 decode

| Quantity | Value | Scope | Class | Receipt |
|---|---|---|---|---|
| Dependent graph pair, normal vs PDL | producer 9.73 us; consumer 46.88 us; normal pair 59.74 us; PDL 58.69 us; saving 1.06 us (1.8%); normal pair minus isolated sum 3.14 us | 16 MiB producer copy; consumer independently reads 128 MiB weights before `griddepcontrol.wait`, then consumes producer output. Normal trials 59.07-60.16 us; PDL 57.57-59.68 us overlap. Difference includes cache/tail interactions, not a universal launch cost. Programmatic graph edge verified locally | measured | `849bd4b650e54c7f`, boundary log |
| BF16 MMA vs WGMMA SS, N8 | 504.23 vs 200.16 TFLOP/s (35.14 vs 88.51 us) | swap-AB W[64,64] x X[8,64], 132 CTAs, four independent accumulator sets, 512 repeats | measured | same job, tensor log |
| BF16 MMA vs WGMMA SS, N16 | 544.66 vs 362.57 TFLOP/s (65.06 vs 97.73 us) | same, X[16,64] | measured | same job |
| BF16 MMA vs WGMMA SS, N32 | 590.24 vs 607.90 TFLOP/s (120.06 vs 116.58 us) | same, X[32,64] | measured | same job |
| NVFP4 LOP3 decoder | 0.9815 trillion weights/s; 0.49 TB/s packed input; 3.926 trillion named LOP3 instructions/s; 136.61 us | 134,086,656 weights; 4 LOP3/weight plus extraction, sign and checksum work | measured | same job, NVFP4 log |
| NVFP4 PRMT LUT decoder | 1.888 trillion weights/s; 0.94 TB/s packed input; 1.888 trillion named PRMT instructions/s; 71.01 us; 1.92x LOP3 weight rate | 1 PRMT/weight; same exact E2M1 magnitude x2 int8 LUT values. All 16 codes and stream checksums agree | measured | same job |

Tensor work and output sums match, but resources differ: MMA uses 128 threads and register operands;
WGMMA SS uses 256 threads (one consuming warpgroup), shared operands and initial tensor-map TMA.
All-one operands are reused. These are implementation rates, not isolated ISA peaks or a test of RS.
Decode rates omit E4M3 scales, BF16 conversion/packing and expert MMA; named instruction rates
are counts divided by elapsed time, not measured pipeline utilization.

### Design consequences (inferences from this check)

| Lane | Consequence | Receipt |
|---|---|---|
| Projection | Increase independent loads and CTA coverage first: LDG grows 0.64 to 3.01 TB/s. At matched request groups, shared staging costs more without reuse. MMA is the stronger tested path at N8/16; keep WGMMA SS as a candidate from N32 with actual reuse | `849bd4b650e54c7f` |
| Attention | Expose more independent row/context tiles: the same random gather rises 0.459 to 2.558 TB/s with CTAs. Use this as a gather reference for the bound, not a TLB diagnosis or attention throughput prediction | same job |
| MoE | Try PRMT decoding in the real expert path; the synthetic 1.92x decode gain is large enough to investigate. Include scales and BF16 packing, then measure whole expert time. Tiny token tiles still favor the tested MMA path | same job |
| Decode layer | Prioritize request concurrency and fusion. PDL is a small lead (1.06 us here) with overlapping trial ranges; test actual stage boundaries before adopting it. evict_first improves this retention case; APW adds little consistently | same job |
| Prefill | Small-N SS results do not close large-N WGMMA/TMA pipelines. Test tiles with shared reuse and more consumers. Warm L2 remains 7.3+ TB/s at 32 MiB here, so do not assume a fixed 32 MiB capacity cliff | same job |

## SGLang's kernels on H200 (the 1.00x we are timed against)

| Quantity | Value | Scope | Class | Receipt |
|---|---|---|---|---|
| cuBLAS BF16 GEMM (qkv_gate, o, pooler, unpooler) | 2.8 us + bytes / 3.54 TB/s (74% of 4.8 TB/s); 47-76% of floor per shape | bf16_proj, 22 shapes with floor >= 3 us | measured | `runs/muse_bf16_proj/*7fb3e93056d34999.json` |
| FP32 router (custom GEMM <= 8 tokens, BF16 hi+lo split above) | 10.3-26.3 us against a 1.8-2.7 us floor: 7-17% | bf16_proj, router, all loads | measured | same |
| Sliced unpooler (<= 8 tokens, 8.4 MB) | 6.8-6.9 us against 1.75 us: 25% | bf16_proj | measured | same |
| Attention, sliding-window layer | 20.5-34.3 us against 0.1-3.7 us: 1-11%, nearly all fixed cost | attention_decode, batch 1-32, q_len 1 and 8 | measured | `runs/muse_attention_decode/*cbebc3d45a804f30.json` |
| Attention, full layer at 80K | 31.6 us (batch 1) to 391.5 us (batch 32); 31-33% at batch 16/32; fit 21.8 us + bytes / 1.66 TB/s | attention_decode | measured | same |
| MoE (Marlin W4A16 + routing + post-norm) | 77.3, 192.0, 207.2, 142.5, 457.3, 620.7 us at bs 1, 16, 32, 8, 128, 256: 15-34% of floor | moe_w4a16, scored shapes | measured | `runs/muse_moe_w4a16/*ea3d5b6a93634bb7.json` |
| Decode layer (whole layer: dense, router, FA3, Marlin MoE, norms) | sliding 169.5 / 252.2 / 253.3 us, full 179.7 / 439.0 / 614.1 us at batch 1 / 16 / 32 (decode); verify (q_len 8) sliding 260.5 / 556.0 / 757.5, full 266.0 / 721.6 / 1112.8 us: 19-34% of floor; fit 1.2 us + bytes / 1.25 TB/s | decode_layer, 12 scored shapes | measured | `runs/muse_decode_layer/*b90f4b3c889348d5.json` |
| Prefill layer | turn 1 (753 tokens at 80K): sliding 1,095, full 1,473 us; turn 4: 3,412 / 4,887 us; first-turn 16K chunks 17,192-24,731 us: 23-41% of floor | prefill_layer, 8 scored shapes | measured | `runs/muse_prefill_layer/*322d17e35dfe454f.json` |
| Where SGLang's time goes (profiles of baseline.py on the judge) | decode layer: batch 1 sliding 180 us = Marlin 45 + cuBLAS/router 55 + FA3 10 + ~70 us of norms, elementwise, routing and reductions over ~40 kernels; batch 32 full 659 us = Marlin 149 + FA3 375 + GEMM 93; verify 32 sliding 798 us = Marlin 604. MoE bs 1: Marlin 41 of 75 us (rest torch routing/glue); bs 256: Marlin 575 of 597. Prefill turn 1: Marlin 641-661 us of 1,008-1,474 (FA3 423 us on the full layer); first-turn chunks: Marlin 10.5-13.4 ms, FA3 0.3-8.1 ms (grows with context), GEMM 4.2-4.4 ms of 16.6-25.4 ms. Attention: FA3 main + combine 13.7 of 20.1 us at batch 1 sliding (9 kernels) | all five problems, scored shapes | measured | `runs/calibration/profiles/*_sglang.*.txt` |

## Model structure (static)

| Quantity | Value | Scope | Class | Receipt |
|---|---|---|---|---|
| Layers | 62: 46 sliding-window (2048) + 16 full attention; all MoE; layers 0 and 61 keep BF16 experts | Muse Spark Open | static | `serving/sglang/python/sglang/srt/configs/muse_spark.py`, `layers/muse_spark_bf16_moe.py`; problems' load weights |
| Per-GPU dense weights per layer | qkv_gate 37.7 MB, o 16.8 MB, pooler 67.1 MB (replicated), unpooler 67.1 MB (8.4 MB sliced at <= 8 tokens), router 8.4 MB FP32 | TP8 | static | `sim/muse.py`, bf16_proj `reference.py` |
| NVFP4 expert | 28.3 MB (w13 + w2 + E4M3 scales); BF16 expert 100.7 MB | EP8, 32 local experts | static | `sim/muse.py` |
| Pooler on H200 | replicated on every GPU at every token count; split + all-gather (`gemm_ag`) only on compute capability (10, 3) | stock serving | static | `srt/models/muse_spark.py:846-853`, `kernels/ops/muse_spark_v12/gemm_ag.py:66-75` |
| Helion verify megakernels | GB300 (sm_103) with TP4 only; off on H200 | stock serving | static | `srt/layers/muse_spark_expert_tp.py:25-37` |
| DFlash2 drafter | 5 dense layers (hidden 8192, MLP 28672, 64q/8kv heads, dim 128), block 8, its top-k through the target LM head: ~1.57 GB read per GPU per step | serving, TP8 | static | HF `RadixArk/Muse-Spark-Open-DFlash2` config.json, `srt/models/dflash.py:1237` |
| LM head | 414 MB per GPU (vocab 202,048 / 8) | serving | static | `sim/muse.py` |
| Busiest GPU's experts touched | 2.65, 8.46, 11.9, 15.9, 22.7, 24.4 at 1, 8, 16, 32, 128, 256 tokens (MoE problem's routing) | moe_w4a16 generator, 200 samples | model | `sim/routing.json` |
| Same, decode layer's own router | 2.50, 7.50, 7.53, 9.00, 18.4, 21.4 (backed out of TASK.md floors / 0.75) | decode_layer | model | `sim/problems.py` (`layer_routing`) |
| DFlash acceptance | 4.16 tokens per verify step on AgentX (80 coding prompts); 4.5 simulated on OpenHands-like turns | Muse + DFlash V2 | official (LMSYS) | https://www.lmsys.org/blog/2026-10-15-meta-muse-spark-open (via research agent) |

## Boards (snapshot 2026-10-23T18:28:17Z)

All six public reads were taken sequentially from `18:28:17.076989Z` through
`18:28:23.338526Z` and saved in
[`runs/boards/20261023T182817Z/`](../runs/boards/20261023T182817Z/). The overall board had 567
teams; `AutoclickerI` was rank 106 with `solved=4`. Prefill is displayed on its own board and is
not one of those four decode-step problems. The per-problem rows for our team were:

| Problem | Board size | Rank | Speedup | Time (µs) |
|---|---:|---:|---:|---:|
| `muse_decode_layer` | 271 | 96 | 1.3554x | 16,688.65 |
| `muse_moe_w4a16` | 363 | 95 | 1.6528x | 8,136.66 |
| `muse_bf16_proj` | 462 | 249 | 1.1728x | 4,118.98 |
| `muse_attention_decode` | 315 | 156 | 1.3840x | 2,721.40 |
| `muse_prefill_layer` | 157 | 51 | 1.3668x | 283,436.33 |

The overall `solved` values were 1, 2, 3 and 4 for 237, 55, 36 and 239 teams respectively; there
were no solved-zero rows. This is a four-board completion/presence field, not a fifth prefill score.

The organizers do not publish the aggregation score. A reproducible snapshot fit is the portfolio
score

```text
score = max(
    decode,
    0.351 + 0.312 * MoE + 0.268 * projection + 0.105 * attention
)
```

Here `decode`, `MoE`, `projection` and `attention` are the four decode-step speedups; a missing
decode-step board contributes `1.0x`, and prefill is ignored. This treats `solved` as a displayed
presence/attempt count rather than a score term. With ties broken by team name, the fit gives 102/567
exact ranks, mean absolute rank error 9.99, Kendall τ 0.9549 over all teams, τ 0.9699 over the top
300, and all 50 top ranks exactly (top-50 overlap 50/50). The tool and constants are
[`tools/board_model.py`](../tools/board_model.py); this is a snapshot-calibrated model, not an
organizer-confirmed formula.

The tested alternatives fit worse on this same snapshot:

| Candidate | Kendall τ (all) | Kendall τ (top 300) |
|---|---:|---:|
| Fitted `max` portfolio above | 0.9549 | 0.9699 |
| Arithmetic speedup with baseline-time weights (0.506/0.301/0.108/0.084) | 0.9133 | 0.8248 |
| Time-weighted harmonic speedup | 0.8928 | 0.7592 |
| Weighted geometric speedup | 0.9030 | 0.7870 |
| Rank sum with missing `N+1` penalty | 0.7077 | 0.6990 |
| Percentile/Borda and geometric per-problem rank variants | ≤0.7722 | ≤0.8054 |

The overall rows have unique ranks, so no board tie rule is observable; solved-first ordering did not
improve the fit. The `solved` field is not a safe missing-score penalty: a few teams' displayed count
exceeds the currently visible ranked rows, consistent with unranked or moving entries during the
six-read window.

For our current speedups the model score is 1.3554x and predicts rank 104 versus the API's rank 106.
The model's immediate +0.01x rank changes are:

| +0.01x on | Predicted rank | Change |
|---|---:|---:|
| Decode layer | 99 | −5 |
| MoE | 104 | 0 |
| Projection | 104 | 0 |
| Attention | 104 | 0 |

The model's rank-50 score threshold is 1.5229x. The least total gain is decode-only: raise
`muse_decode_layer` from 1.3554x to about 1.5229x (`+0.1675x`), leaving the other three at their
current values; the predicted rank is 50. The next single-board path is MoE-only at about `+0.6944x`.
The ranking and threshold move with the boards, and the fitted constants are not independently
verified by the organizers.

## Our kernels on the H200

Per-load speedups against SGLang from our checks. The per-shape tables are in `problems/<p>/profiles/<cand>.<job>.txt`.

| Kernel (job) | Overall | Decode loads | Verify loads (serving's) | Notes |
|---|---|---|---|---|
| MoE V9 headroom model | Receipt V9 apparent HBM bandwidth 2.14 / 3.11 / 3.08 / 3.06 / 2.17 / 1.50 TB/s for decode M1/16/32, verify M8/128/256; direct routing bytes 74.9 / 338.3 / 451.5 / 239.5 / 644.1 / 694.7 MB | M256 is the largest weighted gap; M128/M256 BF16 tensor floors are 19.85 / 39.38 us, below HBM | `runs/muse_moe_w4a16/20261023T115744Z_check_f6028a478d594b31.json`, `sim/routing.json`, P8 bandwidth probe |
| MoE V11 (`41bcf27cfc0e4eff`) | 1.664x | 2.185 / 1.684 / 1.398 | 1.800 / 1.592 / 1.439 | H200 board best; V9 native through M128 and Humming wide at M256; profile shows M256 wide kernels at 251.2 + 140.5 us | `problems/muse_moe_w4a16/profiles/cand_v11.41bcf27cfc0e4eff.txt` |
| MoE V17 adaptive row-tile portfolio (local) | 1.269x sm120 only | 1.680 / 1.253 / 1.076 | 1.251 / 1.118 / 1.319 | Graph/cold-L2 prepare autotunes V9/V12/V16/Humming/adaptive expert-major reuse; H200 result pending | `problems/muse_moe_w4a16/cand_v17.py`, sha256 `967c926a…` |
| MoE V18 router-aware portfolio (local) | 1.264x sm120 only | 1.683 / 1.231 / 1.067 | 1.271 / 1.091 / 1.329 | V17's 1% semantic gate plus optional decode-layer router tuner; standalone MoE weights use the screened fallback | `problems/muse_moe_w4a16/cand_v18.py`, sha256 `3cb4f105…`; `problems/muse_decode_layer/moe_tune.py` |
| MoE grouped TMA: H200 implementation closed | Lab `c8c89f9f111845af`, all12 choices lose. Best synthetic-real routing: M128554.0us vs V9292.5 (1.89x); M256835.7 vs543.2 (1.54x). Board routing:833.2 vsV9280.4 (2.97x),995.0 vs453.4 (2.19x). Other choices are still slower; queued candidate withdrawn and parked | Raw17 ring/swap-AB/hot-row/prefix scheduling fails to amortize at these shapes. The lab times whole kernels and does **not** isolate a causal stall or prove this family inherently loses. Local sm120 gains did not transfer. Synthetic Humming/V11 gains are inadmissible: exact decode-weight lab26d7e05ab97d4f96 fails agreement (rel.11–.32). V9 remains the valid M<=256 decode control | `problems/muse_moe_w4a16/dev/parked/cand_tma_grouped.py`, `dev/tma_grouped/README.md`; exact-weight lab: M128V9p278.7/Marlin387.7us; M256478.1/561.1 |
| MoE V9 phase diagnostic / CTA imbalance | `cand_v9_phase.py` sm120 trace with H200-like local pairs: M128 median p1/p2/wait/barrier 0.771M/0.384M/0.016M/0.061M cycles, 416 norm atomics; M256 0.734M/0.362M/0.014M/0.030M, 656 atomics. M256 routing model averages 386.93 pairs, 24.39 experts, 60.53 V9 token-blocks; V9's 188 CTAs get 163/163/164 tiles (max/mean 1.01x over 200 samples). The `ceil(n_e/8)` blocks rise 36.76 -> 60.53 from M128 to M256, repeating 512/256 weight tiles; padded norm atomics are ~386/320 per pair | Static CTA tile imbalance is too small to explain H200's M256 tail; heavy-expert phase waits remain a hypothesis. V16 CTA-private partial reduction is packed into the diagnostic, passes local correctness, but adds global traffic and was not ported | `problems/muse_moe_w4a16/cand_v9_phase.py`, `problems/muse_moe_w4a16/dev/m256_imbalance.txt` |
| MoE V9 (`f6028a478d594b31`) | 1.654x | 2.196 / 1.717 / 1.392 | 1.813 / 1.590 / 1.356 | 256-thread fused resident kernel; thread 0 prefetches instead of a producer warp; incremental tile iterator. Beats v7 and V6 at every load; hybrid (V6 to bs192, Humming3 above) `318227bc382d465c` 1.536x |
| MoE v7 (`f95bdf5e394e474e`) | 1.535x | 2.136 / 1.614 / 1.326 | 1.739 / 1.312 / 1.255 | v4 with the first tile decoded once per phase and incremental expert/token-block/row indices instead of a binary search per tile; loses V16 to V6 |
| MoE V6 (`eefecf1eeb724feb`) | 1.492x | bs 1/16/32: 2.054 / 1.542 / 1.249 | V1/V16/V32: 1.635 / 1.405 / 1.211 | One fused resident kernel: routing, both GEMMs, SwiGLU, norm and weighted sum; 8-token tiles. Profile shows only the fused kernel |
| MoE v4 (`84c97d9483b34573`) | 1.353x | 1.996 / 1.400 / 1.136 | 1.498 / 1.154 / 1.115 | The kernel in today's serving patch |
| MoE Humming3 (`09ba4af4e6bc4791`) | 1.270x | 1.640 / 1.164 / 0.949 | 1.234 / 1.315 / 1.424 | Beats V6 at V32: wider tiles pay off at 256 tokens |
| Attention hybrid (`4f8c41ef9e924ff4`) | 1.388x | 2.05, 1.65 / 2.09, 1.91 / 1.89, 1.95 | 0.99-1.01 at every verify load (it is FA3 there) | Custom q_len=1 kernel plus SGLang FA3 for q_len=8; scored 1.384x (`526f31018e2c4449`) |
| Attention portfolio q1 (`aff09765beaf429a`) | 1.3727x (board stays 1.384x) | decode: 2.054/1.630, 2.075/1.914, 1.877/1.959 (sliding/full by batch) | verify: 0.952/0.975, 0.957/0.994, 0.966/0.995 (sliding/full by batch) | q1 stream remains strong; FA3 wins verify. Its prepare rejected XQA/WGMMA at 5.4-5.6% L2 because the old 3% FA3 proxy gate was stricter than TASK.md's FP32-reference bound. Receipt `20261023T201903Z_submit_aff09765beaf429a.json` |
| Attention v2 (`51ad2ea9ec4f4724`) | 1.279x | sliding, full: 2.04, 1.68 / 2.04, 1.91 / 1.87, 1.96 | 0.93, 0.89 / 0.60, 0.88 / 0.67, 0.93 | Wins decode on fixed cost; loses verify to FA3 |
| Attention v5 (`ced8d779b16f44a9`) | 1.262x | — | sliding verify 1.49 / 1.04 / 1.14; full verify loses | Sliding verify beats FA3; full verify doesn't |

The verify-stream candidate `cand_attn_verify_stream.py` keeps the q1 KV page
gather and uses 64 rows per KV head for FP8-to-register dequantization, MMA
QK/PV, in-block causal masking, and split `(m,l,o)` reduction. Its prepare
compares FA3, the stream family, XQA/WGMMA and FP8-PV per shape, admits choices
under the TASK.md 15% maximum, rejects failures without aborting, and prints a
compact final winner/choice summary. It passes the local 16 tests and 12 graph
shapes; H200 execution is unverified.

The q1 receipt's H200 profiles put FA3's main verify kernel at 188.5 us of
205.5 us total for B16/full and 375.9 us of 394.0 us for B32/full. Sliding
verify totals are 19.4/25.1/30.1 us at B1/B16/B32, with FA3 main only
6.8/10.9/15.0 us; the rest is combine, norms, cache store, metadata and gate
work. This separates the full-layer KV-stream target from the sliding fixed
overhead.
The latest attention receipts keep that conclusion: `cand_attn_portfolio` (`45785321ca03483c`)
reaches 0.996x/0.996x at verify16/32 full, and `cand_attn_fa3_struct`
(`4ccbf59306a74b8d`) reaches 0.995x/0.994x there, both with 16/16 tests passing. Their
full verify32 main kernels are 379.7/375.6 us of 397.2/394.0 us; the structured family is
therefore a per-shape diagnostic rather than a claimed standalone win. The layer candidate
`cand_r3_attention_v2.py` embeds both sources and lets the H200 choose them against its own
q/gate/KV layout; sm_120 forces the portable stand-in.
| Decode layer moe7 (`66d5d4c9e4054fa9`) | 1.355x | sliding, full: 1.53, 1.51 / 1.49, 1.54 / 1.17, 1.48 | 1.70, 1.63 / 1.17, 1.11 / 1.09, 1.06 | MoE v7 tiles in the tuned layer; all_decode `8577b1a2f9054ca2` 1.289x, glue `d32113fa98f946a3` 1.283x; verify 16/32 are the gap; scored run queued |
| Decode layer tuned (`0fa94cd98ba34c93`) | 1.271x | see profile | see profile | MoE v4 + repaired decode attention + prepare-time cold-L2 GEMV autotuning; moe4_attn `a0b66fe115024a68` 1.220x, native MMA projections `15003b60a9734d28` 1.249x |
| Decode layer moe4 (`6fed72479ef6446f`) | 1.189x | 1.35, 1.33 / 1.25, 1.14 / 1.05, 1.02 | 1.49, 1.47 / 1.10, 1.09 / 1.03, 1.03 | v3 plus persistent MoE v4, FA3 attention; gains still shrink with batch |
| Decode layer v3 (`2c112482e81745af`) | 1.143x | 1.25, 1.23 / 1.18, 1.10 / 1.03, 1.02 | 1.39, 1.35 / 1.06, 1.04 / 1.05, 1.03 | Gains shrink with batch; MoE and attention inside are older than V6 |

| Prefill v11 (`6a9cd943fe404d74`) | check 1.367883x; scored 1.366811x (`965a9a816d4744ea`) | not on overall board | scored turn 1 / turn 4 / first turn: 1.227 / 1.356 / 1.534 | BN128 WGMMA below8192 tokens, timed dequant + grouped BF16 DeepGEMM above; submission is the exact checked source |

### BF16-resident decode MoE hybrid (independent lane)

**Inadmissible (CONSTRAINTS 5, boundary row):** the timed calls read a BF16 bank dequantized in prepare, and NVFP4 weights must stay NVFP4. Its decode run (07d07d96) was withdrawn before scoring and then ran anyway: 1.461x against 0e94268's 1.428x, off the board; the timings below stay as evidence of what dequant costs V9 at M=256 (the hybrid's ~59 us win is the dequant work it moved out of the timed call).

**Decision:** build on `cand_r3_v9.py` (SHA `ba990a86...`, receipt
`0e94268cf5654012`, 1.427904x). No newer merge has beaten it. Only lab payloads
are submitted by this lane; Jihoon submits the layer candidate. The design uses
resident BF16 hot experts and native V9 W4 cold experts, independently of TMA W4
and V19. The first complete-path H200 result is a **modest M256 win**, not the
200/120us target. Deliver the locally validated candidate with prepare-time
selection; the optional resource-budget lab continues asynchronously in the
local queue. Do not hold the reviewable candidate on that additional result.

**Storage and arithmetic:** retain the packed NVFP4 bank. The additional
3,221,225,472B BF16 bank stores E2M1 x E4M3 block products exactly. Apply each
FP32 global scale in the SwiGLU/output epilogue, never in the BF16 bank. Fused
post-expert RMSNorm uses FP32 global scale and routing weights. This one-layer
G1 allocation is not proposed across all62 layers. Prepare drops
unselected BMM layouts and the full derived bank if only subset BMMs win.

**H200 evidence:** proj-host jobs `f8e977320c3c4556` (GEMM ceiling) and
`b37b93deb1b74f3a` (complete hybrid) finished successfully. Both reconstruct the
public decoder weight distribution with seed5353 and use the corrected
`moe_tune.py` fixture: draft sharing within every8-token verify block. Device RNG
produces M128:177 local pairs/hottest57; M256:341/hottest104. These are not the
private loaded router from receipt `0e94268cf5654012`. The other loaded-weight
lab `26d7e05ab97d4f96` has M128:209/55 and M256:414/119; its V9 paired times
are279.0/478.5us. Final dispatch must compare engines using each run's actual
loaded router in prepare.

GEMM ceiling, CUDA-graph microseconds, W13 + activation + W2; **excludes routing,
post-norm and combine**. Grouped MM uses padded contiguous rows; BMM pads each
selected expert to M rows. BMM FP32 output keeps global13 in the activation
(epilogue); the column below uses the NN layout.

| M | Hot threshold | Experts / rows | PyTorch grouped | BMM FP32 | DeepGEMM masked | DeepGEMM psum |
|---:|---:|---:|---:|---:|---:|---:|
|128|16|3 /104|287.95|126.42|98.03|110.64|
|128|32|1 /57|211.68|52.62|66.43|94.78|
|256|16|8 /297|538.56|419.92|247.74|225.04|
|256|32|3 /187|370.26|160.03|101.87|113.95|
|256|64|1 /104|302.46|69.36|97.20|96.37|

Complete hybrid includes partition/gather, cold V9 stream, both hot GEMMs,
fused activation, norm and combine. Three independently refreshed routing
fixtures, 128MiB cache flush, captured graphs; times from `b37b93deb1b74f3a`.
The cold grid fraction describes CTAs, not an enforced SM partition. V9 uses
two resident CTAs/SM on H200 (88,936/94,056B dynamic shared memory at M128/256).

| M | V9 us | Best BF16 hybrid us | Policy | Decision |
|---:|---:|---:|---|---|
|128|237.63|238.13|BMM NN FP32, threshold32, serial|Keep V9|
|256|396.10|337.20|DeepGEMM psum, threshold 16, quarter cold grid|14.87% less MoE time; retune in prepare|

M256 global relative L2 is0.003704 against fully FP32 expert math; every timed
policy also passes refreshed graph gates (global<=1%, row<=5%). Smaller cold
grids are much slower: at M256/threshold 16 psum, one-eighth/quarter/half/serial
are526.4/337.2/374.6/359.9us. Do not substitute the isolated GEMM floor for a
full-path gain. The measured win alone cannot establish the ~1.68x layer goal. Applying its
MoE ratio only to verify32 in receipt `0e94268cf5654012` models ~1.453x overall
(weighted sliding/full load sums, then geomean over six loads), not a measured
score or rank50 result.

**Static bandwidth check:** public fixture0 touches19 M256 experts. Threshold16
reads at least805.3MB of hot BF16 weights plus311.4MB of cold packed weights,
~1.117GB total:233us at the [H2004.8TB/s spec](reference/hardware.md), before
activation traffic and launches. This is conditional on that fixture and these
layouts; it does not bound the three-fixture median or a higher-threshold design.
The original ~70us estimate used a different, narrower expert set. BF16 storage
is lossless and HBM capacity is available, but reading it costs bandwidth.

A final lab varies one-stage native V9 buffering, DeepGEMM's public SM-budget
API, and fixture-sized BMM row capacities (25% headroom). A GPU count exceeding
the prepared capacity stays entirely on V9, preserving every row. The compact
BMM prototype passes90 eager cases and36 refreshed graphs including overflow
routing extremes (max FP32 L2 0.003755). Its lab replaces a locally waiting
snapshot before judge admission; no run was consumed by that replacement. The
current candidate admits these resource policies only through the same fresh
FP32 checks and graph timing against V9. Their H200 resource-ablation table is
still pending; the completed tables above measure the original resource settings.

**Local verification (GPU0/sm120):** exhaustive finite E4M3 bit patterns x all16
FP4 codes expand exactly. Engine passes240 eager cases and120 refreshed graphs
across M32/128/256, three real-router fixtures and all-remote/one-hot/eight-hot/
uniform extremes; max fully FP32 relative L2 0.003832, worst row0.004361.
Masked expert strides must be padded128. FP32-output BMM max L2 is0.002813.
`runs/screen/bf16-engine.json` records cases. The delivered candidate SHA
`123bf83be45da22f1e72ac6f62d7c39c13caf79f3423362e24cdd4e6d9eb6fcf`
passes 13/13 tests, 12/12 graphs and4/4 static checks (three embedded CUDA
sources compile for sm90a); the same holds with Hopper selection forced through
portable grouped stand-ins and real cuBLAS FP32 paths, including stage1, SM
budgets and compact BMM capacities. Normal/forced prepare is1.37/12.23s.
`check_prepare.py` also verifies production reclamation after BMM-only selection:
6 eager plus6 refreshed graph cases,1,610,621,696 live derived-bank bytes. The one-stage/SM-budget
prototype separately passes24 eager cases and16 refreshed graphs with actual
DeepGEMM on GPU0; it restores the global SM setting after each call.

| Tokens | Load | Dispatch |
|---:|---|---|
|1 /8 /16|decode1 /verify1 /decode16|V9 fixed|
|32|decode32|Prepare compares V9 with admitted BF16 policies|
|128|verify16|Prepare compares; public-seed H200 lab keeps V9|
|256|verify32|Prepare compares; public-seed H200 lab selects psum threshold 16|
|Other|nearby shapes|V9|

The standalone candidate embeds the canonical `dev/bf16moe/hybrid.py` engine
and independent prepare-only FP32 oracle. Selection uses three synthetic
loaded-router fixtures, cold graph timing, refreshed graph agreement and a
12-replay finalist comparison against V9. M256 is tuned first, then M128 and
M32;8s of the270s overall budget is reserved for finalist gates. Row thresholds
16/32/64, one/two V9 buffering stages, quarter/half/full resident grids or
serial execution, and even quarter/half/full DeepGEMM SM counts are compared.
BMM uses FP32 outputs in both layouts and a fixture-derived row capacity with
25% headroom; overflow routes entirely through V9. No timed-input specialization. Only
lab payloads were submitted. Full-layer H200 correctness, prepare selection,
score, private-router performance and any TP8 serving integration remain
unverified.

### Decode large-load profile and round-3 dispatch (G1)

The accepted decode control is `cand_moe7` (`66d5d4c9e4054fa9`, 1.354968x, CI [1.334644, 1.375293]); its exact source is frozen in `problems/muse_decode_layer/cand_moe7.py`. The H200 per-kernel profile identifies the limiting stages at the loads that carry the remaining board gap:

| Load | Candidate GPU time | Largest kernel | Other large kernels | Evidence |
|---|---:|---|---|---|
| verify 16 sliding | 400.1 us | v7 fused MoE 285.8 us (71.4%) | BF16 GEMVs 28.1 + 26.2 + 15.1 us; FA3 14.3 us; boundaries 11.6 us | `runs/muse_decode_layer/20261023T104943Z_check_66d5d4c9e4054fa9.json`, profile 8 |
| verify 16 full | 655.3 us | v7 fused MoE 362.1 us (55.3%) | FA3 192.9 us; projection kernels 28.3 + 26.6 + 15.2 us; boundaries 11.7 us | same, profile 9 |
| verify 32 sliding | 649.7 us | v7 fused MoE 499.4 us (76.9%) | projection kernels 35.5 + 33.2 + 23.2 us; FA3 20.3 us; boundaries 19.7 us | same, profile 10 |
| verify 32 full | 1009.5 us | v7 fused MoE 502.9 us (49.8%) | FA3 377.5 us; projection kernels 34.8 + 32.8 + 23.2 us; boundaries 19.8 us | same, profile 11 |
| decode 32 sliding | 218.0 us | v7 fused MoE 119.4 us (54.8%) | projections 22.5 + 19.5 + 21.0 us; custom attention 15.4 us; boundaries 9.5 us | same, profile 4 |

The dominant opportunity is the 128/256-token expert path. Standalone H200 results show Humming3 at 256 tokens reaches 1.424x versus v7's 1.255x, while v7 remains the measured small-M winner. `cand_r3_moe_dispatch.py` combines the v7 decode layer with the MoE lane's canonical-bank V8 and grouped wide-tile implementations; `prepare()` times v7, V8 and grouped variants on synthetic shape representatives and stores only a shape-keyed choice. Its sm_120 screen passes 13/13 tests and 12/12 fresh-input graph cases; local timings are admission evidence only, not an H200 claim.


In the layer on the H200, V9 beats the board's `cand_r3_moe_dispatch` (1.367x, `85e67f5d6ef945c9`) at the first 10 of 12 loads by 1-9%, but `cand_r3_v9_dispatch`'s grouped wide-tile path at the last two loads (verify 32, MoE at 256 tokens) collapses to 0.549x/0.647x against 1.105x/1.071x, for 1.266x overall (`2ad75c19cb514f5c`). V9 below 256 tokens with the board's path at 256 would come to about 1.41x (geomean of the per-load ratios). Standalone, grouped and Humming win at 256 tokens, so a synthetic prepare-time ranking can pick a path that loses inside the layer: read each autotune run's MOE_TUNE choice against its per-load result. The evidence: on the H200, prepare timed MoE at 256 tokens with synthetic routing as grouped 1.87 ms < V9 1.98 (`2ad75c19cb514f5c` log) and V7 1.79 < grouped 1.94 < V8 2.27 (`85e67f5d6ef945c9` log), yet grouped inside the timed layer made verify 32 about 2x slower. Tune MoE on routing from the layer's own router weights (prepare receives them) rather than synthetic ids, and gate paths that lose in-layer receipts.
A router-aware direct MoE probe on sm_120 isolates the missing mechanism. At M=256, real router fixtures had 672 local pairs across 8 experts with a 221-route hot expert; matched-local-count fixtures had 672 pairs across all 32 experts; uniform fixtures had 2,048 all-local pairs. Graph medians in ms were grouped 0.915/0.866/1.818, v7 0.584/0.763/1.693, v9 0.668/0.841/4.738, V11/Humming 0.536/0.760/1.652, and V17 adaptive 0.662/0.827/4.659 for real/matched/uniform. The hot tail leaves grouped's histogram+plan+scatter, two GEMMs and combine (six launches) with fixed task overhead, while persistent paths benefit more from fewer active routes. No host synchronization appears in `_grouped_call`; the remaining full-layer gap is launch/task interaction under skew, not synthetic GEMM ranking. `problems/muse_decode_layer/moe_tune.py` now routes loaded layer router weights on random hidden states and applies V17's 1% semantic gate before timing.


In-layer overlap (`cand_r3_merge_ovl.py`, local only, parked): the only independent work is row partitions inside qkv and the unpooler; boundary 1's dense and routed paths share norm statistics; KV write -> attention, pooler/router -> experts -> unpooler and router -> attention output are ordered. Two-stream overlap was 9.5% slower at decode 32 (505.9 vs 461.8 us, sm_120) and failed the correctness guard at verify 16/32 (implementation suspect: a race), so it was not sent.
The profile also exposes small separate residual, router, norm and gate kernels. Existing structural fusion candidates reduce those launches but regressed complete sm_120 layer timings, so the new diagnostic keeps the v7 glue as the control while selecting the expert family first. Any Hopper result must recheck the Python glue with a portable stand-in and compare the full-layer receipt, not infer a win from the isolated MoE ratio.

#### Glue fusion diagnostic (G1)

The parked `cand_r3_v9_glue.py` (tiled final SHA `0bd8624b275b597b5485daf85a45ac7e6d68d9467b01bcf1a5c1be48df96d7cd`) ports the epilogue fusion into the V9 board candidate: a PDL
O-projection stats kernel publishes the two residual RMS coefficients, then a PDL pooler/router
kernel applies the first residual/norm and writes pooler/router partials for the existing PDL
route/top-8/softmax/pre-expert-norm kernel. The tuner gates the fused proxy against FP32 glue and
routing references, rejects every failure, and selects by CUDA-graph replay time.

Nsight Systems on sm_120 identified the performance cause. The original pool-boundary launch was
a 72×4 grid and averaged about 120 us per invocation; it reread the pooler/router weights for
four K-splits while every CTA processed padded token lanes. The O-stats kernel averaged about
11 us and waited for 128 producer CTAs. The fixed parked source tiles tokens with TB≤32 and a
third grid axis, so T=128/256 now runs instead of failing shared-memory allocation. Nsight now
measures about 81 us for pool-boundary and 11.8 us for O-stats, but the two-kernel chain still
materializes the O branch, performs the four split-K pool/router reads, and launches a second
consumer. That remaining work is why fusion stays slower than regular v9.

Final tiled sm_120 glue timings (regular→fused) are T=1 `77.8→81.9` us, T=8
`81.9→86.0`, T=16 `81.9→109`, T=32 `84.0→123`, T=128 `92.2→303`, and T=256
`137→522`. The fixed source passes 13/13 tests and 12/12 graphs but is parked; no H200 run is
justified while the tuner selects regular at every shape.

#### Union small-shape overhead and merge2 control (G1)

The H200 union receipt `215cafaf8d674c7e` loses to the V9 control `0e94268cf5654012` by
6.6/7.8 us at decode-1 sliding/full, 8.9/12.6 us at decode-16 sliding/full, and
11.2/17.0 us at verify-1 sliding/full. The profile identifies a launch change, not a
graph-replay artifact. At decode-1 sliding V9 has four `_gemv_kernel` launches totaling
45.5 us; union has three `_gemv_kernel` launches totaling 28.1 us, then adds a 22.1 us
`nvjet_sm90_tss_192x8_64x8_4x1_v_bz_splitK_TNT` cuBLAS pool/router kernel and a 1.7 us
cuBLASLt split-K reduction. Verify-1 sliding shows the same replacement: 46.0 us of V9
GEMV versus 29.0 + 22.0 + 1.7 us in union. The isolated projection timer favored cuBLAS,
but the extra kernel and reduction cost 6--17 us in the complete graph. Decode-32's small
apparent gain is MoE/profile variance; union still carries the extra projection family.

`cand_r3_merge2.py` starts from the merged source but forces V9 custom GEMV at every scored
token count, disables the unproven prefix/persistent paths, and leaves V9 attention/PDL/MoE
as the control until pending receipts provide complete per-shape winners. Its sm_120 screen
passes 13/13 tests and 12/12 graphs; prepare is 16.34 s. The H200 receipt behind every
current choice is `0e94268cf5654012`; pending v9all/merge/cost/attention receipts remain
queued and may replace a choice only when the complete-layer median beats V9 without adding
a launch.

#### Best portfolio static repairs (G1)

`cand_r3_best.py` (`82f54e676a0b62d720847a051d6793fb7fb5a57bb4f5f15984224ba8473dfc0f`)
starts from the V9 state and keeps the cost/projection tuner. It adds
`ATen/cuda/CUDAContext.h` to `LT_CUDA_SRC`; this fixes the H200 `getCurrentCUDAStream`
compile failure. It embeds the attention lane's matrix-XQA, grouped-XQA and FP8-PV source
families with the layer's raw-QKV/cache ABI, defines the rope cache used by the FA3 proxy,
and removes closure-invalidating `del` statements that ruff reported as undefined names.
Small pool/router projections stay on V9 custom GEMV (the union launch evidence above);
MoE is V9 for every token count. Prefix fusion is disabled because its 128/256 tiles exceeded
the 232,448-byte Hopper shared-memory limit. The sm_120 screen passes 13/13 tests, 12/12
graphs, all 14 undefined-name/sm_90a source checks, and cold prepare is 16.56 s. H200
attention/projection choices remain unverified until lab job `59e53e746df341a9` is available;
the V9 fallback is backed by receipt `0e94268cf5654012` and the cost-v9 family by
`906ca5580aef4f86`.

#### Larger-load stage floors and non-MoE gap (G1)

Receipt `0e94268cf5654012` supplies the H200 profile and scored layer times. The table groups
profile rows into projections (qkv, O, pooler/router and unpooler), attention (main, combine,
qk-prep and FA3 preparation/norm support), MoE, and glue (boundary plus route). The MoE entry is
shown as `layer-profile → best standalone lane`: the right-hand value comes from `cand_v9_phase`
receipt `ac9dd461f0294e34` at M=32/128/256 (147.5/277.8/422.2 us). `gap` uses the layer-profile
value so it remains the profile's adjusted-time residual; the arrow is the headroom available to
the next merge. Floors are the rules-compliant one-read floors at 4.8 TB/s from
[`bound.md`](reference/bound.md): projection weights are counted once across the dense group,
attention uses the legal FP8-QK/BF16-PV work, and MoE uses `sim.problems.layer_routing`; glue's
scalar floor is below 0.1 us at these sizes.

| Shape | Profile stage time (us): P / A / MoE / glue | Floor (us): P / A / MoE / glue | Adjusted / gap (us) |
|---|---:|---:|---:|
| decode32 sliding | 66.2 / 20.2 / 101.7→147.5 / 15.6 | 41.8 / 3.55 / 53.1 / <0.1 | 217.5 / +13.9 |
| decode32 full | 73.4 / 228.5 / 114.2→147.5 / 24.9 | 41.8 / 139 / 53.1 / <0.1 | 422.9 / −18.1 |
| verify16 sliding (M=128) | 72.8 / 25.8 / 258.9→277.8 / 16.3 | 43.9 / 1.78 / 108.8 / <0.1 | 393.4 / +19.6 |
| verify16 full (M=128) | 72.5 / 202.6 / 280.4→277.8 / 15.7 | 43.9 / 69.3 / 108.8 / <0.1 | 566.6 / −4.6 |
| verify32 sliding (M=256) | 90.0 / 31.6 / 477.0→422.2 / 25.9 | 51.7 / 3.56 / 125.9 / <0.1 | 615.5 / −9.0 |
| verify32 full (M=256) | 90.7 / 389.9 / 449.5→422.2 / 25.8 | 51.7 / 139 / 125.9 / <0.1 | 979.6 / +23.8 |

The profile's `STAGE_TUNE` lines at T=32 independently report qkv/O/pooler-router/unpooler
choices of 21.0/11.8/33.1/28.3 us; PDL makes their visible profile aggregate 66.2--73.4 us.
At M=128/256 the profile's three dense kernel families plus split-K reductions give 72.5--90.7
us directly. The standalone MoE receipt's 422.2 us at M=256 is the reusable lane result; the
layer profile's 449.5--477.0 us shows its routing/context penalty. The largest non-MoE gap is full attention at verify32: 389.9 us versus a 139 us
floor (+251 us), followed by verify16 full (+133 us) and decode32 full (+90 us). Projection gaps
are 22--39 us, while glue has no material byte floor but consumes 16--26 us of critical-path
work. The next score experiment should therefore target the full q_len=8 attention path before
another dense fusion.

#### Megakernel correctness audit (G1)

The held `cand_r3_merge_mega.py` (`98d7e6d6…`) did not have a benign isolated rejection. Two
independent sm_120 prepares rejected the persistent qkv chain at `(tokens=32,q_len=1)`; one
repeated debug prepare later accepted that same shape, which identifies nondeterministic launch
ordering rather than a stable tolerance margin. The `(8,8)` check was repeatably different: the
persistent qkv chain produced about 0.864% hidden and 0.826% stream relative error and changed
expert IDs. Turning PDL off removed the launch dependency but did not remove that qkv disagreement.
The qkv repair therefore never exposes the qkv producer to a timed replay, but it does not make
the dense islands safe: a later prepare rejected `(tokens=16,q_len=1)` even with regular qkv.
Both mega sources are parked. The ready `cand_r3_attention_v2.py` keeps the validated merge path
and adds the embedded XQA/FP8-PV and FA3-structured WGMMA verify families; its final
`ATTN_PORT_SUMMARY` is printed after all prepare work.

#### Batch-1 stage accounting and megakernel rationale (G1)

The H200 profile for `cand_r3_v9` (`0e94268cf5654012`, profiles 0, 1, 6 and 7) gives the following batch-1 decomposition. “Kernel sum” is the labelled GPU time grouped by layer stage; “gap residual” is the raw graph median minus that sum. Negative residuals are possible because PDL and concurrent work make a profile's per-kernel rows overlap. The byte floors use one read of each stage's weights and 4.8 TB/s; projection and attention bytes are the per-stage formulas in [`bound.md`](reference/bound.md), and MoE bytes use its layer-routing expectation.

| Load | Adjusted / raw layer (us) | Kernel sum by stage (us) | Gap residual (us; share of adjusted) | 4.8 TB/s byte floor by stage (us) |
|---|---:|---|---:|---|
| decode 1 sliding | 108.49 / 111.22 | projections 45.5; attention 14.8; MoE 31.7; boundaries 12.9; route 3.6; misc 0.7 = 109.2 | +2.0; 1.9% | projections 28.85; attention 0.11; MoE 14.75; glue <0.1 |
| decode 1 full | 128.57 / 133.31 | projections 44.8; attention 33.9; MoE 32.3; boundaries 13.3; route 3.6; misc 0.8 = 128.5 | +4.8; 3.7% | projections 28.85; attention 4.33; MoE 14.75; glue <0.1 |
| verify 1 sliding | 149.16 / 153.25 | projections 46.0; attention 23.3; MoE 76.6; boundaries 13.6; route 3.7; misc 0.7 = 163.2 | −10.0; PDL/profile overlap | projections 28.99; attention 0.12; MoE 44.25; glue <0.1 |
| verify 1 full | 164.44 / 166.40 | projections 46.4; attention 34.7; MoE 70.8; boundaries 14.0; route 3.7; misc 0.7 = 169.5 | −3.1; PDL/profile overlap | projections 28.99; attention 4.34; MoE 44.25; glue <0.1 |

The four loads therefore do not have a large positive launch-gap share in this receipt. Their measured stage kernels are still 1.6–3.0x the one-read byte floors, especially dense GEMVs, MoE decode/route work, and FA3 fixed work. A single persistent launch is still a useful experiment because it can remove barriers and let weight streams advance across operators even when the byte floor does not bind.

This motivated the held `cand_r3_merge_mega.py`; its persistent qkv producer later failed repeated M=32 agreement checks, and the regular-qkv repair still rejected a dense island at M=16. Both are parked. Hazy Research's batch-1 Llama megakernel work reports that kernel-per-operator boundaries leave bandwidth idle and that merging the forward pass into one megakernel exceeded 1.5x over existing systems on H100 ([Hazy Research, “Look Ma, No Bubbles”](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)). Mirage Persistent Kernel generalizes the same idea with an SM-level dependency graph, cross-operator software pipelining and an in-kernel scheduler; its paper reports up to 1.7x over kernel-per-operator serving systems ([Mirage Persistent Kernel, arXiv:2512.22219](https://arxiv.org/abs/2512.22219)). Those results motivate the structure, not a speed claim for Muse: the new candidate must pass the layer's own correctness and full-shape screen.
| Proj scored (`68e2f34b1d4744cb`) | 1.142642x, CI 1.1255–1.1598 | router 1.68 / 2.26 / 1.93x; pooler m32 loses at 0.930x | router m8 4.05x; m128/256 all 0.976–1.017x | `submission.py` is the scored snapshot, sha256 `4bc18cec…`; tuned check `9294b3237c4f4e2d` 1.141605x; full table below |
| Proj hybrid (`6b706627babc4cc6`) | 1.106x | router 1.55-2.01x; unpooler m=1 1.135x | router m=8 3.81x; others 0.98-1.01x | qkv_gate and o stay at cuBLAS parity at every m |

### Projection evidence and next experiment (G1, G2)

PDL dense experiment (send **4abb8bc59fee7ae0**, payloadSHA **016706db77b05c32…**,
MoE check **7426e17cd3bc4659**):
`problems/muse_bf16_proj/dev/pdl_prefetch/README.md` freezes best7 pairedSHA3baaaa96,
its measured dense dispatch and paired glue/attention/NT1 choices. Eight cold-L2
whole-layer cases cover decode1/16/32 and verify1, sliding/full;16 standalone
role/shape rows compare original-row bulk rings, lossless tile-packed bulk rings,
512-thread GEMV and a matched528-CTA read. Early/late controls isolate the initial
weight requests before `griddepcontrol.wait`; final24 AB/BA pairs use3 refreshed
public streams, A/A controls and single-stage layer ablations. The host-independent
payload synthesizes missing dense BF16/router FP32/norm tensors from per-name
seed72140, and uses fixed public attention input, residual and FP8 KV fixtures.
MoE supplies its first32 real packed experts/scales and post_norm_w. Every arm
uses common baseline-derived route IDs/weights/localIDs; the routing/norm kernel
remains timed. The protected output metadata names all synthetic tensors. The
detached log confirms acceptance on MoE; prefill waiter3107409 was terminated,
and its waiting file removed. No H200 result yet.
Best7 remains unchanged. This tests the gap above our current1.2-2.4TB/s dense
implementations; the3.80TB/s stream and historical4.04-4.25TB/s reads disprove
an inferred2.5TB/s hardware envelope.

Local evidence:2,060 native checks/708 captured graphs pass, max FP32 relative
error4.7e-6 (unchanged native kernel); all8 layer/16 stage cases and5 MoE host
tests pass with375 forced attn_main and150 FA3 glue calls. The512-thread GEMV
and packed rings cover every M1/8/16/32 role; standalone times include the
incumbent BF16 output boundary and identical activation banks.16 native sm90a kernels and12 Triton hint variants
have no spills; SASS checks place weight requests before ACQBULK only in the
early variants;4 BF16 output-boundary kernels also cross-compile. The first screen
exposed an omitted BF16 output boundary (~0.35% layer drift): the timed reduction/
cast now preserves the incumbent ABI. Matching M32 KS512 fixes the routing change
from KS1024; exact IDs and the original1e-4 raw/router,1% BF16 and0.3% whole-layer
A/B gates all pass for the screened portfolio. A synthetic verify1-sliding QKV
cell exceeded the unchanged0.3% layer bound (0.372-0.379%): compound
A0/A4/R0/P0/P2 retain the baseline QKV there, explicitly labelled `c6 Q=B`.
`local_exclusions.json` preserves this rejection. Standalone QKV tests still
cover every kernel. Fixed-route timing is a workload-controlled diagnostic;
live-route candidate agreement remains required for promotion.
Drop-in `dev/pdl_prefetch/projection.py`SHA5769cab895f35afd…;
`manifest.json`, `native_validation.json`, `compile_sm90a.json`, `validation.json`
and `sent.json` carry full source hashes. Local timings are not H200 claims.

Current priority is **put projection winners inside the decode layer**. The fitted
overall model in “Boards” takes max(decode, component composite); standalone projection
improvement currently cannot change our overall rank. The earlier standalone 1.26x goal
and gap table below remain diagnostic evidence. Current scored best is `f3786341611f4f1f`,
**1.181998x** (the overall increment overlaps its CI; the router m256 gain is much larger). `submission.py` is its exact
frozen fused-select source, SHA256 `10d02fa32b572b1649f5e0c63e5f9841567835dc3b83ffe17cd7d843429117e9`.
The overall-board formula is unpublished; the old 11% projection weight is superseded
by the better-fitting max model in “Boards”.

| Finding | Result / scope | Mechanism or remaining limit | Evidence |
|---|---|---|---|
| Decode verify projection control | Union `215cafaf8d674c7e`, M128/M256 raw graph us: qkv **24.320/27.328**, pool_router **32.320/37.120**, unpooler **27.392/32.032**, o **13.920/15.680** | At M>64 `custom` GEMV delegates to cuBLAS, so those two choices do not compare independent kernels. FP32 pool_router is explicitly excluded from Lt. The inherited Lt source fails SM90a compilation without `ATen/cuda/CUDAContext.h`; optional rejection can leave a passing run with no Lt trials. Raw graph times include replay overhead and cannot be subtracted directly from pure byte/FLOP floors | Receipt `UNION_SUMMARY.projection`; `cand_r3_union.py`; `runs/screen/verify_proj/inherited_lt_failure.json` |
| Decode projection overlap, implementation dropped | Historical control 0e94268 has a 14.8 us decode-1 sliding attention window and an o bank that fits L2. The combined standalone/overlap lab b63389f0 ran on the withdrawn bf16moe host and hit a sticky `cudaErrorInvalidAddressSpace` in overlap | Root cause and safe native-Hopper stream ABI remain unverified. Deleted active `dev/overlap_lab.py`; historical generated sources/receipts remain evidence only. No overlap payload rides the new GEMV lab; revisit after GEMV | `runs/*/*b63389f01e4741b9.json`; projection FOCUS |
| Standalone GEMV bandwidth probe | Exact T1/8/16 qkv, o, pool/router and unpool shapes compare cublas, all returned Lt heuristics, incumbent split-K and 4x-CTA/8x-wave split-K with cold-L2 CUDA graphs; each emits achieved TB/s, CTAs and waves | Local sm120 is diagnostic only: qkv T1 cublas/incumbent/more-CTA **1.08/1.12/1.02 TB/s**, o **1.02/0.91/0.82**, pool/router **1.32/1.32/1.32**, unpool slice **0.68/0.68/0.59**, full unpool T16 **1.17/1.24/1.17**. H200 determines whether the low rate is kernel or dependency/tail overhead; no drop-in winner promoted yet | `problems/muse_decode_layer/dev/gemv_standalone_lab.py`, SHA `7e3d1a522c7edfe87d3f6fc15262267fe10310d095b6214edbae18dc09c227b6`; sent priority 0.8 |
| Decode verify Lt search landed | H200596b2a5d436e4fee, cold128MiB/fresh3/20-pair finalists: M128 qkv **21.104→20.224us** (NN23,paired ratio **0.9549**), unpooler **24.000→22.960us** (NN0, **0.9574**). M128 pool/router27.664→27.264us (0.9785), o11.408→11.168us (0.9853). M256 qkv23.248→23.168us (0.9979), pool/router33.248→32.960us (0.9942), unpooler27.968→27.808us (0.9881); o NN0 loses13.168→13.200us (1.0036), so retain incumbent | This revises the missing-search hypothesis: only small gains, not the large floor gap. Earlier union controls are slower than this run; do not count run-to-run clock/overhead differences as gains. All expanded candidates checked without rejection. H200 total prepare14.1s/lab4.3s. Embedded7-key shortlist retimes against the layer incumbent; full forced-recipe sm120 fixture13/13+12/12,4.877s, all10 static checks. Module SHA97c874743a23…; final formatting is AST-identical to the tested body and max267chars/line | `runs/muse_bf16_proj/*596b2a5d436e4fee.json`; `dev/verify/{projection_dropin.py,README.md,h200_recipes.py}`; `runs/screen/verify_proj/{integration_h200_fixture,block_admission}.json` |
| Verify projection search, authorized decision | Library596b2a5d436e4fee: best M128 qkv/unpooler gains4.5%/4.3%, others near parity. Corrected custom TMA v2 `92e7802783a54212`: M256 qkv23.49->37.73, pool33.39->65.55, unpool28.19->44.85, o13.34->20.51us; M12821.20->25.55/27.66->40.40/23.47->32.29/11.57->16.46. PDL chain loses1.32–1.62x | Custom v2 is a measured losing implementation, **not a closure of TMA/WGMMA**. Next is shipped CUTLASS Hopper cooperative split-K/Stream-K with timed workspace reset, tile/cluster/grid/operand choices at exact decode shapes, retaining the incumbent elsewhere. No judge polling: send, build the handoff, inspect receipts once per turn | `prof/lab/verify_proj_cutlass.py`, `problems/muse_bf16_proj/dev/cutlass/`; denseV11 remains open, glue parked, MoE grouped withdrawn |
| Owned NVFP4 weight-bank audit | Owned dense serving modules contain BF16 projections only. `cand_v11_gemm` retains packed expert bytes/E4M3 scales in prepare, expands in each timed expert call. Raw17 grouped path permutes NVFP4/E4M3 bytes; FP32 diagnostic reference is not returned in runtime state | No prepared BF16/FP8 expanded NVFP4 weight bank found. Inherited V9 prefix in the withdrawn/parked grouped candidate recodes scales to uint8 exponent/mantissa and folds sign/zero into FP4; exact but not a permutation of stored E4M3 bytes. The encoder is lossless and allowed (CONSTRAINTS 5) and is faster on H200 ("Scale bank" above); the MoE lane owns its incumbent | `problems/muse_moe_w4a16/dev/tma_grouped/README.md`; new CUTLASS block handles BF16 dense only |
| Stock CUTLASS verify recipe set closed | H200 lab45f58e7009c440eb:144 recipes/key, every choice loses.20-pair best M256 control/new: qkv23.792/31.024us (1.306x), pool/router33.648/41.856 (1.254x), unpool28.528/30.752 (1.082x), o13.312/15.664 (1.183x); M128 qkv20.416/27.200 (1.322x), pool28.352/35.072 (1.244x), unpool23.632/27.264 (1.142x), o11.728/13.920 (1.186x). The often-quoted1.07 unpooler ratio is coarse, not the20-pair finalist | Standard image CUTLASS Hopper cooperative TMA schedules, including split-K/Stream-K, multicast, swapped AB, rings/raster/grid/reduction choices do not beat cuBLAS/Lt at these exact M128/256 shapes. Close this **recipe set**, not the family. No native recipe is installed. Cold compile71.61s; lab80.64s; total prepare88.7s | `runs/muse_bf16_proj/20261024T063237Z_check_45f58e7009c440eb.json`; payloadSHAe43ed9b225d98cf1… |
| Verify weight-wide WGMMA (local admission; H200 pending) | New TMA kernel uses tokens on M64, weight rows on N64/128/256, 1–4 consumer warpgroups sharing 2–8-stage rings and one-launch split-K reduction. **744** local checks/**90** fresh graphs/**415** native descriptors; full lab **407** passing recipes at12 keys, zero rejections; forced decode **13/13 tests,12/12 graphs,15/15 static**; prefill host **5/5 tests,8/8 graphs,3/3 static**. Native15 kernels have no spills, max154 registers | Exact best6 `1b80ffc0` cuBLAS/Lt controls at M8/128/256, real loaded BF16/router weights,3 fixtures,20 cold rotated finalist pairs. Pooled capture streams cause a replay failure; distinct native streams fix the full reproducer (cuBLAS workspace lifetime is inferred). M128/256 dense bank is197.13MB/50.47GFLOP at256 (41.07us read floor,63.08us at800TFLOPS);138.41MB applies to M8's unpool slice. No prepared NVFP4 expansion; no H200 winner or dispatch installed | `problems/muse_bf16_proj/dev/verify_streaming/README.md` and JSON admission; payloadSHA`7b67602f1d4b77c0…`; merged with proj glue on prefill, send key **be5d6aa5ae1ac0bd**, priority0.5; detached log reports waiting for a slot, no rejection |
| Verify weight-wide WGMMA synthetic MoE-host check | Regular tiled split-K TMA/WGMMA (not a balanced Stream-K scheduler), projection SHA `34bb98cc52792155`; synthetic real-shape BF16 banks, cold 256MiB graphs. Local 5 MoE tests/6 static and 744-check admission pass. | H200 job `19f3a13b17b44b6f` loses at every role: summed137.4/159.9 us vs incumbent85.3/100.4 at M128/256. No best7 projection promotion; the decode-lever row above owns the result. | `prof/lab/verify_streaming_synthetic_moe.py` SHA `294cceb1593352c09`, detached key `120effb2432ea635` |
| GEMV/glue fusion: H200 implementation closed | H200 lab `479aab3038c6407c`: every fused stage, prefix and chain lost. Ratios were b1 **1.06–7.8x**, pool **2.04–11.36x**, b2 **1.09–9.02x**, PREFIX **1.88–9.74x**, and chains **1.04–9.60x** slower; `picks=0` | Folding norms, residual gates and routing into these skinny GEMVs costs the GEMV's parallelism and does not recover the nominal 12–25us glue. Close this implementation, not the general idea of eliminating glue launches; keep the service lane free to revisit it with a fused vector/consumer design | `runs/muse_bf16_proj/*479aab3038c6407c*.json`; `dev/glue/projection_glue.py` |
| Corrected native projection local admission | **sm120**:2148 fresh checks,342 executable configs per output dtype; all8 target keys. Full Hopper lab forced73 tile/ring/K-group families,1247 calls and4 PDL chains; injected AttributeError rejects. Lab host9/9 tests+30/30 graph cases,2.091s cached prepare; full-layer fixture13/13+12/12,3.264s. Shortlist32 trials+140 fresh/stride/weight cases and expired/short/module rejection pass; all SM90a static builds pass | Local reader uses mma.sync and ordinary TMA as the multicast transport stand-in; actual native multicast also passed24 target cases but is extremely slow on sm120. Isolated MC1 controls are68–80us versus MC2~1.9–2.7s at M128 pool/router; wait/scope changes do not fix that timing. NVIDIA documents multicast as optimized for sm90a. No inference of Hopper latency from this local penalty | `runs/screen/verify_proj/{grouped_validation_v2,grouped_lab_glue_v2,cluster_contrast,poll_probe}.log`; `integration_fixture.json`, `lab_tma_v2.json`; [CUDA TMA guide](https://docs.nvidia.com/cuda/archive/13.1.1/cuda-programming-guide/04-special-topics/async-copies.html) |
| Persistent projection barrier lifetime | Native v1 `6b72087f64584258` reached the judge before cancellation and is **invalid for dispatch selection**: persistent tiles reinitialize valid barriers. Local numerical passes did not expose that contract violation. Scored kernel_fn remains the frozen incumbent | V2 invalidates full/empty barriers after cluster/CTA users finish, before reinitialization, and uses cluster-scope acquire for remote releases. The [PTX lifetime requirement](https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-init) establishes the defect; it does not explain the sm120 multicast timing penalty (invalidation alone did not fix it). No execution cancellation API exists; the result reader rejects v1 | `dev/verify/build_tma.py`; `runs/screen/verify_proj/barrier_lifetime_probe.log`; corrected v2 admission above |
| Decode projection port | `cand_r3_proj.py`, SHA256 `235434b35d1407cdb61ff79f291b9a962fa4a56369f60277fb666863f5b1174e`; 24 projection keys at T1/8/16/32/128/256, including the layer's FP32 4608-row pool/router | Skinny/GEMV winners, mixed pooler/router with hi+lo in the same FP32 accumulator, strided pooler Lt + fused router, TMA/WGMMA direct/last-CTA/DSMEM/Stream-K, and all checked Lt TN/NN/split-K algorithms. Trials time the next consumer, validate two fresh inputs within 1% of cuBLAS and reject/log failures; FA3's 27 settings and PDL remain tunable under a shared 240 s deadline | `problems/muse_decode_layer/cand_r3_proj.py`; projection-owned `dev/build_decode_proj.py` and embedded templates |
| Port admission, **sm_120 only** | **13/13 tests, 12/12 fresh-input graphs**, **29.05 s prepare**, no trial rejections; **2,742** native projection configurations ×3 fresh replays; **730** forced Hopper ABI cases, **16** injected rejections; sm_90a projection build passes | Includes strided/zero-tail outputs, FA3 tensor/tuple returns, q_len 1/8 split-qkv tuples and expired budgets. Local success cannot verify Hopper WGMMA execution, H200 ranking or cold H200 prepare time | `runs/screen/decode_proj/muse_decode_layer.cand_r3_proj.json`; `problems/muse_bf16_proj/dev/decode_proj_{validation,sm90a}.log` |
| Full-layer control overrides isolated MoE ranking | Pure V9 **1.427904x**, CI1.406485–1.449322 (`0e94268cf5654012`), versus grouped dispatch **1.265668x** (`2ad75c19cb514f5c`), with a verify32 collapse | New H200 evidence landed during the port. Keep V9 at every token count for this projection experiment and skip the unused V7 bank; the inherited synthetic MoE tuner can select a path that loses in the real layer | `runs/muse_decode_layer/*0e94268cf5654012.json`, `*2ad75c19cb514f5c.json`; full-layer profiles |
| Generated tuner rejection/budget handling | All/max/native builders now inherit coarse/finalist/layout rejection and a 240 s trial deadline; v2 template also rejects failures and stops trials | Fault injection rejects launch, numeric, late finalist and layout failures on all 30 standalone projection shapes; expired budget skips all shapes. Sent `cand_portfolio_all.py` stays untouched at the owner's patched `0caf7322…` hash | `problems/muse_bf16_proj/dev/validate_tuner_safety.py`, `tuner_safety_validation.log`; `build_portfolio_all.safe_tuner` |

<!-- PROJ_GAP_BEGIN -->
Historical standalone H200 gap to 1.26x, from all 14 clean completed receipts. SGLang and board
times use scored `f3786341611f4f1f`. Best = that SGLang time / the highest observed
paired speedup for this shape; it is a common-baseline equivalent, not a new measured
median. Choosing the maximum across receipts is optimistic and includes timing noise;
the new portfolio must reproduce any gain. All times below are microseconds.

| m | Layer | SGLang | Board | Best equivalent | Best receipt median | Receipt | Excess over SGLang/1.26 |
|---:|---|---:|---:|---:|---:|---|---:|
| 1 | qkv_gate | 14.208 | 12.833 | 12.768 | 12.566 | `68e2f34b1d4744cb` | 1.491 |
| 1 | o | 6.880 | 6.888 | 6.670 | 6.608 | `a3b29a26372c4b35` | 1.209 |
| 1 | pooler | 21.440 | 18.905 | 18.595 | 18.761 | `9294b3237c4f4e2d` | 1.579 |
| 1 | unpooler | 7.104 | 5.674 | 5.447 | 5.398 | `9294b3237c4f4e2d` | 0.000 |
| 1 | router | 10.304 | 6.356 | 6.109 | 6.005 | `dd5ffc6950bc439e` | 0.000 |
| 16 | qkv_gate | 14.384 | 14.210 | 13.818 | 13.449 | `9294b3237c4f4e2d` | 2.402 |
| 16 | o | 6.784 | 6.784 | 6.652 | 6.872 | `9294b3237c4f4e2d` | 1.268 |
| 16 | pooler | 21.824 | 19.833 | 19.833 | 19.833 | `f3786341611f4f1f` | 2.512 |
| 16 | unpooler | 18.640 | 18.680 | 18.410 | 18.599 | `849bd4b650e54c7f` | 3.616 |
| 16 | router | 14.976 | 7.043 | 6.515 | 6.508 | `849bd4b650e54c7f` | 0.000 |
| 32 | qkv_gate | 14.688 | 14.682 | 14.473 | 14.079 | `a3b29a26372c4b35` | 2.816 |
| 32 | o | 6.912 | 6.823 | 6.656 | 6.533 | `9294b3237c4f4e2d` | 1.170 |
| 32 | pooler | 22.224 | 22.142 | 22.042 | 20.772 | `849bd4b650e54c7f` | 4.404 |
| 32 | unpooler | 19.024 | 19.033 | 18.845 | 18.544 | `dd989ef2d76147e7` | 3.747 |
| 32 | router | 15.520 | 8.207 | 7.987 | 7.896 | `849bd4b650e54c7f` | 0.000 |
| 8 | qkv_gate | 14.208 | 12.528 | 12.528 | 12.528 | `f3786341611f4f1f` | 1.251 |
| 8 | o | 6.688 | 6.624 | 6.534 | 6.440 | `dd989ef2d76147e7` | 1.226 |
| 8 | pooler | 21.824 | 18.850 | 18.850 | 18.850 | `f3786341611f4f1f` | 1.529 |
| 8 | unpooler | 7.008 | 5.439 | 5.362 | 5.509 | `dd989ef2d76147e7` | 0.000 |
| 8 | router | 26.240 | 6.399 | 6.098 | 6.031 | `dd989ef2d76147e7` | 0.000 |
| 128 | qkv_gate | 17.152 | 17.001 | 16.501 | 16.347 | `dd989ef2d76147e7` | 2.888 |
| 128 | o | 7.296 | 7.310 | 7.208 | 7.477 | `a3b29a26372c4b35` | 1.417 |
| 128 | pooler | 25.264 | 24.336 | 24.336 | 24.336 | `f3786341611f4f1f` | 4.286 |
| 128 | unpooler | 19.952 | 19.232 | 19.232 | 19.232 | `f3786341611f4f1f` | 3.397 |
| 128 | router | 16.592 | 10.133 | 10.133 | 10.133 | `f3786341611f4f1f` | 0.000 |
| 256 | qkv_gate | 19.584 | 19.745 | 19.061 | 19.746 | `122997f357d04fa2` | 3.518 |
| 256 | o | 9.824 | 9.701 | 9.693 | 9.409 | `458c6d3e931c4f63` | 1.897 |
| 256 | pooler | 30.592 | 29.714 | 29.714 | 29.714 | `f3786341611f4f1f` | 5.434 |
| 256 | unpooler | 24.512 | 24.247 | 24.247 | 24.247 | `f3786341611f4f1f` | 4.793 |
| 256 | router | 17.792 | 11.792 | 11.792 | 11.792 | `f3786341611f4f1f` | 0.000 |

The last column is a sufficient uniform-shape target, not the score formula. The
official projection score is the geometric mean of six *load* ratios; each load sums
its five equally weighted projections (weight 62 cancels). The overall-board formula
remains unpublished; the earlier 11% projection weight is a model, not a rule.

| m/load | SGLang sum | Board sum | Best equivalent sum | Best load ratio | Budget at 1.26x | Saving to that budget |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 59.936 | 50.656 | 49.588 | 1.209x | 47.568 | 2.020 |
| 16 | 76.608 | 66.550 | 65.227 | 1.174x | 60.800 | 4.427 |
| 32 | 78.368 | 70.886 | 70.003 | 1.119x | 62.197 | 7.806 |
| 8 | 75.968 | 49.840 | 49.371 | 1.539x | 60.292 | 0.000 |
| 128 | 86.256 | 78.012 | 77.411 | 1.114x | 68.457 | 8.954 |
| 256 | 102.304 | 95.199 | 94.507 | 1.083x | 81.194 | 13.313 |

The optimistic stitched score is **1.197544x**, versus the board's
1.181998x. If the other four loads hold, m128 and m256 each need
another **1.165x** speedup to reach 1.26x overall.
Starting from the actual board rather than optimistic stitching, that factor is **1.211x**.
The largest uniform-budget deficits are pooler/unpooler at m16/32/128/256, then qkv and
other BF16 layers. Router fusion alone cannot close the complete gap; tune all 30
shapes, especially memory-heavy BF16 projections. Receipt/source pairs in `runs/`
own these measurements; profiles diagnose kernels but do not replace paired medians.
<!-- PROJ_GAP_END -->



| Finding | Result / scope | Mechanism or remaining limit | Evidence |
|---|---|---|---|
| GEMV receipts landed | `cand_gemv` 1.112106x (`a3b29a26372c4b35`), `cand_gemv2` 1.114312x (`122997f357d04fa2`), both 9/9 | M1 GEMV loses qkv/pooler/unpooler to the incumbent; o M1's best normalized improvement is about 1%, inside run noise; retain the incumbent there | `runs/muse_bf16_proj/*a3b29a26372c4b35.json`, `*122997f357d04fa2.json` |
| Pooler m32 dispatch | Incumbent skinny 0.930x in scored run; cuBLAS 0.991/0.996x in the GEMV checks | The GEMV candidates route this shape to cuBLAS; the portfolio folds that per-shape winner into its incumbent dispatch | Same receipts and `cand_gemv*.py` |
| cuBLAS split-K overhead | qkv m128: main 12.5 us + reduction 3.1 us (20%); router m128: two mains 9.0 us, two reductions 3.9 us and add 1.1 us (36% outside mains) | Split-K supplies parallelism but requires another graph node/global partial round trip; fuse the merge, and accumulate router hi+lo in one product | `runs/calibration/profiles/muse_bf16_proj_sglang.458c6d3e931c4f63.txt`, profiles 20, 24 |
| Wave quantization | Interpreting cuBLAS kernel tile names as weight-rows × tokens gives unsplit qkv m128/256 48/96 output tiles, o 128/128, pooler 128/128, unpooler 128/128 | **Inference**, not measured CTA occupancy: 48/96/128 tiles are 0.36/0.73/0.97 of one 132-SM wave; qkv m128 actually splits K; the others run long K loops without a second full wave | Same profiles 20–29; [NVIDIA Hopper occupancy/cluster guidance](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html) |
| Roofline changes with m | At m128 all four BF16 projections have HBM floors; at m256 qkv/pooler/unpooler become tensor-bound while o remains narrowly HBM-bound | **Model**, using compulsory bytes and 989 TFLOP/s / 4.8 TB/s. W-only bandwidth cannot explain the entire m256 gap: tensor utilization and wave tails also bind | `docs/reference/bound.md`, scored receipt floors; `reference.speed_of_light_ns` |
| Cold L2 and attainable bandwidth | W-only effective rates at m128/256: qkv 2.22/1.88 TB/s, pooler 2.73/2.26, unpooler 3.40/2.81; all below the measured 4.25 TB/s stream | **Calculated** from scored SGLang medians, not memory counters. Cold L2 removes inter-replay reuse; repeated M tiles can still reuse weights within a replay. The 67 MB pooler/unpooler exceed H200's 50 MB L2; qkv/o fit in capacity but still start cold. Compute, scheduling and launch cost also matter, so this does not isolate cache misses | Prior scored `68e2f34b1d4744cb`; sweep2 bandwidth probe; [NVIDIA Hopper architecture](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth) |
| Router WGMMA lead | Old prepare diagnostic: 11.55 us at m128 and 18.46 us at m256 for BT128/256, BR64, ST4, cluster split16 | Measured diagnostic timings, not scored shape speedups; the old dispatch did not select its measured winner | `profiles/cand_diag4.dd5ffc6950bc439e.log`, `cand_diag4.py` |


| Current evidence | Result / scope | Consequence | Source |
|---|---|---|---|
| Portfolio H200 router win | m128 **10.327 us, 1.583x**; m256 **13.434 us, 1.327x**. Both choose BT128/BK128/ST4, DSMEM split16/8 | Real Hopper WGMMA execution and gain are now verified for the original doubled-K router. Keep these exact kernels eligible alongside real-K hi+lo fusion | `runs/muse_bf16_proj/*dd989ef2d76147e7.json`; `profiles/cand_portfolio.dd989ef2d76147e7.log` |
| Portfolio BF16 choices | qkv m128/256: Lt ordinal3/4; pooler 0/0; o m128 0. Unpooler and o m256 retain cuBLAS | Large BF16 projections remain near parity. Qkv m256 selected a prepare-time winner but scored **0.982x**; synthetic tuning can choose a regression | Same receipt and prepare log |
| New fused kernels | Router stages hi+lo together for each real-K tile into the same FP32 accumulators. Router **5 -> 1 kernels**, qkv **2 -> 1** with an in-kernel last-CTA/DSMEM merge, at both m128/256 | The router is now H200-verified at **10.133/11.792 us (1.637/1.509x)** with one kernel in profiles24/29 of `f3786341611f4f1f`. Qkv custom paths lose in prepare and cuBLAS remains selected. Hierarchical K32/K64 merges distribute the last cluster's work across 16 resident CTAs | `dev/cand_reduce*_kernel_counts_local.json`; standalone frozen hashes/screens in NEXT_CHECKS |
| All-shape portfolio | `cand_portfolio_all.py`: **1.117166x sm_120**, 9/9 tests, 30/30 fresh-input graph shapes; **122.66 s cold prepare**, 65.53 s compiling/repacking. **9,912** forced Hopper cases; sm_90a build passes | Tunes all 30 shapes, preserves measured H200 winners, and compares NN/TN layouts, every returned cuBLASLt heuristic, checked explicit split-K, direct/last-CTA/DSMEM/Stream-K, 128/160/256-thread fused paths, skinny/GEMV/TC/scalar-m16/DSV3 kernels | `runs/screen/proj_portfolio_all/muse_bf16_proj.cand_portfolio_all.json`; `dev/portfolio_all_{glue,sm90a}.log` |
| Full superset | `cand_portfolio_max.py`: **1.120767x sm_120**, 9/9 tests, 30/30 graph shapes; **71.77 s prepare with a cold extension build** (38.92 s compiling/repacking), independently checked with a new extension directory and ccache disabled; image-library JIT caches may already be warm. Adds 22 shared-X tile and 32 multicast WGMMA kernels, plus the default DeepGEMM SDK call (cuBLAS dispatch) | These are distinct methods, not a claim that 1.1208x beats 1.1172x on H200. Both files keep the incumbents eligible. All 189 skinny/paired/legacy/tile/wide specializations compile for sm_90a; **12,265** forced Hopper cases and **72** native TMA/MMA wide-kernel fresh-input graph cases pass | `runs/screen/proj_portfolio_max/muse_bf16_proj.cand_portfolio_max.json`; `dev/portfolio_max_{glue,sm90a}.log` |
| Native DeepGEMM superset | `cand_portfolio_native.py`: **1.111248x sm_120**, 9/9 tests, 30/30 graph shapes, **63.91 s prepare with the extension cache warm**; **12,281** forced Hopper cases; sm_90a build passes. CUDA bodies match max exactly | Explicitly calls `use_deterministic_algorithms(True)` to request native BF16 DeepGEMM, retains explicit SDK cuBLASLt as a separate variant, and confirms `deep_gemm::sm120_bf16_gemm_impl` in GPU profiling. The stock projection dispatcher was checked to use cuBLAS/its own kernels, so that documented SDK setting does not change the baseline | `runs/screen/proj_portfolio_native/`; `dev/portfolio_native_{glue,sm90a}.log`; `dev/portfolio_native_dg_kernels_local.json` |
| Latest-winner superset | `cand_portfolio_h200.py`: **1.102951x sm_120**, 9/9 tests, 30/30 graph shapes, **39.81 s prepare with the extension cache warm**, **12,286** forced Hopper cases and an sm_90a build | Includes native DeepGEMM and both receipts’ H200 winners. Latest 32 MB Lt NN choices are preserved exactly; old and new winners survive into final paired tuning. CUDA bodies match native/max | `runs/screen/proj_portfolio_h200/`; `dev/portfolio_h200_{glue,sm90a}.log`; `RECENT_H200_WINNERS` in the source |
| Tuning protocol and resources | 128 MB L2 flush, CUDA graph replay, order-swapped paired ratios, best-per-family and global finalists, three fresh synthetic input sets including probes/outliers; 512 MB Lt workspace, 32 MB legacy workspace, weight transposes, router hi/lo/paired layouts, per-shape scratch | No precision is reduced, no benchmark-input result is cached, no math moves outside timed calls. A final compact dispatch log retains all 30 choices in the judge's 16K log tail. CUDA/Lt algorithm configurations are checked through the [NVIDIA cuBLASLt API](https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmulalgocheck) | `cand_portfolio_{all,max,native,h200}.py`, builders/ABI in `dev/` |
| Small-m and diagnostics | The unmeasured fusion/layout diagnostic sources and 13 inherited sources remain eligible; their hashes/queue states are in NEXT_CHECKS. Existing wg/wg2 never set `wg_ok`; the new superset directly enables the multicast kernels | Fused-select now covers the same GPU bodies/configurations as fused_lean and fused; those queued duplicates need no further send. This round has no diagnostic hold. Do not claim WGMMA measurements from the queued wg/wg2 bodies; use the superset's explicit active branch. Reuse a completed receipt rather than sending duplicate GPU work | `dev/NEXT_CHECKS.md`; `cand_wg{,2}.py` |
| P8 hardware probe | `849bd4b650e54c7f`, clean **1.136951x**, uses an earlier projection dispatch. The [P8 section](#p8-hardware-probes-on-h200-2026-10-24-kst) owns its microbenchmark values | Retain small-m MMA winners. Test CTA coverage and flight depth; TMA is useful where a tile is reused. The probe has no N64–256 tensor measurement, and its cache/PDL results are not projection wins | Same receipt and labelled profile log |

New portfolio numbers are local sm_120 evidence; H200 score and prepare duration remain
unverified until their receipts land. The board source is H200-verified at 1.181998x;
no 1.26x result is claimed. `dev/gap_table.py --write` regenerates the receipt-derived table.

### Attention verify development (G2; evidence scopes distinguished)

### Decode attention drop-in (2026-10-24)

**FP8 split decode attention loses (`366726675b5a42b2`, q1, ctx 80K):** vs our control attn_main (and FA3): B1 sliding 14.5 vs 14.7 (25.5), B1 full 25.0 vs 23.6 (38.9), B16 sliding 18.2 vs 17.0 (32.9), B16 full 237.8 vs 118.9 (208.8), B32 sliding 22.0 vs 21.2 (36.5), B32 full 388.1 vs 226.0 (391.9) us. The control beats FA3 everywhere at q1; at full context it is ~5.5x the KV floor at B1 and ~1.8-2x at B16/B32.

**FP8-PV verify variants are accurate but 3-15x slow (`948044d4f32a4736`):** settings (3|5, 1|2) pass the gate (0.093-0.097) at b16 sliding 90.9-101.7, b16 full 1,253-1,294, b32 sliding 117-131 and b32 full 2,144-2,372 us vs FA3 ~372 us at b32 full; about 15x the b32-full KV floor (139 us). Implementation suspect, not closed. The same run's M=1 MoE path (v9m1) failed agreement on the H200 (1.889), like NT4 in `3920d3f0`; both passed on sm_120.

**Our q8 verify kernel loses and fails at full context (`b02a42ce75dc44d4`, phase clocks, ctx 80K):** b16/b32 sliding q8 81.6/88.1 us (gate pass) vs q1 16.7/21.1; full q8 412.5/761.1 us and the post-store FP32 gate fails (1.252) vs FA3's ~372 us at b32 full. Its per-tile combine phase dominates (38-44K cycles median vs 0.6-1.4K for QK/softmax/PV). Closed for this kernel; the FP8 split variants (labs 948044d4, 5068f5e6) replace it.

`problems/muse_decode_layer/dev/cand_r3_v9_attention_dropin.py` is a generated drop-in from the
v9-derived attention portfolio. Its prepare-time plan is keyed by all twelve decode-layer shapes:
decode/verify batch 1/16/32, q_len 1/8, sliding/full. It admits the layer path, FA3 split/pack,
page-gather/XQA families, and verify kernels behind the post-store quantized-KV FP32 reference gate.
The builder and structural test are `problems/muse_decode_layer/dev/attention_dropin.py` and
`problems/muse_decode_layer/dev/test_attention_dropin.py`.

Local sm_120 admission is complete: 13/13 tests, undefined-name checks, and all 14 sm_90a CUDA
source compilations pass. Generated source SHA256 is
`bc86dd5f2fc0807cba9244b217f0e433c10beaeb48516eddf004e8c00f39ad8c`, with 8.95 s prepare.
The second H200 sweep `59e53e746df341a9` again marked every variant F, FA3 included, so its gate is still wrong; its timings (median of 3, ctx 80000, page 1, agentic spread) are valid. Decode q1: attn_main beats the best FA3 split ~1.9x at every shape (b16 sliding/full 17.2/108.0 us vs 31.5/205.6; b32 22.1/199.7 vs 35.9/386.1). Verify q8: FA3 wins every shape (b16 33.7/208.4, b32 39.5/389.7 us) over attn_main (52.8/234.9, 55.9/419.8) and every XQA/stream/WGMMA variant (xww >=55 us, xgg/xp/fs/wg >=91 us at b16 sliding). FA3's q8 full times equal its q1 times, so verify is KV-read bound; a q8 kernel reading KV as attn_main does could reach ~110/200 us at b16/b32 full. That kernel matters for serving too, which runs only verify and prefill. The first sweep
(`87a05615b39e438c`) had a faulty FP32 gate and is not correctness evidence, but its timings are
valid diagnostics: q1 B16 sliding attn_main 16.9 us versus best FA3 31.5 us, and q1 B16 full
attn_main 107.6 us versus best FA3 205.7 us. B16 q1 sliding/full are therefore the first
provisional drop-in changes; the other ten choices remain H200-unverified pending the corrected
receipt.

The FP8 WGMMA verify source had dead branches referencing `_vmath_verify` and `_vmath_twog_verify`;
those branches are deleted. Its local 16/16 and static checks pass. Verify32 full H200 selection
remains pending attention candidate receipt `05fe4997`.

The forced `port_verify` path was run through the complete layer ABI locally with the post-store
FP32 gate: B16 full passed at 0.0058 relative L2 and B32 full at 0.0072. The H200 comparison lab
`prof/lab/verify8_attn_compare.py` is queued on the proj host; it measures B1/16/32 sliding/full
against FA3 and records the gate per variant.

The first q8 comparison receipt (`59e53e746df341a9`) marked every row F because its lab reference
flattened the 16 heads before RMS normalization and its B>1 agentic page table stopped at 80,008
rows. The reference now follows the layer's per-head 64-wide post-store FP32 gate, and the lab
allocates 86,008 rows for the +/-7.5% agentic spread. A local replay of the same control gives
relative errors 0.0068 (B1), 0.0062 (B16), and 0.0058 (B32). The packed port's Hopper path now
uses a four-stage KV ring (three-stage sm_120 stand-in), with `_splits` targeting two waves of
the 132 H200 SMs: 17 splits at B16 and 9 at B32 before cluster combine. The replacement lab is
queued at priority 0.8 with payload SHA256 `2adee96d8f72588134aa1d21520f051b8c451906ed8b34cc5e82b15d33775dde`;
its H200 timings and winner choices remain unverified.

The exact attention-work bound from `sim/bound.py` is 332,804,224 bytes for B16 q1/full and
332,833,344 bytes for B16 q8/full: 69.33/69.34 us at the 4.8 TB/s H200 specification. B32 is
665,608,448/665,666,688 bytes, or 138.67/138.68 us. The model uses HQ=16, HKV=2, head_dim=64,
and one-byte FP8 K/V, so q8's extra work is compute/reduction rather than another KV read.

The replacement payload now records clock64 phase samples for q1 and q8 at B16/B32 full with
the packed two-wave split target (17/9 splits): KV wait, QK MMA, softmax, PV MMA, and split
combine. Its local sm_120 profile path runs for both q_len values and all five CUDA sources pass
sm_90a compilation. The phase lab is queued at priority 0.8 with payload SHA256
`1d932e7e164ecf497c32a6c77a3589a17d5a2a0450324e5361657e575afa3055`; the dominant H200 phase
and the corresponding kernel change remain unverified.

Receipt `0467893d28a84f53` isolated the failure: the two-wave q8 plan's odd 17/9 split counts
made the Hopper q8 reduction path fail its gate at F1.252, while q1 passed. Combine was 44,096
cycles versus 7,678 for q1. The q8 plan now uses one resident wave with floor split counts
8/4 at B16/B32, reducing combine traffic; local sm_120 gates pass at 0.00616/0.00583 and the
full decode drop-in passes 13/13 tests plus 15/15 static checks. A replacement full+sliding
phase lab is queued at priority 0.8 with payload SHA256
`8047ce325de2bfbf63e0305f0ee59c09aaecd0626d12aa3b3685aeb818b72e6e`; its H200 correctness and
timings are the only remaining verification.

For the next full-layer attention trial, the existing raw-FP8 probability/PV path is now gated
through the decode-layer ABI at verify B16/B32 full for waves 1 and 2: all four local cases pass,
with attention-input relative L2 0.0385--0.0403 and update relative L2 0.0667--0.0698. It uses
FP8 QK, FP8 probabilities/V, split-K over the full context, and one FP32 combine. Payload
`prof/lab/verify8_attn_compare.py` (SHA256 `dbb0b99e8d16d51d71756e5ae9cb6f6c2e67e6c18340400728e37a8cecab0182`)
is queued on the off-quota prefill host at priority 0.6 under send key `dee678ffa7175feb`.
Promotion requires each setting to pass the FP32 gate and beat FA3 at verify-16 and verify-32
full without regressing the q1/q8 control rows; otherwise FA3 remains selected for G0.

The landed diagnostic profiles contain only the selected FA3 path: full B16 main189.2us of206.7us total, B32 main373.0us of391.2us; combine4.4us each. They do not profile the losing custom variants. Their XQA prepare timing lines provide custom totals. The nominal P8 gather-only times for327.68/655.36MB are128.1/256.2us, versus FA3 roughly207/391 and K128617/1174. These comparisons establish unused throughput, not a measured TLB or tensor-pipe cause.

| Finding | Result / scope | Mechanism or remaining limit | Evidence |
|---|---|---|---|
| Eight-warp K128 swap-AB | Exact source: 16/16 tests, 12/12 graph shapes, three replays and two input sets; final shared-GPU full verify B16/B32 570/802 us | 64 queries on MMA N, keys on M; raw E4M3 QK; FP16 or FP8 PV; 3/4/5-stage page-row gathers; last-CTA merge; separate K32 causal tail | `problems/muse_attention_decode/dev/xqa_w8_final_local.json`; frozen `b519462266ffbfd9379979cd59891fecbc6b12c9f563d9c3ba8009a0efa4730f` |
| K128 register pressure | All six sm_90a stage/PV specializations have zero register spills | Eight warps replace the spilling four-warp K128; FP16 stage 5 needs 104 KiB shared memory and is exercised locally through a stage-3 stand-in at the same Python ABI | `problems/muse_attention_decode/dev/xqa_w8_sm90_compile.log`, `xqa_w8_glue.log` |
| K128 correctness beyond default dispatch | Both PV formats pass three-seed graph replays through 1M context; all five locally fitting stage/PV combinations pass split extremes, short contexts, window edges and DP heads | Exact 2048-token window, causal new keys, page1/page64 and non-unit KV scales; FA3 tensor/tuple and tuner branches forced with stand-ins | `problems/muse_attention_decode/dev/xqa_w8_formats.log`, `xqa_w8_variants.log`, `xqa_w8_glue.log` |
| Packed dispatch | K128 prepare sweeps 24 configurations, then pairs its fastest custom variant with FA3 in alternating order; custom selected above 1.03x | Synthetic tuning only; no input-derived state survives. H200 outcomes are pending; current submission remains the hybrid above | `cand_xqa_k128.py`; queue identities and frozen hashes in `problems/muse_attention_decode/dev/NEXT_CHECKS.md` |
| K128 stage profile | On sm_120 at full B32, clock64 averages per tile were gather issue/address 2.1k cycles, QK 1.2k, softmax 1.2k, FP16 PV 1.1k and FP16 widening 0.9k; `cp.async.wait` was 13 cycles | ncu cannot access counters (`ERR_NVGPUCTRPERM`); cross-SM merge clock marks are not a valid latency measure | `problems/muse_attention_decode/dev/k128_stage_profile_local.json`, `k128_stage_profile.log` |
| Split-KV occupancy | 24 splits were best locally: B16 waves 4 measured 405 us and B32 waves 8 measured 773 us; one split was 562/1,095 us | The 132-SM target is already oversubscribed at 24 splits (768/1,536 CTAs for two KV heads); merge is not the dominant measured cost | `problems/muse_attention_decode/dev/k128_stage_profile_local.json`, `xqa_pipeline_sweep_local.json` |
| Full-specialized pipeline | K128 with compile-time full/sliding RoPE selection measured 355/667 us at full verify B16/B32 versus old K128 406/773 us on the same sm_120 sweep | K128 3-stage FP16 path won; K64 and FP8-P variants lost locally; this is not an H200 claim | `problems/muse_attention_decode/dev/xqa_pipeline_sweep_local.json`, exact candidate `b56b70c299139071298edb1bfac7ef3768c5ac8d73bce312822410184b01b43a` |
| Gather follow-up | Register row-base offsets pass 16/16 correctness but measure 358/670 us versus pipeline 355/665 us at B16/B32; shared row-base offsets measure 407/781 us | Extra shared synchronization and row-base traffic erase the address arithmetic saved; integrated cp.async.bulk ring hangs on q8 despite standalone bulk/mbarrier success | `problems/muse_attention_decode/dev/xqa_gather_compare2.log`, `runs/screen/attention_gather2/muse_attention_decode.cand_xqa_gather2.json`, `xqa_gather_tests.log`, `test_bulk_run.py` |
| Landed XQA H200 diagnostics | K64 `32c5b4b287c24c38` 1.390910x overall, CI 1.37005–1.41519; K128 `a0511c0001464dbe` 1.382808x, CI 1.36207–1.40730; both select FA3 at all six verify shapes | K128 paired custom/FA3 full B16 616.74/207.36 us (0.3364x), B32 1173.81/391.36 (0.3334x); sliding B1/16/32 0.683/0.565/0.508x. Deeper stages 3–5 barely move full times. Pipeline receipt remains pending | `runs/muse_attention_decode/*32c5b4b287c24c38.json`, `*a0511c0001464dbe.json`; profiles' XQA_DIAG/XQA_W8_PAIR lines |
| Promoted exact source | `submission.py` is the exact `32c5b4b287c24c38` source, SHA256 `610a590120ae79c446248cce5b6a857e283b425290aba67af29e18283c96c810` | Owner's >1.388x promotion criterion is met, but this is FA3 parity on verify and its overall change is inside noise. Previous scored board entry remains 1.384x until orchestrator submits | Exact receipt/source SHA match; `runs/screen/attn_diag/diag.json` 16 tests +12 graph shapes |
| Producer/consumer swap-AB candidate | `cand_xqa_next.py`: K128 eight-warp MMA plus K64 WGMMA, each with 128 separate producer threads, full/empty mbarriers, 32–64 KiB gathered KV in flight; unchanged q_len=1 | FP32 softmax, exact window/causality, last-CTA FP32 merge. WGMMA QK uses E4M3, PV exactly widened FP16; same-kernel MMA stand-in runs locally. Per-shape synthetic prepare times 16 variants plus FA3 and checks agreement before selection | `dev/build_xqa_next.py`, `dev/xqa_group.cu`, `dev/xqa_next_tail.py`; frozen hash/complete admission in `dev/NEXT_CHECKS.md` |
| Hopper binary/layout validation | Group family uses 180–181 registers; producer MMA 153–156. Eight SM90a specializations have zero spill loads/stores; SASS contains QGMMA E4M3, HGMMA FP16 and LDGSTS | All 4096 coordinates each of K/V/P match installed CUTLASS canonical atoms; K descriptor LBO1/SBO32 B64, V LBO0/SBO64 B128, P LBO1/SBO64 B128. Stand-ins cannot verify Hopper execution | `dev/xqa_group_sm90_compile.log`, `xqa_warp_sm90_compile.log`, `xqa_group_sm90_sass.log`, `check_wgmma_layout.cpp`, `wgmma_layout.log` |
| Local producer comparison | Same sm_120 sweep: full B16/B32 producer MMA 355/676 us versus synchronous pipeline 339/660; sliding B1/16/32 29.7/39.9/54.2 vs30.7/42.0/56.3 | Producer overlap alone is roughly neutral on full contexts. Warpgroup stand-in has extra cross-warp softmax/barrier costs and is slower; it cannot rank Hopper WGMMA | `dev/xqa_warp_bench.json`, `dev/xqa_next_bench.json`; local only |

| Verify admission gate v2 | `cand_attn_verify_stream2.py` computes a local FP32 dequantized-KV reference with the same causal/window mask and output gate, then computes the FP8-E4M3-query reference error and judge bound. Each FA3/XQA/stream choice is accepted only at <=0.8x bound and <=0.8x the 5x row bound; exceptions are logged and rejected. | This removes the old FA3-vs-FA3 15% proxy, which could accept a coherent-case failure. The tuner uses a compact coherent q8 input (ctx 2048) because the landed coherent q8 case has the tightest bound. | `cand_attn_verify_stream2.py`, local screen 16/16 tests + 12/12 graphs; H200 unverified |

| Verify versus decode KV bandwidth | At B32, 80K, full context, q_len=8 reads the same KV bytes as q_len=1: the q1 stream is 196.5 us while verify attention is 391.8 us; the full verify layer is about 980 us, with roughly 480 us outside MoE. | The extra verify cost is attention math/scheduling over the same bytes, not a larger KV transfer. A q_len-8 stream at q1 gather bandwidth would therefore be the relevant win target. | q1/verify receipt notes; `problems/muse_attention_decode/cand_attn_verify_stream2.py`; pending decode-layer stream receipts |


## Closed recipes
A negative closes only the recipe tested. "Leaves open" names what the result does not rule out.

| Recipe tested | Result | Mechanism | Leaves open | Receipt |
|---|---|---|---|---|
| MoE V9 with PRMT NVFP4 decode (P8 measured PRMT 1.92x LOP3) | Bit-exact (524,288 cases); local 1.126x vs V9 1.283x; V32 501 -> 703 us | V9's bit-plane decode is already cheaper than P8's scalar raw-nibble decoders; rebuilding PRMT selectors on V9's layout grows the sm_90a inner loop 492 -> 1,255 instructions (LOP3 162 -> 545, SHF 112 -> 368) | A layout designed for PRMT from the start (changes prepare's packing) | local only; source in git `cand_v9_prmt.py` (commit before this one) |
| Proj: original K64 TMA/WGMMA with one/two consumer warpgroups and fixed DSMEM dispatch | 0.820x and 0.824x; prepare diagnostics m128 BF16 1.3–1.8x slower, m256 1.7–2.1x slower than cuBLAS | Unbalanced output waves, per-K64 synchronization and cluster merge overhead; the selected fixed dispatch also misses faster diagnostic variants | Larger grouped K tiles, last-CTA/Stream-K merge, per-shape autotuning; router m128 diagnostic is a positive lead | `87dd70d5f5894f6a`, `dd5ffc6950bc439e`; diagnostic logs |
| Proj: skinny mma.sync split-K with cluster DSMEM reduce for all m <= 64 | 0.991x; qkv/o lose 0.55-0.83x at m=1/8/32 | Split-K and cluster-reduce fixed costs dominate at these sizes | Other GEMV layouts (the two landed checks are below incumbent); in-kernel last-CTA reduce | `03113fd349d347dc` |
| Attention: synchronous K64/K128 swap-AB MMA, FP16/FP8 PV, deep gather ring | Best K128 full B16/B32 is only 0.336/0.333x FA3, stages 3–5 nearly equal | Nominal raw KV327.68/655.36MB divided by paired custom time gives0.531/0.558TB/s, versus2.558TB/s in the standalone P8 gather; whole-kernel issue/softmax/layout overhead remains. More splits help, deeper rings alone do not. No H200 counters isolate the dominant stall | Separate gather producers, larger warpgroup tensor operations, FA3 settings; pending full-specialized pipeline is not closed | `32c5b4b287c24c38`, `a0511c0001464dbe`; `dev/xqa_sm90_compile.log` shows K64 202–239 registers and runtime-indexed32B stack |
| Attention: split partials merged through global memory/L2 instead of a DSMEM cluster | 1.190x against v2's 1.279x | An extra global round trip per split | Global merge with fewer splits, or at long full-layer contexts | `8bfd5e25e582402d` |
| Prefill: contiguous FP8 cache gather (v4) | 1.194x against v3's 1.345x | Gains about 5% in the FA3 stage but loses elsewhere against the control | v9 contiguous/unpacked with DG experts also loses (1.185x); combining with canonical expert winners remains open | `27f3e312727d423c` |
| Prefill: 4-warp residual boundaries and rowwise SwiGLU (v2) | 1.229x, the same as v1 | Glue launches are not the prefill bottleneck | Glue fusion once expert time drops | `7ef49d84f4a14791` |
| MoE: Humming wider tiles / Stream-K / forced MMA (Humming4, Humming5) | 1.142x and 1.128x against Humming3's 1.270x | Wider token tiles lose at small batch | Per-batch dispatch is now measured: V9 native at M128, Humming wide at M256 (`318227bc382d465c`) | `14545ac2e22f48d6`, `630397d6cd414734`, `318227bc382d465c` |

## Failures
Failures whose cause was not obvious, so they are not repeated.

| Symptom | Cause | Avoid / fix | Cost | Receipt |
|---|---|---|---|---|
| A small-M MoE lab passes locally but fails agreement on the H200 with relative error ~1.8-1.9 (`3920d3f0`, `948044d4`) | The lab built its bank with recoded E4M3 scale bytes (the older V9 encoding) and fed them to readers that consume raw E4M3 bytes; the local screen used the candidate's own raw-scale bank | Build lab banks from the candidate's current raw-scale code, never from an older lab or host bank; print the first mismatching expert/row/channel | two prefill checks | `fee5f3b05fc5e53c` (fixed lab) |
| A lab that passed locally fails on the H200 with `router FP32 route IDs differ` (paired decode lab, `7307be7878604e6d`) | Its FP32 router oracle sums the split-K partials in another order than the tuned kernel, so a near-tie at the 8th expert flips under the H200's split plan (inferred; the A/B route check passed) | Exact top-k oracles must allow flips where the 8th/9th scores are within FP32 noise and report count and gap otherwise | one decode check | `runs/muse_decode_layer/*7307be7878604e6d.json` |
| The judge rejects a send before queueing it: `rejected: edits loaded modules` (paired decode lab, send 731545a5) and `rejected: starts processes` (small-M MoE lab, send 0d7b6cf0); the sending lane reported the run as pending | The first registered an exec'd module in `sys.modules`. The second's only process word was a docstring mentioning the subprocess module; 20 sources accepted before 04:00 UTC 10-24 carry the same docstring, so the check probably tightened that day (inference). Both rejections print only in `runs/detached/<key>.log` | Exec payload code in a plain dict (the paired lab's DPSpace). `okc_run.py send` refuses sources with the rejected words (`REJECTED_WORDS`); read the detached log right after a send | two lab sends, ~1 h each | `runs/detached/731545a5def446ad.log`, `0d7b6cf010101bd1.log` |
| A send fails with a 429 `daily cap of 10 runs ... counting runs not yet queued` although `okc problems` showed scored runs left | The server's cap counts sends it has accepted but not yet queued; `okc problems` doesn't | `okc_run.py` now tries the next mode (check) on that 429 | one MoE lab send | `runs/detached/053c794fc0aba59a.log` |
| Two files' receipts carry the same job; one of them never ran | When `okc submit --no-wait`'s reply named no job, `okc_run.py` took "our newest job of that problem since the send", which was the previous send's job seconds earlier | Removed the guess: a reply without a job id now fails the send. Corrected ledger rows: `d318167e` ran cand_expert_tma_cluster (1.419x; cand_v11_gemm never ran, re-sent), `f7579d53` ran cand_retire (not cand_ready), `5f402fe4` ran cand_chain_nopf (not cand_chain6), `907004cb` ran lab_moe_humming_real_routing (the attention sweep re-sent) | a false closure of dense prefill (cand_v11_gemm) and three wrong attributions | `d318167eedfd4c19`, `f7579d5311914b25`, `5f402fe4b1224a08`, `907004cbcc7c4b82`; judge duplicate verdict `9f738e2261164c30` |
| A decode run fails to compile on the H200 (`at::cuda` has no `getCurrentCUDAStream`), and another loses every XQA attention variant to `NameError: ATTN_X_CUDA` | Hopper-only branches (cuBLASLt extension; XQA ports) never build or run on sm_120, so local screens pass them: a missing `<ATen/cuda/CUDAContext.h>` and names never defined in the merged file | `tools/local_eval.py` static checks: ruff F821-F823 on the whole file and an sm_90a compile of every CUDA source string; both fail the screen | two decode runs (v9all failed; merge 1.359x with decode 32 full 581 vs 423 us), one more at the judge (moe_rt) | `83a8a755fece45e3`, `43a1f3a3d7ba46a9` |
| A packed candidate with gated alternatives dies with exit 112 before any test (`cand_r3_omni` a704a37c) | One alternative (the CUTLASS mixed-input MoE engine) hit a C++ `assert` in CuTe's `make_tma_copy_desc` (E4M3 scale tensor passed with globalDim (1,1,1)): an abort kills the process, so no Python gate can reject it | A native alternative rides in a candidate only after it has run on the H200 in a lab; TMA descriptors are built on the host, so build them locally on the sm_90a module (sm_120 has TMA) before any send | decode's last scored run of 10-24 | `runs/muse_decode_layer/*a704a37c*` |
| CUTLASS SM90 mixed-input GEMM on NVFP4 raises `RuntimeError: cannot implement 11` on the H200 (`3a98cfc8`) | Confirmed in the header: the stock RS mixed-input mainloop requires scale-group K to be a multiple of tile K64; NVFP4 requires16. Its one-scale-per-tile broadcast cannot implement group16 | can_implement is host code: call it locally on the sm_90a-built module before any lab | 1 MoE run; two more CUTLASS labs (560d1adb, d8d2cc5c) were already at the judge | `runs/muse_moe_w4a16/*3a98cfc8*` |
| A lab spends its slot on `LAB_ERROR ModuleNotFoundError` (`fbf64bd3`, Humming probe) | The payload imported a path that exists in no SGLang build in the image (`sglang.kernels.ops.moe.fused_moe_triton`; the function lives in `sglang.srt.layers.moe.fused_moe_triton`), and nothing ran the payload's imports before sending | `tools/h200_lab.py` resolves every import of the payload, nested ones too, in the judge image (`tools/img`) before building | 1 MoE run and ~1 h of queue | `runs/muse_moe_w4a16/*fbf64bd3*` |
| An attention run fails every test with `TypeError: _next_input() got an unexpected keyword argument 'ctx'` (`05fe4997`) | prepare's Hopper-only WGMMA portfolio called a helper with arguments it doesn't take; the branch never runs on sm_120 and ruff's name check doesn't see argument lists | `local_eval.py` static check `call_signatures`: every call to the file's own module-level functions must fit its signature | 1 attention scored run | `runs/muse_attention_decode/*05fe4997*` |
| Four decode checks fail on the judge with `AttributeError: 'tuple' object has no attribute 'device'` | The sm_90 branch received attention v3's `(partials, split_count)` tuple; only the Hopper branch reached that glue, so local tests passed | Force each Hopper branch's Python glue locally with a portable stand-in kernel | 4 checks | `2f3960c275774a20`, `69a655d3b2364fc4`, `a5ab4a2011f44321`, `b2a8999311034eb3` |
| Queued proj wg/wg2 can score without running WGMMA | `kernel_fn` requires `state["wg_ok"]`, but their prepare never sets it | All/max/native portfolios explicitly tune and store selected branches; max/native directly activate multicast WGMMA | Those queued runs cannot measure WGMMA performance | `cand_wg.py`, `cand_wg2.py` |
| TMA K128 BF16 descriptor rejected locally | A 256-byte inner box exceeds SWIZZLE_128B | Group two legal K64 transfers under one pipeline barrier; keep each WGMMA descriptor on a K64 swizzle atom | Local compile/test iteration | `cand_portfolio.py`; [CUDA tensor map documentation](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html) |
| Proj tuner cuBLAS graph replays fault after trial graphs are destroyed | Reproduced without custom kernels; PyTorch clears stream-keyed cuBLAS workspaces when any graph captured on that stream resets | Give each tuning graph its own native CUDA stream; reset its graph before destroying its stream. Warmup alone and retaining outputs do not fix it | Local test iteration; no judge run | `problems/muse_bf16_proj/dev/graph_debug.py`, `dev/graph_debug.log`; [PyTorch issue #193402](https://github.com/pytorch/pytorch/issues/193402) |
| Prefill v12 large-shape BN256 faults on H200 after small loads pass | Illegal memory access surfaces at `combine_tiled_kernel`; asynchronous producer source is unresolved | Treat the old384-thread BN256 implementation as suspect, not a negative performance result. New canonical BN256 is a separate128-thread reader with fresh local graph/reference coverage and no SM90 spills | 1 check | `53b79ddde4a048d5`, `problems/muse_prefill_layer/profiles/cand_v12.53b79ddde4a048d5.log` |
| A "DeepGEMM dense" prefill candidate profiled as cuBLAS | Plain BF16 `deep_gemm` calls dispatch to cuBLAS unless `deep_gemm.use_deterministic_algorithms(True)` | Set it, and confirm the kernel names in the profile | 1 check (v5); v13 corrects it | `84b5433a1f2f43e5` |
| A send waiting for a judge slot vanished: no receipt, no pending or waiting entry | The sending process died with its caller (a Codex exec's exit, a tool timeout, `h200_lab.py`'s foreground call), and `runs/waiting/` drops entries whose process is gone | `okc_run.py send` now detaches into its own session after its checks (log `runs/detached/<sha16>.log`) | On 10-24, labs prefill_experts_sweep (3 tries), humming_nvfp4_probe (2), bf16_real_deepgemm (2), and candidates cand_attn_verify_stream2 and cand_r3_v9_vstream never ran | `runs/sent/` snapshots with no ledger row |
| Local ncu fails (`ERR_NVGPUCTRPERM`) | `RmProfilingAdminOnly=1`, and there is no sudo | Use nsys, judge profiles and SASS inspection | — | — |
| An in-prepare tuner raised on its own agreement check and failed the H200 run (`cand_fa3_tune`, check `dd927c6c46264afe`) | It required each FA3 setting within 0.5% of FA3's default; num_splits=1 differed by 0.86%, while the judge allows 3-15% against FP32 (TASK.md) | Tuners reject a variant (log it) rather than raise, and accept by the judge's tolerance, not agreement with one incumbent | one judge slot and the sweep's data | `problems/muse_attention_decode/cand_attn_portfolio.py` |
| A layer candidate imported a sibling module (`from moe_tune import tune`) and passed the local screen | The local run imports from the candidate's directory; the judge gets only the one file | `tools/screen_all.py` fails any import of our own modules; inline helpers into the candidate | caught before sending | `problems/muse_decode_layer/cand_r3_moe_rt.py` |
| Runs orphaned in the judge queue after their waiting process died | The judge keeps the job; only the waiting process was lost | `tools/okc_run.py collect <job>` (receipts start in `runs/pending/`) | hours of re-waiting | `runs/pending/` |
| Proj broad portfolio nvcc crash | `std::vector` path triggered nvcc internal assertion in `stl_construct.h`, `find_allocated_name_reference` | Replaced the host-only heuristic buffers with explicitly owned C buffers; both architectures build without dropping variants | Local admission, no judge run | `dev/all_lt.cuh`; original compiler module `muse_proj_lean_bcfdb871d5_*` |
| Proj broad serial cold prepare | sm_120 compilation/repack 278.96 s + tuning 69.93 s = 348.89 s, exceeding 300 s | 16-way nvcc and ptxas optimization reduced cold all-shape prepare to 122.66 s; full superset 71.77 s with a cold extension build | Same kernels and searches retained; H200 total prepare still unmeasured | `dev/portfolio_all_serial_cold.log`; new all/max screens |

## Decode q1 FP8-KV split candidate (2026-10-24)

The standalone q1 port initially hard-coded `cs=1`; at full context that made
the split reduction diverge by 1.16 relative L2 even though sliding happened to
pass. The corrected `port_decode_fp8_split` uses the extension's occupancy and
cluster planner, keeps two waves at B=1 and one at B=16/32, and fuses the
norm-gate into the final combine. The cache stays FP8 and no dequantized bank is
made in prepare.

The local full-layer gate passes all six decode shapes (B=1/16/32 ×
sliding/full); update L2 is 0.0063/0.0095, 0.0080/0.0082 and 0.0084/0.0094
respectively. `tools/local_eval.py --tests-only` passes 13/13 tests and 15/15
static checks, including sm_90a compilation.

The batched off-quota prefill-host lab payload is
`prof/lab/decode_fp8_split_compare.py`, key `5068f5e616648715`, queue position
2/5. It compares q1 split, attn_main and FA3 at B=1/16/32, ctx 80000, page 1,
agentic sliding/full, and carries the ready q8 FP8-PV settings. Promotion is
receipt-controlled: require a passing post-store FP32 gate and a q1 time below
both controls for the same shape; otherwise keep best6.

## FP8PV verify diagnosis and fast combine (2026-10-24)

Receipt `948044d4f32a4736` established that the FP8 probability/V path is
correct (post-store FP32 error 0.093–0.097) but costs 1.25–2.37 ms at verify
B16/B32, versus FA3 about 0.208/0.390 ms full. The source explains the gap:
one 128-thread CTA owns one KV head and one split, streams its split serially
with a barrier per 64-key tile, and the elected final CTA reduces every split
for all 64 query/head rows in a single 128-thread combine. The H200 launch
geometry is about 160 CTAs at B16 (`ns=5`) and 192 at B32 (`ns=3`) for one wave.
Each full-context CTA loads roughly 1.9–2.2 MB at B16 or 2.0–2.2 MB at B32;
total K+V traffic is about 327.7/655.4 MB. The receipt timings imply only
about 0.262/0.306 TB/s, far below the 4.8 TB/s floor, so this is issue,
barrier, and combine bound rather than HBM bound.

A fast combine variant assigns two 16-lane query groups per warp, with four
contiguous output dimensions per lane; it preserves FP32 split max/sum and the
post-store gate. Local sm120 gates pass (full/sliding error 0.083/0.088 for
B1), and local B16 full time improved 987 -> 848 us. The new payload
`prof/lab/fp8pv_phase_compare.py` adds clock64 phases, exact CTA/tile byte
accounting, baseline-vs-fast timings, and gate checks. It was accepted as
`muse_prefill_layer` job `0812f09758da4fa5`; detached send key/log is
`9913f6a6493e24dd` (no rejection). Promotion requires the fast reducer to
retain the gate and materially reduce combine/whole-kernel time on H200; if
bandwidth remains below about 1 TB/s, the serialized per-tile producer path
also has to be redesigned and this reducer alone stays diagnostic.

H200 result (`0812f09758da4fa5`): the fast combine changes almost nothing. B16/B32 full
1,271.8 -> 1,255.4 and 2,143.0 -> 2,130.8 us (attn_main at q8: 412.7/758.4; FA3 ~208/390),
0.13-0.15 TB/s; sliding 92.0 -> 79.4 and 116.9 -> 100.3 us. Per 64-key tile the CTA spends
~570 cycles in QK MMA, ~2,200-2,700 in softmax and ~2,800-3,100 in PV, serialized, so the
per-tile producer/softmax path binds, not the combine. This kernel family stays parked; the
attention lane moved to FlashInfer's page-1 paths (10-24 23:40).

## FlashInfer page-1 attention (H200, `8ffbac2d68074a0b`, 2026-10-25)

FlashInfer 0.7.0.post1 at the scored full-context shapes (ctx 80000, page 1, FP8 KV): tensor-core decode (q1) takes 45.5 / 195.2 / 371.9 us at B1/16/32 (1.68-1.76 TB/s), the same as FA3 (205.6/386.1 at B16/32) and ~1.8x our attn_main (108.0/199.7). The page-1 prefill path (fa2) at q8 takes 60.4 / 469.1 / 912.3 us, ~2.3x FA3's q8 (208/390). Errors 0.0023-0.0027; graphs capture; plan 0.2-0.3 ms outside the call (11 ms on first use). Not measured: every sliding case (FlashInfer asserted on the window setup), the fa3 backend (JIT build failed in the image), XQA (rejects page size 1 by contract) and the lab's own FA3 control (dtype bug). The run failed because the payload pushed prepare to 304 s (cap 300). FlashInfer's full-context paths are closed for the decode layer; its fa3 backend and sliding paths remain untested.


## Single-launch sliding attention measurement candidate (2026-10-25)

Candidate `problems/muse_attention_decode/cand_sliding_singlelaunch.py`, SHA-256
`68c50c706423f6d6cfe52cc4cd487840e9794a9c62664f7633d59bfd5cbdcb75`, was sent
directly to the attention board with key `68c50c706423f6d6` (priority 2), accepted job `32f2f07acfe144c2`.
The candidate changes all six sliding shapes; full q1 retains the fused
`attn_main` control and full q8 retains stock FA3 on Hopper. It is a measurement
candidate, with no H200 promotion yet. FlashInfer receipt `8ffbac2d68074a0b`
(from send key `c96e0f89bcdcdf60`) became available in the shared files: FA2
verify full is 469/912 us at B16/B32 and supplies no winner over FA3. Its
sliding paths asserted and its FA3 backend failed to build; those remain
unmeasured. Keep the candidate's FA3 full-verify choices.

The new CUDA path performs Q/K RMSNorm, GPT-J RoPE at the task's theta,
post-quantization FP8 KV store/attention, output RMSNorm and sigmoid gate in
one launch. A two-stage pipeline stages 256 FP8 KV rows; each warp's MMA
computes eight query-token rows at one query head for q8. For q1 B>1 the
same eight rows are the whole GQA group, removing eight redundant KV streams.
B1 uses eight splits per query head; q1 B16/B32 uses four/two splits per KV
head; q8 B16/B32 uses one CTA per query head. Four warp partials merge in the
CTA. The last split CTA merges global partials and applies norm/gate before
returning; counters return to zero and no partial is reused by another call.
Every CTA builds quantized new KV locally and reads global history only below
`seq_lens`, eliminating cross-CTA cache-store races for q8.

The layer adapter `problems/muse_decode_layer/dev/sliding_attention.py`, SHA-256
`68fb992bd795b29a2257b88fc8c08363749a4446eaf1e1241e1961c1ae8264f6`, accepts
BF16 QKV or the layer's `(partials, splits)` tuple. FP32 projection partials
are summed and rounded to BF16 inside the kernel, avoiding an extra reduction
or cast launch. `attention_dropin.with_sliding_attention(source)` adds the
`('singlelaunch_sliding', None)` choice to an existing v9-derived portfolio;
it does not select it. Keep best7 settings until the H200 row wins.

Verified locally on sm120: 16/16 official tests, 12/12 fresh-input graph
checks, undefined-name and call-signature checks, and sm90a compilation of
both CUDA sources. The adapter passes all six scored sliding shapes and
12 additional gates for 1/4 FP32 projection partials after changing captured
input values. Its worst-row error is about 0.0012. CUDA profiling confirms
one kernel at B1 q1/q8 for BF16 and split-projection inputs. The actual full
q8 norm/store/epilogue Hopper glue passes graph replay with a tensor stand-in
for FA3, including KV scale 1/64; the FA3 binary itself remains H200-only.

Local screen timings (us, **not H200 evidence**, shared sm120):

| B | q1 sliding | q8 sliding | local gate | H200 |
|---|---:|---:|---|---|
| 1 | 9.2 | 13.3 | P | pending |
| 16 | 16.4 | 81.9 | P | pending |
| 32 | 26.6 | 122.9 | P | pending |

Early scalar and broad Triton gather trials were correct but lost locally;
serial gathers / excessive per-head work were the limiter. Staging plus MMA
reduced B1 from roughly 48–70 us to 9–13 us. A short-context test also found
`-inf - -inf` when every warp in a split is empty; the merger now keeps its
partial exactly zero. Every corrected config was re-gated before sending.
The larger q8 shapes remain slow locally; H200 must establish their viability,
and they must not enter decode settings solely because they fuse launches.

At the requested B1 target of 6 us, decode1 sliding attention would save
`46*(14.8-6) = 404.8 us` per 62-layer load; verify1 would save
`46*(19.4-6) = 616.4 us`. These are projections, not measured layer gains.
Using best6 load times, decode1 alone would contribute about +1.04% overall;
verify1 alone about +1.08%. Promotion requires a passing H200 gate and a
same-shape incumbent win, then a full-layer check including projection
partials, graph capture and the actual best7 projection scheduling.
