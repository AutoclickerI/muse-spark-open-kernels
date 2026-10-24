# Findings

Each row is one scoped quantity. Class: **measured** (on the judge, by our runs), **static** (counted
from code, config or spec), **model** (an estimate under stated assumptions; numbers from `python -m sim`, definitions in
[ceiling](reference/ceiling.md)), **official** (published by the organizers). Per-shape tables:
`python -m sim.calibrate`.

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

- **Scope and bar:** free lane owns decode verify32/M256; serving/prefill-only work is below decode levers under R-20261024-4. Masked Humming's12-seed mean is448us vs V9 NT2545.5 (de6a56fc); best6 spends262+146us on its GEMMs plus25us glue. Scalar296fdf45 and plane01e74ee6 are correct but lose badly (M2562648/2361us vs V9479–480). R0 ablation276739d2 showed W13 grid132/264/528=3067/1758/1197us; no-conversion757, constant-weights463, no-MMA1670 at grid264. These motivate latency hiding and conversion work; they do not isolate every cycle as an HBM stall.
- **Vector rejected; correctness cause reproduced locally:** H2004ee35d3dd5374c40 rejects every Vec16/32/64 row (relative L2~1.415), with timings471–501/660–763us at M128/256 versus V9NT2341/548. Its B128 shared tensor starts at byte5632 (phase512 modulo1024), but the descriptor uses base_offset0 and the portable MMA reader incorrectly shared the writer's relative XOR. `check_descriptor.py` emulates Hopper's absolute pattern on the frozen failed source: all three NT choices give1.416042 L2; aligning the pattern gives0.0004215. The [PTX matrix descriptor rules](https://docs.nvidia.com/cuda/archive/12.9.0/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-shared-memory-layout-matrix-descriptor) specify the alignment/base-offset relationship. Current vector/pipeline/fused sources align the dynamic base and B offset to1024; the fused portable reader includes absolute phase. Corrected H200 agreement remains pending.
- **Deeper pipeline and compiler repair:** K128/256,3–6 cp.async stages and two register-A banks overlap conversion and WGMMA using wait1. Packed Q+SF prologue reaches27648B/CTA vs vector's4608; gathered X is async; source indices are hoisted per task and barriers occur per stage. Fully unrolled K256 previously triggered ptxas C7513 and per-MMA wait0; input-only keep points plus a non-unrolled loop over pairs of K64 chunks fix it. Older pipeline lab rehosted on MoE under send keyc686764c8d6fdc1f remains pending; its chosen stage geometries already align B.
- **New three-launch M256 producer:** `dev/mixed_wgmma/fused.cu` launches GPU route/reset, W13+SwiGLU, then W2+postnorm+combine. W13 pairs32gate/32up channels per M64 tile and supports independent FP32 global scales. W2 last-channel CTAs finish each pair's norm; the last local pair combines each token. Split-K1/2/4 uses last-split epilogue reduction, acquire/release GPU counters and all-thread publication fences. No inter-CTA spinning or grid-residency barrier. Weights/scales remain packed E2M1/raw E4M3 in prepare, converted inside timed CUDA; no expanded resident bank. V9 is unchanged.
- **Checked locally:**210 independent FP32 checks, including192 changing-route graph replays over8configs×3grids, plus signed/zero/subnormal scales, independent gate/up GS and nonzero local_start. Max L2=0.004225, max row=0.004546. All16 actual sm90a GEMM specializations compile at91–112 registers, zero local-memory instructions, two static wait1 sites and one final wait0. The complete MoE-host wrapper passes5/5 official tests,6 static checks, all12seed/19aggregate gates; direct payload static8/8. This is SM120 math plus native compilation, not an H200 win.
- **Queued M256 lab:** `prof/lab/moe_fused_m256.py` includes13fused stage/splitK/grid choices plus6controls,12public weight seeds with3fresh real-router fixtures each,8rotated/reversed cold128MiB graphs, finite/L2<=1%/row<=5% gates and fixed-variant seed means/medians. One active-architecture extension and shared packed bank keep all12seeds within prepare's budget. Exact R0 IDs (414pairs/21experts/hot119, refreshed public weights) and synthetic BAL32 (256pairs/32experts/hot8) are separate tracks. Their unique packed floors at4.8TB/s are124/189us respectively;126us does not cover32unique experts. Phase probes reset completion counters; Q+SF B/s is nominal requested bytes, not a DRAM counter. Queued on MoE as send key`21b28ddbe14e119a`, batched with unchanged M128 Humming;4189-character report retains every seed/phase within its4500-character share. Queue identity and current hashes: [admission.json](../problems/muse_moe_w4a16/dev/mixed_wgmma/admission.json).
- **Boundaries:** canonical V6 and Humming layout inversions recover codes/raw scales/FP32 GS exactly. Stock CUTLASS compiles E2M1 but its K64 scale broadcast cannot implement group16; private host TMA repair passes exact/empty routes, numeric GEMM remains quarantined and out of omni. No CUTLASS slot, candidate promotion or serving producer change by this lane. A decode drop-in waits for an H200 seed-mean win over masked Humming and complete-layer agreement. [Source and protocol](../problems/muse_moe_w4a16/dev/mixed_wgmma/README.md).

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

- **Humming at M=128/256 is bandwidth-starved (`896866402dc94112`, R0 + 12 seeds, eager event spans):** W13 streams the touched experts' unique NVFP4 bytes (16 MiB W13 + 8 MiB W2 per expert; 11-25 experts, 264-600 MiB) at 0.70-1.18 TB/s and W2 at 0.63-1.07 TB/s (M=128 R0: W13 261, W2 146 us for 432 MiB, whose read at 4 TB/s is ~113 us). Humming's own tiles (block [16|24,256,128], 4 stages, TMA, warp-specialized) leave ~4x of HBM bandwidth unused, so the M=128/256 lever is bytes in flight per SM, the same limit as the dense GEMVs (b63389f0). Its SM-count and block16 launch variants all failed the FP32 gate (0.37 relative error), a lab defect rather than a finding.

- **V9 variants at M=1-32, seed means (`b68afb72b72542b4`, 7-8 public seeds each; the AGG lines were truncated):** NT1 is fastest at M=1/8/16 (49.0/84.4/94.1 us), hotNT1 at M=32 (136.8 vs NT1 140.8). NT2 costs +16% at M=1 and +28% at M=8, and NT4 +63-96%, so best6's NT1 stays below M=128. At M=1, NT1 spends 42-59 us on 2-4 local pairs, 2.5-4x one read of those experts (24 MiB each). The same run's real-routing payload died on a lab bug (`KeyError: (128, 'R')`).

- **Small-M H200 failures isolated to the lab's scale contract (`3920d3f0`, `948044d4`):** the stale embedded `v9_prepare` recoded E4M3 scales, while NT4/v9m1's `sf_bits` reads stored E4M3 bytes. Scale 128's stored byte `0x70` became `0xe0`, which that raw reader interprets as -32. The local lab skipped these branches and its stand-in only checked argument shapes. The exact wrong bank/reader pairing now reproduces locally at **1.862656 relative error**, first diverging at expert 3's SwiGLU row; the corrected raw pairing passes. `prof/lab/build_moe_small_debug.py` rebuilds the compact `lab_small_m_nt4.py` from the current candidate's raw bank and readers, asserts that no scale encoder is copied, and removes the obsolete embedded layer/portfolios. Local evidence: **192 inverse tile checks** preserve FP4 codes and stored E4M3 bytes; **160 forced Hopper launch/graph calls** cover all five native dispatch names and both shared-memory paths; four native M1/8/16/32 stage checks plus 12 fresh graph replays pass (max output FP32 L2 **0.208%**). A second run through the real-router lab entry passes all four shapes (max L2 **0.236%**, worst row **0.246%**); all eight native source strings compile for sm90a and both Python static checks pass. These are sm120 correctness results, not H200 latency. The new H200 self-check reports the first expert/token/channel mismatch at the bank, SwiGLU, W2 or squared-norm stage before timing. Candidate dispatch remains NT1 at M1/8/16 and hotNT1 at M32; NT4/v9m1 are explicit lab choices, and NT4 was removed from automatic M128/256 selection pending the corrected H200 result. Logs: `runs/screen/moe_small_{debug,lab_entry,static}.*`. Corrected lab send key **`fee5f3b05fc5e53c`**, payload SHA-256 `710422949de76f0371126b8408c1ea984105250f45c67c9a63cbc6837785111a`; its detached log says waiting for a judge slot (not yet judge-accepted), with no rejection at the immediate log read.

- **Deeper packed WGMMA pipeline, exact R0 routing (`7156d8d8b56a40d1`; seed means truncated):** the best whole-MoE config P5 (NT32, K128, 4 stages, overlap) takes 313.4 us at M=128 (NT1 331.3, masked Humming 348.0) and 415.5 at M=256 (masked 437.0, NT1 576.0), ~5% ahead. Its W13 alone peaks at 195 us for M=128 (P1, grid 1056: 1.74 TB/s of unique bytes) and 261.6 at M=256 (P5: 1.51 TB/s), still ~2.5x one read at 4 TB/s. Humming's masked-TMA variant is ~3% faster than masked (337.6/427.4 vs 350.9/437.6 us); its indexed variant fails the gate (0.37). The run's prepare reached 295 of 300 s because the first payload overran its time share.

## The judge's H200 (measured, 2026-10-23)

| Quantity | Value | Scope | Class | Receipt |
|---|---|---|---|---|
| Judge GPU state | H200, SM 1980 MHz, memory 3201 MHz, 700 W limit, ~130 W idle draw | node n-qtqz4qekgtq37epqb5xaarah6h | measured | `runs/muse_bf16_proj/*7fb3e93056d34999.json` (`machine`) |
| Graph replay fixed cost the judge subtracts | ~4.2 us (raw median minus reported median) | bf16_proj shapes | measured | same, `replay_overhead`, `median_raw_ns` |
| Historical HBM/L2 sweep | 52-61 MB stream: 12.8-14.6 us = 4.04-4.25 TB/s; best used 132 CTAs x 1024 threads and source unroll 4 x 16 B/thread. Warm 4-16 MB: 8.2-9.6 TB/s; 32 MB: 4.5-5.1 TB/s | CUDA events with the sweep's baseline subtraction, different loads/cache hints and CTA sizes from P8 below; outstanding requests were not counted. The 32 MB drop is specific to this protocol | measured | `problems/muse_bf16_proj/profiles/cand_sweep2.62bd26b45aeb4992.log` |
| V9 MoE time depends on the run's router seed | The same `moe_fused<1>` kernel took 258.9 vs 304.6 us at verify 16 sliding and 449.5 vs 546.6 us at verify 32 full in 0e94268 (1.428x) and 7b2165c8 (cand_r3_best, 1.351x), while SGLang's baseline moved 1-2% (537.7/540.9, 1088.8/1113.1 us). prepare() gets a per-run seed, so routing and the hottest expert's load change per run; V9's tail (one CTA set finishing the hot expert) carries the variance. Decode scores swing a few percent per run from this alone: compare candidates by non-MoE stage times or on the same routing, and balance the hot expert (split its N, hot-first scheduling) to make V9 both faster and steadier | receipts' per-kernel profiles | measured | `problems/muse_decode_layer/profiles/cand_r3_best.7b2165c875bd48de.json`, `cand_r3_v9.0e94268cf5654012.json` |
| best5 vs the board run, per stage | cand_r3_best5 `691bff0bcff44e99` 1.353x vs 0e94268 1.428x: MoE at verify 16/32 ran at the 12-seed mean (341/527-530 us vs 259-280/450-477), attention at decode loads improved (decode 1 full 33.9->18.9, decode 32 full 228.5->197.8 us), and verify-32 full FA3 regressed 377.9->501.5 us: prepare picked num_splits=32, pack_gqa on a synthetic input whose KV lengths don't match the scored 'agentic spread', while SGLang's baseline on that load moved -1.6% | decode_layer | measured | `problems/muse_decode_layer/profiles/cand_r3_best5.691bff0bcff44e99.{txt,log}` |
| best5_glue vs best5, per stage | `7370b18221964412` 1.367x vs `691bff0b` 1.353x: the three PDL glue launches left glue kernel time and total-minus-MoE-minus-attention within -6.8..+2.6 us per case (noise); the score moved with MoE routing (verify 32 527/530 -> 503/490 us) and FA3 at verify-32 full (509 -> 380 us, same tuner code). Single scored runs can't resolve few-us changes; paired labs can | decode_layer | measured | `problems/muse_decode_layer/profiles/cand_r3_best5_glue.7370b182*.txt` |
| v9 vs the best5/best6 line, 3 seeds | The board entry v9 (`0e94268c`) reran as the paired labs' host: 1.428/1.422/1.432x on three router seeds (`7307be78`, `1365af66`). best6 1.380x and best5_glue 1.367x are below it. Per load, v9 mean vs best6: decode 1 1.526/1.606, decode 16 1.553/1.472, decode 32 1.321/1.203, verify 1 1.725/1.670, verify 16 1.316/1.226, verify 32 1.164/1.263. The gaps are MoE: decode 32 sliding moe_fused<1> 101.7 (v9) vs moe_fused<2> 137.6 us (best6); verify 16 sliding 258.9 vs 315.7 us (layer 387-393 us on three seeds vs 423/432). best6 wins at decode 1 full (attention) and verify 32 (Humming). Per-case best of both models 1.455x | decode_layer | measured | `runs/muse_decode_layer/*0e94268c*`, `*7307be78*`, `*1365af66*`, `*1b80ffc0*` |
| best6 vs best5_glue, per stage | `1b80ffc037cc454c` 1.380x: prepare picked humming_masked at M=256 (V9 NT1 at M=128); verify 32 MoE (W13 262 + W2 146 us + swiglu/combine/gather 25 us) vs V9 503/490 us, layer 648->583 (sliding) and 986->884 us (full); FA3 at verify-32 full 376 us. Other loads moved with the router seed | decode_layer | measured | `problems/muse_decode_layer/profiles/cand_r3_best6.1b80ffc0*.{txt,log}` |
| Lab host on a withdrawn run | `tools/h200_lab.py` picked the host as the best scored run, which for the decode layer was the withdrawn, inadmissible bf16moe `07d07d960b814b86` (1.461x): the batched GEMV+overlap lab `b63389f01e4741b9` runs on it (a check, not on the board; its host-function timings are not the incumbent's). Host selection now skips withdrawn runs (decode host: 0e94268) | tooling | measured | `tools/h200_lab.py` best_source |
| Streaming GEMV vs a plain read (H200) | `feb7b287e3464e4a`, cold 256 MiB flush, 20 pairs: a plain one-launch read of each weight takes qkv 18.6 us (37.75 MB, 2.03 TB/s), o 11.3, pool+router 30.8, unpool slice 8.7, unpool full 28.4 us (0.96-2.45 TB/s; ~2.4-3.1 TB/s after a 5.2 us launch/ramp envelope). The streaming GEMV matches that read at M=1 (qkv 18.3, o 12.3, pool+router 29.6, unpool slice 9.3 us; 0.77-1.01x cuBLAS) and loses to cuBLAS at every M>=8 (1.1-1.7x at M=8-32, 2.3-4.9x at M=128-256). Drafter shapes lose too (1.12-1.80x) | One launch can't stream 8-75 MB faster than ~2.5 TB/s, so the M=1 projection lever is overlap across launches (the next stage's weights in flight during the current one, as best6's PDL prefetch does), not the GEMV mainloop | measured | `problems/muse_prefill_layer/lab/profiles/lab_streaming_gemv_lab.feb7b287e3464e4a.log` |
| Fused glue GEMVs and verify-load WGMMA (H200) | `5535d2f171b34d72`: fusing norms/residual/routing into the streaming GEMV costs 2.3-21x cuBLAS at every M (o at M=1 69 vs 17.8 us; 0.03-0.7 TB/s). The weight-wide WGMMA for the verify loads loses 1.28-2.18x to best6's own Lt/cuBLAS choices: dense total M=8 102.3 vs 64.9 us, M=128 128.0 vs 84.4, M=256 161.3 vs 99.6. best6 already streams the dense weights at 2.0-2.3 TB/s (138-197 MB), near the ~2.5 TB/s one launch reaches at these sizes | Standalone dense kernels are at their per-launch limit; the dense lever at every load is overlap across launches, not a new mainloop | measured | `problems/muse_prefill_layer/lab/profiles/lab_streaming_glue_lab+verify_streaming_lab.5535d2f171b34d72.log` |
| Standalone projection GEMVs on the H200 | Cold-L2 graph replays, decode-layer shapes (`b63389f01e4741b9`): qkv 37.75 MB in 20.7-21.9 us (1.73-1.82 TB/s), o 16.78 MB in 11.9-13.9 us (1.2-1.4), pool+router 75.5 MB in 32.3-34.4 us (2.2-2.3), unpooler 67.1 MB in 27.6 us (2.4), unpooler slice 8.4 MB in 11.2 us (0.75), at M=1/8/16 alike. cuBLAS, every Lt heuristic, our GEMV and a 4x-CTA split all land within a few percent, while a plain 52-61 MB read stream reaches 4.04-4.25 TB/s. So the GEMV kernels, not launches or dependencies, leave ~45% of the bandwidth unused. The same run's overlap payload hit a sticky `cudaErrorInvalidAddressSpace` (on the withdrawn bf16moe host) and failed the run | decode-layer projections | measured | `tools/h200_lab.py read b63389f01e4741b9 gemv_standalone_lab` |
| Streaming projection lab | `streaming_gemv_lab.py`, payload SHA `1d9b7bad…`, rehosted on prefill with send key `b1047d984f6c5437` (original proj key `d962f665f754dea0` was quota-blocked). M1: 16-byte register loads with a true dependency enforcing 4/8/16 loads before the first FMA in SASS. M>=8: padded shared rows, 2-4-stage cp.async, BF16 mma.sync; one-launch last-CTA split-K, separate FP32 pool/router hi+lo rows. Cold-L2 paired cuBLAS and plain-read controls use the same protocol; report raw and replay-cost-subtracted TB/s | sm_120: 537 FP32 gates, 135 fresh-input graph cases, max relative error 1.44e-5; exact lab host 9/9 and 3/3 static checks, prepare 8.7 s. Native sm_90a build: no spills, max 160 registers. Larger ring stages use fewer stages on sm_120. No H200 speed claim or decode dispatch change yet; next vector shared-load consumer is locally gated and unsent | local admission; H200 pending | `problems/muse_bf16_proj/dev/streaming/README.md`, `validation.json`, `compile_sm90a.json` |
| Streaming projection + glue lab | `streaming_glue_lab.py`, payload SHA `8ec41783…`, queued on prefill host cb8939d0 at priority0.5; send key **552621abdd6a86a3**. Thirty keys at M1/8/16/32/128/256 plus12 dependent chains. Each fused projection completes split-K, boundary RMS/residual/next norm or pre-expert RMS/router in one launch. Direct last-CTA RMS and distributed FP32 quadratic moments, both shared consumers and both pre-FFN placements; cuBLAS+glue and identical-GEMV+separate-glue controls, cold256MiB/20 AB/BA repetitions | sm120:6105 configuration/edge gates,510 captured fresh-input graphs; raw FP32 1e-4 and independent exact router IDs, ties/cancellation/counter reset; exact prefill host5/5 tests,30/30 keys and12 chains without rejection,34.4s cached prepare. Native sm90a33 kernels, no entry/device spills, max240 registers. Local timing loses to cuBLAS; no H200 speed claim or decode dispatch change. Unsafe overlap stays deleted; no NVFP4 expansion in this dense path | local admission; H200 pending | `problems/muse_bf16_proj/dev/streaming_glue/{README.md,validation.json,compile_sm90a.json,host_admission.json}` |
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
| Serving | Apply these as candidate rankings for the relevant operators; no end-to-end serving gain was measured. APW/hints need validation with concurrent streams and the real live working set | same job |

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

## Serving track

| Quantity | Value | Scope | Class | Receipt |
|---|---|---|---|---|
| Serving metrics | the site: "throughput (tokens/s per GPU) and interactivity (tokens/s per user) at each load"; nothing on input tokens, TTFT or percentiles. LMSYS's own Muse sweeps count input (with cache hits) + output tokens and use 1000 / p90 inter-token latency | serving | official (site bundle, LMSYS blog) | site JS `index-DQmeD77Z.js` ("Score" card); https://www.lmsys.org/blog/2026-10-15-meta-muse-spark-open (fetched 2026-10-23) |
| LMSYS's Muse serving numbers | 8x B300 only: headline 14.8K total tok/s per GPU at 543 tok/s per user (p50) at concurrency 1; AgentX single session 457.8 tok/s per user. No H200 serving numbers published | serving | official (LMSYS) | same blog |

### Local host/device step trace (2026-10-24, sm_120)

The tiny margin fixture was served with `serving/harness/run.py`'s judge flags, DFLASH,
one 2,608-token prompt per request, 220 output tokens, and GPU0 only. Nsight Systems was
injected into the server child (not the client wrapper). The harness completed all five
requests (one at c=1 and four at c=4) with 220 tokens and no request errors. The probe put
nonblocking CUDA events around each DFlash forward and NVTX/host spans around the scheduler,
accept, draft-preparation and output functions. These historical device-event intervals
include launch gaps; they are not GPU-busy measurements. Subtracting that interval from
`scheduler.run_batch` does not establish 2.1 ms of physical GPU idle or an H200 saving.
The corrected kernel/copy activity measurement follows below. Nested attribution rows
must not be added to the step wall time.

| Per decode/verify scheduler step | TP1 c=1 (ms) | TP1 c=4 (ms) | TP2 c=1 / c=4 |
|---|---:|---:|---:|
| Scheduler step wall (`run_batch`) | 12.06 | 11.40 | unmeasured / unmeasured |
| GPU event interval (includes launch gaps) | 9.91 | 9.33 | unmeasured / unmeasured |
| Outer-span minus event interval | 2.15 | 2.07 | unmeasured / unmeasured |
| DFlash forward host span (launches + bookkeeping) | 9.90 | 9.32 | unmeasured / unmeasured |
| Named CPU batch build (`get_next_batch_to_run` + update) | 1.64 | 1.93 | unmeasured / unmeasured |
| Verify accept bookkeeping (`_accept_block`) | 0.31 | 0.30 | unmeasured / unmeasured |
| DFlash draft-prep CPU helpers | 0.10 | 0.10 | unmeasured / unmeasured |
| Target-hidden → draft-KV materialization host span | 1.95 | 1.88 | unmeasured / unmeasured |
| Sampling on decode path | 0 (folded into graph) | 0 (folded into graph) | unmeasured / unmeasured |
| Output processing, regular step | 0.25 | 0.50 | unmeasured / unmeasured |
| Host-device sync API time | 0.026 (14 `cudaStreamSynchronize`) | 0.055 (54.8 us; 523 stream + 41 event sync, 353 query, 174 wait; zero device sync) | unmeasured / unmeasured |

The c=4 steady window contained 63 outer batches (59 steady decode batches and four extend/transition batches). Nsight counted 523
`cudaStreamSynchronize`, 41 `cudaEventSynchronize`, 353 `cudaEventQuery` and 174
`cudaStreamWaitEvent` calls, 3.45 ms of synchronization API time in total, or 54.8 us per
outer batch. A single c=1 steady event had 14 stream synchronizes taking 25.6 us; c=4
stream-synchronize calls had a long tail up to 1.32 ms even though their median was short.
The decode sampling path is already graph-folded, so there is no measured host sampling
budget to recover locally. TP2 was deliberately not run: GPU1 was occupied by another job,
and a `CUDA_VISIBLE_DEVICES=0,0` TP2 attempt failed at CUDA device enumeration before load.

The runner metadata reports a post-run source-tree hash change while profiling (the workload itself completed); the retained trace and summary are still usable, but this is not a clean provenance run. Evidence: `/data2/jihoon/muse_kernel/harness/results/nsys-device-tp1/measurement_summary.json`, `/data2/jihoon/muse_kernel/harness/results/nsys-device-tp1/host_trace.jsonl`, `/data2/jihoon/muse_kernel/harness/results/nsys-device-tp1.nsys-rep`, and `/data2/jihoon/muse_kernel/harness/results/nsys-device-tp1.sqlite`.

### Host overhead: implemented, H200 gain unverified (2026-10-24)

**DFLASH spec-v2 overlap is already enabled by the judge flags.** `_handle_dflash` in
`arg_groups/speculative_hook.py` does not disable it; the PP/diffusion/embedding overrides
do not apply. The normal FA3 backend declares `needs_cpu_seq_lens=False`; local FlashInfer
requires that mirror. The local length relay and FlashInfer planner's blocking metadata
conversions therefore cannot be booked as a recoverable H200 budget.

The patch now prepares the next batch and settles the previous result after target verify
is enqueued, using a separate scheduler/spec-input snapshot and the existing shared-read
WAR fence. It does not mutate the batch in flight. Decode target-hidden materialization
uses exact-shape CUDA graphs for every batch size 1..32. Inputs are refreshed each replay;
the fused helper's captured external workspaces stay alive when a larger prefill replaces
its eager workspace. `FutureMap` issues its optional D2H at publish and waits on that leaf
copy's event at consumption. FA3's GPU-only path still performs no CPU-length copy.
A request arriving after a decode plan is prepared can wait one extra decode iteration
before prefill; the local TTFT checks below pass, while the H200 cap remains unverified.

GPU0/TP1 sm_120, margin fixture, 2,608-token initial prompts, 30 requests x220 tokens:
the `host-step` experiment switches scheduling/KV capture every eight decode steps in one
server. Both modes retain early length copying, isolating those two mechanisms without
changing the relay mid-step. Node-level Nsight records kernels, copies and memsets;
GPU busy is their interval union. Step wall is build-to-next-build cadence, including
output work; moving preparation inside `run_batch` makes that function alone incomparable.
Only steady full batches inside the capture window, away from switch boundaries, are used.
Columns are separate per-step medians and need not add exactly.

| Metric, ms | c1 before | c1 after | c4 before | c4 after |
|---|---:|---:|---:|---:|
| Steps measured | 49 | 47 | 45 | 54 |
| Scheduler step wall | 54.485 | 54.009 | 50.800 | 51.067 |
| GPU busy (kernel/copy union) | 1.870 | 1.874 | 1.975 | 1.980 |
| Gap (wall minus busy) | 52.618 | 52.136 | 48.825 | 49.085 |
| `run_batch` host span | 53.650 | 53.794 | 49.942 | 50.890 |
| CPU batch build | 0.348 | 0.351 | 0.410 | 0.391 |
| Target-hidden -> draft-KV host span | 0.593 | 0.077 | 0.605 | 0.080 |
| Accept host bookkeeping | 0.119 | 0.103 | 0.121 | 0.116 |
| Output processing | 0.154 | 0.165 | 0.198 | 0.206 |

**Verified saving: about 0.52 ms of KV-materialization host work per step. No demonstrated
native c4 end-to-end win.** Native FlashInfer planning, profiler overhead and other GPU0
contexts dominate this trace; its gap is not all host waiting and is not an H200 estimate.
Long separate-server/ABBA timings varied substantially with contention and are not credited
as speedups. The old 2.1-ms idle ranking is withdrawn. Inference: FA3's existing overlap may
already hide much of the removed CPU work; incremental H200 credit remains zero pending a run.
Accept bookkeeping and local FlashInfer planner synchronizations remain; sampling/verify
were already graph-folded.

Local verification on the frozen `harness/trees/host-final` build:

- `host-final-allloads`: **174/174 exact**, c1/4/16/32, every response exactly220 tokens,
  first divergence220, all74 prefill sizes retained and verify replay observed; comparison
  to `q02-stock-tp1-allloads`. `host-final-random` also completes30x220.
- `host-stock-margin-tf2` / `host-final-margin-tf2`: identical220 generated IDs and
  prompt/continuation logprob max/mean drift **0/0** (2,827 prompt-side,220 continuation values).
- Fresh TP1 stock teacher-forcing A/B/C against the common `g15` continuations, compared
  with `host-final-tf`:90 comparisons pass the standard median-within-stock-p90 gate.
  Request-mean drift patch p50/p90 **0.188183/0.198271**, stock **0.185864/0.197916**;
  request-max patch **0.681901/0.813093**, stock **0.647934/0.807272**. Individual maximum
  **1.232753** exceeds stock **0.960120**; the tails are not equal. Random generated-token
  agreement p10 is0 for stock and patch, so that diagnostic alone is uninformative.
- `host-dual-{0-stock,1-patch,2-patch,3-stock}`: both pristine stock and patch resident
  on GPU0, alternating identical cold-cache workloads,120x220 exact. Worst paired TTFT
  p50 ratios c1/c4 **1.55/1.12**, p90 **1.18/0.61**, all within2x. A separate-server c1
  comparison reached7.9x under changing contention; the controlled test resolves attribution.
- `host-final-gpuonly`: Triton attention forces `needs_cpu_seq_lens=False`,5x220 complete,
  exercising the judge-like Python relay branch. This is not FA3 arithmetic coverage.
- Regression tests:96 fresh-input graph replays across all32 batch sizes, exact FP8 cache
  bytes and untouched rows even after workspace growth;100 reordered length-relay cycles;
  GPU-only relay and3 CPU plan/lifecycle tests pass.

Rejected experiments exposed and fixed two lifetime bugs: copied draft inputs must start
with fresh preparation scratch (otherwise its device/capacity check doubles buffers), and
graphs must retain the fused helper's external workspaces across larger prefills. Initial
snapshots outside the container mounts imported the image tree and are excluded. A stock
teacher-forcing startup rejected memory fraction0.3 under contention; the accepted retry
uses0.6 and a100K-token cap. No GPU1, judge run or commit was used.

Reproduction tools: `prof/sitecustomize.py`, `run_trace.py`, `ab_serving.py --step-ab`,
`summarize_host.py`, `dual_ttft.py`, and `run_forced.py`. Evidence under
`/data2/jihoon/muse_kernel/harness/results/`: `host-step-trace.nsys-rep` / `.sqlite`,
`host-step/host_trace.jsonl`, accepted labels above, and `host-evidence/` (summaries,
drift/comparisons, source manifest, `host-overhead.diff`, `worker-hunks.diff`). Own source
patch:13,970 bytes,5 files, longest added line96. The five worker hunks initialize graph
state, capture KV graphs, add the verify-launched callback argument, invoke it after target
enqueue, and dispatch decode KV materialization through the graph.
Regression sources: `serving/tests/test_dflash_host_overlap.py` and
`serving/tests/test_dflash_overlap_plan.py`.

**Unverified:** actual H200 TP8/EP8 FA3 execution, captured drafter-FC collectives, post-capture
KV/memory capacity, held-out agreement and TTFT at all judge loads. GPU0 timings do not prove
an incremental serving speedup; the CPU reduction alone must not be subtracted from H200 steps.

### Prefill/verify overlap: measured locally, no scheduler patch (2026-10-24)

**Decision: no qualified ~5% score gain established.** The current expert producers'
stream behavior prices at **0.17–0.44%** incremental score after request readiness;
the mixed-row prototype prices at **0.32–4.39%**, or **0.16–3.14%** with first-prefill
work scaled to five chunks for80K tokens. These are local-to-H200 model sensitivities,
not measured serving gains. The mixed prototype is not agreement-qualified.
The existing KV host-work cut above is retained; no serving source changed in this experiment.

The premise needs one correction: `SpeculativeAlgorithm.supports_mixed_chunk()` already
includes DFLASH. The judge leaves `enable_mixed_chunk=False` by default; its flags do not
enable it. The existing mixed path advances running rows by **one target token** per
chunk. Its zero-cost modeled score ceiling is only0.76–2.25%. Eight-token DFlash
verification inside a mixed forward needs additional worker/result/KV handling.

**Measurement scope.** GPU0 sm120, full8192/4096 widths,32 local NVFP4 experts,
FP8 paged KV. A chunk is16,384 queries at cached context32,768; each verify request
has8 queries at context80,000. Reuse one immutable layer bank for46 sliding and16 full
layer graph replays; exclude LM head, drafter, two BF16-expert layers and collectives.
The serving producer is V9 for verify and grouped for prefill. The grouped producer's
MMA stand-in runs locally; native WGMMA and FA3 do not. Attention uses the portable
BF16 implementation in `cand_v13.py`. Forward streams have separate inputs, caches,
Marlin scratch, router streams and private graph pools. This is a target-stack proxy.
Stream cases select the synthetic bank's busiest rank independently per forward;
actual EP routing imbalance remains unmeasured. Mixed cases share one rank mapping.

| Requests | Verify steps per chunk | Prefill alone, ms | Verify work alone, ms | Serialized pair, ms | Concurrent, decode priority−1, ms | Concurrent, both priority0, ms |
|---|---:|---:|---:|---:|---:|---:|
| 16 | 1 | 3027.41 | 105.14 | 3240.68 | 3219.15 | 3218.38 |
| 32 | 1 | 3187.64 | 163.91 | 3289.20 | 3320.10 | 3405.41 |
| 16 | 8 | 3321.84 | 921.59 | 4225.28 | 3923.18 | 3925.48 |
| 32 | 8 | 3731.92 | 1434.53 | 4783.60 | 5270.47 | 6100.41 |

These are **separate marginal medians under changing GPU0 contention**; they do not
add. Orders alternate forward/reverse. Six sustained rounds and four single-step rounds
are retained per concurrency. Model inputs use per-round ratios from sustained work:
`eta=(P+D−concurrent)/D` is0.326/0.207 at c16/c32, and verify progress is only
0.245/0.315 of isolated speed. Bootstrap95% intervals for eta are[0.231,0.685] and
[−0.462,0.438]; c32 does not establish positive overlap. Ordinary-priority decode gives
eta0.340/0.059. Earlier stock-Marlin runs overestimated this port's benefit and are retained
as exploratory evidence, including unstable and failed pilots.

**Why complementarity is insufficient.** A fresh sm90a compile of the current V9 reports
124 registers/thread and256 threads/CTA. Its cooperative grid uses
`SM_count * resident_blocks`. Resource accounting predicts two CTAs/SM on Hopper:
63,488 registers before allocation rounding, leaving at most2,048 of65,536. The smallest
compiled grouped-prefill CTA needs7,168 registers; the large W13 variant needs17,920.
Inference: the default resident V9 grid excludes those prefill CTAs during its main work.
Actual H200 occupancy is unqueried. Hardware limits: [NVIDIA Hopper tuning guide](https://docs.nvidia.com/cuda/archive/13.0.3/hopper-tuning-guide/index.html).

SGLang also explicitly shares prefill/decode graph memory and input buffers because
forwards are mutually exclusive (`runner_utils/pool.py`, `input_buffers.py`); a second
stream alone is unsafe. The image has NCCL2.30.7: graph mixing is supported, captured
communication is internally ordered by default, and disabling that ordering together
with mixing is unsupported. TP8 needs rank-consistent launch ordering and independent
collective scratch. [NCCL graph support](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2307/user-guide/docs/usage/cudagraph.html),
[ordering requirements](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2307/user-guide/docs/env.html#nccl-graph-stream-ordering).

**Mixed-forward ceiling probe.** Concatenate prefill and256 verify rows, run shared
dense/expert operations once, and use separate causal attention segments in the local
stand-in. Actual serving expert producers are used. At c32, the paired mixed/prefill
time ratio is **1.0498** for16K and **1.4975** for753 rows; serialized/mixed speedups are
**1.0046x/1.0481x**. Both mixes require more work than prefill alone. The16K mixed
forward has16,640 rows and would also need the serving adapter's16K admission policy
addressed. Inputs are exact; a diagnostic locates the first numerical difference in
QKV GEMM after its row count changes (0.37–0.43% relative L2). Attention/routing amplify
it: the753-row mix has6.4–7.5% layer-output L2 against separate forwards. This is
unqualified numerical evidence, not a teacher-forced gate result.

`sim/serving_overlap.py` adds FIFO per-user prefill→220-token→next-turn dependencies,
13 turns, mean acceptance4.16 and a search over minimum ready batch size. Its comparison
keeps the original capped/e2e score calculation. Small ready batches lose weight-read
amortization; one mixed verify opportunity per chunk cannot hide every decode step.
Mixed costs scale with ready rows and include modeled draft/head/AR work; CPU/KV
materialization cost remains unpriced, so those estimates are optimistic. The first-turn
benchmark contains **three representative chunks**, despite `sim.serving`'s four-chunk
docstring. The80K sensitivity scales that timing by5/3 and uses five opportunities.

| Model basis | Streams, score gain | Mixed, score gain | Streams with80K first prefill | Mixed with80K first prefill |
|---|---:|---:|---:|---:|
| Practical decode, stock prefill | 0.44% | 0.32% | 0.29% | 0.16% |
| Practical decode and prefill | 0.17% | 2.06% | 0.09% | 1.42% |
| Receipts, complete accounting | 0.38% | 4.39% | 0.25% | 3.14% |

**Checked:** isolated versus concurrent serving-producer outputs, route IDs and all KV
bytes are exact across the fresh-input graph checks. A separate full-width stock-Marlin
sanity check covers16K prefill and verify32 on full/sliding layers: relative L2≤0.003933,
exact routes and KV bytes. Model conservation, stock/no-overlap and c1 identities pass;
all primary source hashes still match their files. No GPU1 computation, judge run or commit.

**Unverified:** native H200 FA3/WGMMA overlap, actual TP8 collectives, full-model graph
and KV allocator memory, teacher-forced/logprob agreement,220-token request completion
and TTFT for a new scheduling implementation. H200 mixed attention and a smaller V9
cooperative grid could change the ceiling; these measurements do not rule them out.

Reproduce with `prof/prefill_overlap.py --expert-backend serving --concurrency 16`
or32, `--burst 8 --one-step-reps 4 --layers 62 --reps 6 --out <json>`;
`prof/mixed_prefill_probe.py --expert-backend serving --concurrency 32` at
`--prefill-length 753` and16384; then `python -m sim.serving_overlap` with both
`--measurement` and both `--mixed-measurement` paths. GPU commands use
`CUDA_VISIBLE_DEVICES=0 tools/img python`. Frozen sources, all raw runs, compiler resource
log, producer agreement, model inputs and summaries live in
`/data2/jihoon/muse_kernel/harness/results/prefill-overlap/` (`report.json`, `sources/`).

### Serving upstream ports, expert storage and TTFT admission

**Shared-bank V9 reader loses on the H200 (`6fab497d67e2420b`, serving routing, cold, 3 fresh draws):** reading the shared Marlin bank directly costs 0.20-0.27x stock at every M=8-256 (M=128: 1,430 vs 314 us). V9 in its own layout ties stock at M=8-24 (77.9 vs 79.2 us at M=8) and loses from M=32 (389 vs 314 at M=128, 698 vs 510 at M=256); the shared Marlin path equals stock at M=512-16K. Serving keeps stock Marlin at every M; agreement passes everywhere (rel 0.0057).

**Current source:** `harness/trees/shared-bank-r08-final`, evidence
`harness/results/shared-bank-r01-evidence/`, both under `/data2/jihoon/muse_kernel/`.
The current patch has not run as a full H200 server. No commit or candidate
judge run was made; authorized component labs are queued, with no judge polling.

**Expert storage correct; direct reader loses on H200.** The direct V9 reader
in `kernels/ops/moe/muse_marlin_v9_reader.py` reads stock Marlin's code/scale
arrays. `layers/muse_spark_expert_bank.py` joins them without changing a byte;
stock retains contiguous tensor views into those same allocations. V9 converts
Marlin's stored scale representation inside the kernel. No second resident
layout, timed repack, prepared BF16/FP8 expansion, or MoE-lane source edit is
needed. Bank residency remains905969664B/layer,54358179840B for60 layers;
additional resident expert bytes and timed repack scratch are both0.
Stock's native loading aliases are released before service.

The serving policy remains stock Marlin. H200 `6fab497d67e2420b` rejects the
direct-reader implementation: M8/32/128/256 direct **298/651/1430/2590us** versus
stock **79.2/147/314/510us** (3.76–5.08x slower at those points). Its separate
native-V9 control is77.9/163/389/698us; that different reader cannot admit this
one. Shared-bank Marlin stays528/736/15900us at M512/753/16384, essentially
identical to stock528/737/15900us. `MARLIN_READER_ADMITTED=False` already keeps
the losing reader out of serving. No new serving dispatch or port is justified.

The derived source preserves V9 routing, arithmetic, launch ABI and dense
outputs. The serving global sidecars reproduce **stock Marlin's rounded shared
gate global**, including inversion of its power-of-two bias. Substituting
standalone V9's independent FP32 gate/up globals initially caused a0.021484375
mismatch; the bank itself was not wrong. With stock globals, the reader is
bit-identical to native V9 at M1/8/32/128/256, including all127 positive finite
E4M3 magnitudes, zero/subnormals. Against stock Marlin, relative L2<=0.00538,
maxabs0.03125; it is not bit-exact. Own changed-input graph drift is0. The
production loader/dispatch test covers13 sizes M1..2048, every257–753 boundary,
shifted EP ownership and remote-zero routing. The Marlin path through753 is
bit-exact; larger graph drift is within stock's control. Derived sm90a compile
passes. Logs: `direct_reader_bitparity.log`, `production_reader_layer.log`,
`direct_sm90.log`.

Direct-reader H200 payload `prof/lab/marlin_v9_reader.py` SHA-256
`78be248dda10d8473e42887d4c50572b64faa1b63b1a291f60a2f46744c700c2`, job
**6fab497d67e2420b**, passes23 shapes: all20 verify graph buckets plus512/753/16K
prefill,3 refreshed real-router distributions,20 rotated cold graph medians,
stock-byte equality and native-V9 bit parity. Relative L2 against stock's
normalized output is≤0.00571, maxabs≤0.00781. Extra resident bytes0; lab69.2s,
total prepare196.3s. Correctness/storage pass; performance admission fails.
Evidence: `runs/muse_moe_w4a16/*6fab497d67e2420b.json`.

| Finding (serving only) | H200 result | Decision / evidence |
|---|---|---|
| Serialized shared-bank bridge | `afb3b3649e664cb3`: byte-exact codes/scales, 905969664B workspace, **2500us** cold repack. M8/32/128/256 stock→V9+bridge **113→111 / 164→176 / 317→396 / 513→711us**; M257/512/753/1024/2048/16384 stock→Marlin+bridge **345→2840 / 532→3030 / 745→3230 / 999→3480 / 1880→4360 / 15800→19200us**. Ten shapes ×3 fresh fixtures pass; relative L2≤0.00569, maxabs≤0.00781, 20 cold graph medians | Serial conversion loses except the M8 noise-sized difference. Keep it out of shipping source; the conversion-free direct reader also loses (`6fab497d67e2420b`). `problems/muse_moe_w4a16/lab/profiles/lab_shared_bank_bridge.afb3b3649e664cb3.log`; `prof/lab/v9_marlin_bridge.py` |

**TTFT margin passes locally with interleaved controls.** The accepted code
fixes remain: drain ready first tokens before ingress/planning, remove completed
DFlash rows before admission on ordinary/early paths, preserve stock Triton
finalize at M>=32, and coalesce contiguous staged HiCache copies within each
CUDA registration. Eight deterministic scheduler regressions pass. At c16 the
prior per-request trace decomposes as follows (mean ms;48 requests each):

| TP2 native diagnostic | Queue | Prefill GPU envelope | First emit | TTFT |
|---|---:|---:|---:|---:|
| Stock r30 |1906.38|108.77|648.10|2663.25|
| Before copy fix r22 |1792.73|132.14|20.90|1945.78|
| After copy fix r30 |893.13|157.62|7.47|1058.22|

Envelope includes communication, host launch gaps and stream waits. The trace
is not isolated busy-compute timing. Earlier uninstrumented native TP2 c16
p50=1.757x and c32 p90=2.059x remain failed evidence, not discarded runs;
stock-self c16 varied4.034x/5.029x under concurrent GPU1 load.

`shared-bank-r05` now holds stock/native/portable servers resident and submits
only one traffic stream at a time, cache-flushed per bracket: stockA, native,
stockB, portable, stockC at each rung. Each patch mode must pass against both
adjacent controls. All use mem-fraction0.3 and actual300000 full-KV slots;
TP1 is on GPU0, TP2 on0/1. Every bracket validates actual prefill and verify
replay, all74 prefill graph shapes, DFlash8, stable source/checkpoint provenance,
no tracebacks and exact220 tokens. P50 is `statistics.median`; p90 is the
unchanged harness quantile. No harness files or thresholds changed.

| sm120 mode | Exact220 | Worst p50 c1/c4/c16/c32 | Worst p90 c1/c4/c16/c32 |
|---|---:|---|---|
| TP1 native |174/174|0.833/0.601/0.542/0.292|0.880/0.317/0.235/0.225|
| TP1 portable |174/174|0.858/0.596/0.720/0.321|0.855/0.428/0.248/0.249|
| TP2 native |348/348|0.817/0.319/0.611/0.283|0.875/0.385/0.422/0.416|
| TP2 portable |348/348|0.794/0.694/0.488/0.308|0.831/0.290/0.326/0.394|

The table takes the worse of r05 and full-history r11 matched ratios (below);
there are1044 unique exact patch responses.2615-position and8-generated-token probes
have mean/max drift0. Worst p50/p90 ratio is0.880, below1.5. This is a passing
matched local gate, not exclusive-node evidence: resource logs retain other
GPU jobs. No extra production queue change was justified by these brackets.
The current reader source changes only an unadmitted Hopper path relative to
this timing snapshot. The first sequential run failed startup when other GPU0
jobs allocated30GB during graph capture; the actual300K-capacity check was
not weakened. A first persistent wrapper also failed because `run.main` kills
servers in cleanup; the corrected wrapper uses public launch/run_level/scan_log
and owns cleanup explicitly. Both failures are retained.

A fresh full-grid diagnostic (`shared-bank-r09-trace`, all174 exact) retains a
c16 p90 failure2.071x versus its second stock control (c1 p50/p90=1.568/1.561).
It is not a passing full-grid margin gate. The resource logs expose a topology-
load change: GPU1's foreign PID1541795 held30950MiB at100% during patch startup,
and is absent at stockB startup (GPU1 memory7334MiB,utilization0%). StockA/B
c16 p90 differ5.346x. This explains why those sequential brackets cannot
isolate a patch queue regression; it does not erase the observed failure.
Fresh per-request decomposition (mean ms;48 c16/96 c32 rows per run):

| Load / trace | Queue | Prefill envelope | Emit | TTFT |
|---|---:|---:|---:|---:|
| c16 stockA |1653.99|105.81|738.89|2498.69|
| c16 patch |673.08|108.04|6.22|787.34|
| c16 stockB |333.39|80.11|106.47|519.98|
| c32 stockA |4854.08|162.19|580.72|5596.99|
| c32 patch |1666.00|147.51|8.78|1822.28|
| c32 stockB |1471.72|94.09|242.15|1807.96|

The patch's long c16 tail is first-turn admission/chunk waiting:16 initial
requests average1906.97ms queue versus50.75/61.52ms on follow-ups. Emit is6ms;
the patch queue is between the two stock controls, while prefill envelope is
1.35x the idle-GPU1 control. The trace does exercise620/707-row prefills at c16
and583/726 at c32; all select stock Marlin, with no producer gap. Raw CSVs are
in `shared-bank-r09-trace-evidence`.

**Full-history TP2 bracket passes** (`shared-bank-r11-full-history`): separate
stockA/patch/stockB servers retain independent c1→c4→c16→c32 cache histories,
with no intermediate flush/probe. Native/portable phases each use three resident
models to leave GPU1 memory for its other project. Each patch mode passes
against both controls at every rung:

| Full-history mode | Worst p50 c1/c4/c16/c32 | Worst p90 c1/c4/c16/c32 |
|---|---|---|
| Native |.817/.318/.382/.209|.875/.171/.422/.364|
| Portable |.794/.300/.488/.182|.831/.247/.326/.381|

All348 patch responses are exact220, final2615-position/eight-token probes
have0 drift, actual300K capacity/full graph checks pass and source stays stable.
The combined standard-margin count is1044, and the worst matched ratio across
both protocols is.880. This does not discard r09's changing-load failure or
claim an isolated H200 latency result. No new runtime queue hunk was justified.
The five-server setup r10 was stopped during warmup to limit aggregate GPU1
residency; no test pass was claimed. Reproduce with
`full_history_ttft_brackets.py`; `final_gate_summary.json` joins both protocols.

**Real-reader server agreement passes:** a separate six-layer fixture has real
4096-wide NVFP4 experts,32 local experts per rank, nonzero random packed
weights and unequal input gate/up globals. The target hidden width is512,
so this is not the full released checkpoint. A test-only hook forces the
actual reader on sm120, with no stand-in arithmetic. Both TP1/TP2 are174/174
exact at220 tokens, first divergence220/token agreement1; all prefill/verify
graph checks pass. TP2 logs12 joined-bank storage assertions per run, including
local offsets0/32. Thirty-request random gates use three stock controls per
TP size and220 canonical teacher-forced continuation positions each (90 pairwise
comparisons). They pass the unchanged median-within-stock-p90 gate:

| Actual-reader fixture | Stock request-mean p50/p90 | Patch request-mean p50/p90 | Stock request-max p90 | Patch request-max p50 |
|---|---|---|---:|---:|
| TP1 |0.172189/0.181281|0.175243/0.222008|0.756927|0.668917|
| TP2 |0.179670/0.194040|0.183148/0.213330|0.797136|0.696103|

Random first divergence and token agreement remain0 for stock-self and patch,
so exact counts remain nondiscriminating. Patch tails are wider, not identical.
Canonical220-position teacher forcing scores the continuation via **prefill**;
it does not isolate V9 verify-logit drift. Unforced generation and layer tests
exercise verify. H200/full-checkpoint verify agreement remains required.
The actual-reader margin's single-control p50 ratios c1/4/16/32 are
TP1 .937/.822/.711/.754, TP2 .771/.546/.669/.310;
p90 .985/.685/.684/.714 and .888/.347/.471/.495. All are<=1.5, but the resident
interleaved brackets above are the primary TTFT gate. Files:
`shared-bank-r07-evidence/`, summary `actual_reader_server_summary.json`.

**Other accepted integrations retained:** router handoff shares4.46MB/stream,
recovering modeled2.759GB/GPU;24 fresh two-stream router graphs are exact and
both hunks reverse-apply. Upstream #45015 pooler512/unpooler64, #45895 existing
FC shard gather<=64, #44401 held-row admission, #46192 dead scales and #45447
tiny-prefix graph bucket remain wired. TP2 same-ABI eager/graphs are exact;
actual TP8 multicast remains unverified. `DFlashDrafterPortfolio`, prefill dense
and GEMM modules remain unbound pending their own H200 receipts. No runtime
receipt lookup is used.

**Radix top-k rejected:** BF16-valued cutoff ties change selected support.
Continuous FP32 is exact, BF16 logits probability maxdiff0.01289545, flat logits
**0.015625** versus stock self-drift0. Graph replay itself is exact. Stock
`torch.topk` remains production; rejection test/log are retained.

**Simulator:** tau4.05 replaces4.16 in both serving models. Capped composition
1.807103017→1.812740536 (+0.311964%); extrapolated1.819826481→1.825689238
(+0.322160%). Both stock/candidate assumptions change. This is not current patch
score: the composition still credits canonical-bank V9 and omits measured host
copy savings. The shared reader's performance remains unpriced.

**Patch:**359061 bytes,38 source files, maximum added line374 characters/bytes.
All36 accepted paths remain,30 byte-identical; incoming MoE source is unchanged.
Clean-base apply, AST/Ruff F821–F823, whitespace, worker-hunk and router-handoff
reverse-apply checks pass. The final snapshot includes incoming unbound prefill
changes: two module-docstrings plus removal of the obsolete256x scale factors.
The latter has an independent Torch-reference eager/three fresh graph test
(SwiGLU maxabs3.73e-9,down0,padding unchanged); both epilogues compile for
sm90. Full reader server tests use
r06; those incoming modules have no production import sites. Full H200 TP8/EP8 FA3 arithmetic, graph residency,
released-checkpoint agreement and hidden-workload TTFT remain unverified.

H200 copy diagnostic `3828ee433f0f43c3` (payload `prof/lab/hicache_coalesced.py` `7094d943`): every case byte-exact. Coalesced copies complete in 73.5 us / 1.27 ms at P=753 / 16384 pages (L=16) and 187 us / 3.62 ms (L=46), against stock's 3.33 / 76.0 ms and 3.53 / 82.0 ms; enqueue drops from 2.8-80 ms to 10-20 us. Measured on the judge.

### KV residency and long-context capacity (2026-10-24)

**Decision: bound Muse's SWA reservation by live windows and share router scratch.**
With the receipt-selected projection layouts, the old patch leaves34.110GB before
dynamic allocations; c32/80K needs37.880GB under the default0.8 SWA/full split.
It cannot fit even with zero graph overhead. This audit uses the requested nominal
141GB decimal H200 and0.85 fraction:119.850GB. Actual CUDA-visible capacity and
pre-load CUDA/NCCL occupancy still require a full-node measurement.

**Source-count residency, GB/GPU at TP8/EP8.** The frozen source is
`harness/trees/kv-audit-r0`; the candidate is `kv-audit-cap-r2`, both under
`/data2/jihoon/muse_kernel/`. `prof/kv_capacity_report.py` reuses the weight count in
`serving/tests/muse_residency.py` and adds every active layout and scratch category.
The released target configuration remains gated; vision/audio defaults and distinct
target RoPE tables are assumptions, not measured checkpoint residency.

| Resident component | Stock GB | Patch before GB | After router handoff GB |
|---|---:|---:|---:|
| 60 NVFP4 expert banks, codes + FP8 block scales |54.358|54.358|54.358|
| Two BF16 expert layers |6.442|6.442|6.442|
| Target QKV/gate/O, pooler/unpooler, FP32 router and vocab |13.050|13.050|13.050|
| Router hi/lo weights |0.520|0.520|0.520|
| DFlash attention/MLP + FC |1.741|1.153|1.153|
| DFlash dynamic convs, selector tables, norms |0.547|0.547|0.547|
| Vision + replicated audio defaults |0.721|0.721|0.721|
| Target norm/gate weights and effective caches |0.018|0.018|0.018|
| One target + one draft FP32 RoPE table |0.805|0.805|0.805|
| Existing TP push/pull plane and graph pointer table |0.159|0.159|0.159|
| Selected full pooler NN + O128 NN transposes |0|5.201|5.201|
| Router split-K scratch, all20 verify capture sizes,62 layers |0|2.763|0.00446 per routing stream|
| **Subtotal for those selected plans** |**78.363**|**85.740**|**82.981**|
| Additional selected NN layouts: full unpooler / two local TP slices |0|0–4.161 /0–1.040|same|
| One Humming candidate bank, only if its warmup selects it |0|0–0.906|same|
| Unbound `muse_prefill_dense`, `muse_prefill_gemm`, `muse_dflash` |0|0|0|
| Removed experimental verify/prefill attention paths |0|0|0|

These are selected-plan scenarios: startup tuning may choose fewer transposes. No
unselected NN layout survives `MuseProjection.prepare`. Its source/detached weights
alias loaded parameters. The32MiB Lt workspace is temporary and reused by the graph
allocator; it is not32MiB times62 layers or74 prefill shapes. The router's high/low
views alias its pair allocation and the model reuses those views. Stock already
reserves the8MiB collective push slots; fused finalization aliases that plane.

The full-width local bank census corrects a tempting but false source inference:
Marlin temporarily converts scales to BF16, then `nvfp4_marlin_process_scales`
returns FP8. Stock and canonical banks are both0.906GB/layer; there is **no6.04GB
scale saving**. The patch's steady weight saving is chiefly the0.587GB sharded
DFlash FC. Production deletes the native NVFP4 tensors after canonical repacking;
the measurement proxy intentionally retains both representations for comparison.

**Dynamic residency must also be charged.** The following allocations are additional
to the subtotal; call their aggregate U, avoiding double-counting graph outputs:

| Component | Source / local measurement and H200-shape implication |
|---|---|
| Target prefill/verify graph pools, MoE scratch and five hidden taps |62-layer full-width local proxy: stock9.278GB, patch8.022GB reserved in one shared pool; verify32 adds no reservation. Includes five1.342GB total taps, layer outputs and scratch. Uses one immutable NVFP4 layer bank, identity collectives and portable attention; the two BF16 edge-layer graph peaks are not measured full-width. This is not the H200 breakable-graph total. Do not multiply by62 or74.|
| Draft graph pool and graph I/O |Tiny server target/draft/aux graph pools total2.640GB stock and2.621GB patch; these totals must not be substituted for H200. Both models share the main global capture pool; draft extension and committed-context graphs may own separate pools.|
| Prefill shared I/O |One `[16384,8192]` BF16 input-embedding buffer0.268GB; shared output holds final hidden plus five taps,1.611GB. Those returned taps may already be counted in a graph-pool measurement. Verify output logits are TP-sharded.|
| DFlash KV helper and graph inputs |Full-width flat KV weight copy0.02097GB;16K helper projection/K/V workspaces0.08389GB; pinned old M256 workspace0.00131GB; committed-context graph hidden input0.02097GB plus small index/length tensors. Keep pinned references: prefills replace the helper workspace while captured graphs retain its old addresses. Private graph-pool reservation remains dynamic.|
| Request maps, allocator free lists and eager input slots |Local request map `[33,131084]` int32 =0.01730GB. Full-to-SWA map and free lists scale with pool rows; eager IDs/positions/locations/mRoPE buffers also scale with full capacity. Source baseline is about64B/full row +8B/SWA row, in addition to the request map. Recompute after changing full capacity.|
| FA3 metadata and library storage |Target full/SWA plus draft32x131072 int32 page tables alone are about0.050GB. cuBLAS, CUDA graph driver state, NCCL and alignment/allocator slack need actual TP8 measurement. Local FlashInfer's workspace and KV-index arrays are not H200 FA3 allocations.|
| Additional distinct target RoPE theta |0.268GB per distinct64-wide FP32 table.|

The Humming admission guard charges pending VMM KV backing, but a selected copy still
uses memory at post-capture resizing. Its0.906GB is conditional, not free. Keep its
selected speed path and all projection winners; the window split removes the larger
capacity risk without removing kernels.

**Pool policy and need.** Actual stock and old-patch runs confirm a hybrid SWA pool
with0.8 SWA rows per full row. `SWAChunkCapPoolConfigurator` does not apply while the
judge's prefix/hierarchical cache is enabled. Thus the real default cell includes
`16*256 +0.8*46*256 +1280 =14796.8 B/full row`, counting DFlash. Charging every
target layer for full history instead gives15872B target +1280B draft; that is a
different, more conservative scenario, not the allocator's implemented split.

The new `MuseKVPoolConfigurator` keeps the standard live-request cap, including
eviction interval128, two prefill chunks in flight and speculative preallocation,
plus two cached-fragment windows per request. Under current flags it reserves262657
SWA rows =**3.093GB/GPU**, independently of full history capacity. The cache frees
out-of-window request rows and can evict unlocked SWA branches independently of full
history; a match locks fragments until they cover the trailing window. Its boundary
fragment can overhang by another window. The extra64 windows cover both cached
fragments while extensions run. Explicit operator SWA ratios remain honored.
Muse exposes2047 previous positions (2048 including the current position); the
page-size-one padding in the cap covers the current position. The full side
charges **4096+1280 =5376 B/row**, including every private DFlash row.
The generic existing chunk-cap arithmetic omits `_draft_cell_size`; reusing it
unchanged would overallocate DFlash memory. The new configurator explicitly charges it.

| c | Default split need80K /131072, GB | Bounded split need80K /131072, GB | Maximum U that fits after router sharing,80K /131072, GB |
|---:|---:|---:|---:|
|1|1.184 /1.939|3.523 /3.798|33.346 /33.071|
|4|4.735 /7.758|4.813 /5.912|32.056 /30.957|
|16|18.940 /31.031|9.974 /14.367|26.895 /22.502|
|32|37.880 /62.062|16.856 /25.642|20.013 /11.227|

The bounded need includes the entire262657-row SWA reservation even at c1. At80K
history plus753 new and220 output tokens, c32 bounded need is17.023GB. The131072
column includes generation. Without the router handoff, subtract2.759GB from the U
allowance; if all optional NN layouts survive tuning subtract another5.201GB, and
subtract0.906GB if Humming is selected. These allowances are capacity conditions,
not proof that all32 maximum-length contexts fit on H200.

Before U, stock's default pool upper bound is2.804M full rows; the old patch's is
2.305M. With the bounded split and router sharing, the patch's upper bound is6.283M
full rows, decreasing by0.186M per GB of U. Required full rows at c1/4/16/32 are
0.080/0.320/1.280/2.560M for80K, or0.131/0.524/2.097/4.194M at131072. Post-capture
resizing measures actual free memory, reserves the static-fraction headroom and
rebuilds target/draft pools; it does not assume that U is zero.

**GPU0 local verification.** All use the tiny model, TP1, native FlashInfer, the
judge's other flags and an explicit4.3M full-row test cap. This cap makes runs
comparable; it is not an H200 pool-size observation. The hook synchronizes at census
boundaries, so these timings are checks rather than a serving speed claim.

| Snapshot after capture / resizing | Stock | Old patch | Bounded patch |
|---|---:|---:|---:|
| Full rows / SWA rows |4.3M /3.44M|4.3M /3.44M|4.3M /262657|
| Target logical KV, GB |7.485|7.485|2.605|
| Draft logical KV, GB |5.504|5.504|5.504|
| Torch allocated / reserved, GB |18.029 /20.867|18.067 /20.883|13.185 /15.991|

The bounded two-turn80K run completes106/106 responses at220 tokens; every response
is exact against matched stock. All53 second turns reuse80008 cached prompt tokens;
there are no retractions and logged SWA use peaks at41%. TTFT p50 patch/stock is
1.052 /1.062 /1.032 /1.078 at c1/4/16/32. All four rungs stay below the2x gate.
P90 ratios are1.017 /1.003 /1.005 /0.858. The final near-cap run completes32/32
requests at220 tokens with130848 prompt tokens (131068 total), no retractions,
and36% peak logged SWA use. The rejected earlier edge test requested130852+220;
stock's `context_len-2` response clamp legitimately shortened those to218. Its
failed220-token check is retained in `kv-capacity-cap-131k-r1`; it is not a KV failure.
The final teacher-forced probe matches the retained stock repeats
`margin-lp-tp1-stock-a/b`:2615 non-null input logprobs and8 forced continuation
logprobs have mean/max drift0, versus stock-repeat max0. It uses a2608-token
prompt on the Triton backend; it is separate from the native FlashInfer80K run.

The router allocation census runs all20 capture sizes through62 real full-width
router instances: scratch is2,763,037,440B before and4,456,512B per stream after.
All eager outputs and changed-input graph replays agree exactly. Two capture streams
run concurrently with separate scratch, and completion counters return to zero.
The current Hopper source accepts the two-hunk handoff with `git apply --check`:
`prof/kv_capacity_router.patch`; it has not been edited in the live kernel lane.
The native sm120 pipeline ran; native sm90a WGMMA remains unverified.

Production changes: new `srt/model_executor/muse_kv_capacity.py`; one four-line
factory hunk in `srt/model_executor/pool_configurator.py`; new
`srt/managers/muse_workspace.py`. The live kernel file is untouched. The router
handoff has exactly two hunks in `kernels/ops/muse_spark_v12/projection_hopper.py`:
delete instance `self.scratch`, then replace per-M allocations with the shared
workspace call. Both hunks are applied only in the tested frozen candidate.
No worker/model/expert-kernel hunks were made. The live owned files match the
final candidate byte-for-byte. `capacity-source.patch`, `router-handoff.patch`
and `source-manifest.json` preserve this boundary in the evidence directory.

Reproducers: `prof/kv_capacity_hook/sitecustomize.py`, `prof/kv_capacity_components.py`,
`prof/kv_capacity_report.py`, `serving/tests/test_muse_kv_capacity.py`. Evidence is in
`harness/results/kv-capacity-components/`, `kv-capacity-stock-r2/r3`,
`kv-capacity-patch-r2`, `kv-capacity-cap-r2`, `kv-capacity-cap-131k-r3` and the teacher-forced labels.
The smaller197121-row prototypes are retained as `kv-capacity-cap-r1` and
`kv-capacity-cap-131k-r2`; the final cap includes radix-fragment overhang.
All paths are under `/data2/jihoon/muse_kernel/`; local launcher logs are in
`runs/screen/kv-capacity-*`. No judge run, commit or GPU1 computation was used.
The required upstream refresh also records the organizers' check-quota change
20→10/day for attention, decode layer and prefill; serving flags did not change.

**Unverified:** full-checkpoint H200 TP8/EP8 non-KV and graph-pool residency, actual
automatically sized pool capacity, startup-selected optional layouts/Humming bank,
FA3 and NCCL memory, and held-out full-model agreement/TTFT. The80K c32 capacity
margin is substantially larger; the131K margin still depends on measured U.

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
G3 allocation is not proposed across all62 serving layers. Prepare drops
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

### Decode large-load profile and round-3 dispatch (G3 priority)

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

#### Glue fusion diagnostic (G3)

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

#### Union small-shape overhead and merge2 control (G3)

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

#### Best portfolio static repairs (G3)

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

#### Larger-load stage floors and non-MoE gap (G3)

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

#### Megakernel correctness audit (G3)

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

#### Batch-1 stage accounting and megakernel rationale (G3)

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

### Prefill evidence and serving path (G2, G1)

Prefill is rank6/P2 under R-20261024-1 and is excluded from the overall kernel board. Its kernel
results feed serving's 753–16K prefill path. The current H200 control is v11, not v3.

| Finding | Result / scope | Mechanism or remaining limit | Evidence |
|---|---|---|---|
| Current best | Check `6a9cd943fe404d74` 1.367883x; scored `965a9a816d4744ea` 1.366811x. Check loads 1.216 / 1.364 / 1.543x; scored 1.227 / 1.356 / 1.534x | BN128 WGMMA below8192; dequant + grouped BF16 DeepGEMM at16K. Source SHA256 `5162da15f86b3ab5d28fe9cd4db9d0dbce778e1bdace526ab87e7f69acfc643e`. Last board snapshot rank46,1.3668x; a snapshot, not a live rank | `runs/muse_prefill_layer/20261023T110413Z_check_6a9cd943fe404d74.json`, `20261023T162945Z_submit_965a9a816d4744ea.json` |
| H200 turn1 ranking | Weighted scored profile: gate/up22.505ms > dense/router12.858 > down11.158 > attention8.273 > boundaries2.582; total59.992ms | Expert total33.663ms vs inferred BF16 tensor floor8.501ms (~25% of peak-equivalent work); dense12.858 vs9.306ms; attention8.273 vs3.228ms. Profiles are individual calls, not timing medians | `problems/muse_prefill_layer/profiles/promoted_6a9cd943fe404d74.965a9a816d4744ea.txt`, `dev/v11_profile_ranking.json` |
| H200 turn4 ranking | Gate/up53.205ms > dense/router48.161 > attention28.726 > down24.408 > boundaries10.208; total170.452ms | Expert77.613ms vs modeled floor34.633ms; dense48.161 vs37.223ms; attention28.726 vs12.911ms. Gate/up remains the largest visible kernel | Same scored profile; `problems/muse_prefill_layer/dev/V11_PROFILE.md` |
| H200 first-turn ranking | Grouped expert GEMMs744.857ms > dense kernel family485.201 > FA3264.493 > dense TNT family262.849 > dequant172.280 > boundaries157.742 > SwiGLU67.549; total2250.498ms | All dense/router781.297ms; expert GEMM+dequant917.137ms. Expert GEMM tensor floor573.812ms (~77% peak-equivalent); dense607.427ms; attention114.422ms. Per full64K chunk, FA3 alone8.431ms is largest | Same scored profile; official weights138sliding +3*16full; bound model in `docs/reference/bound.md` |
| Profile/floor scope | Replay-adjusted tensor floors per weighted load:21.035 /84.767 /1295.661ms. Scored profile totals are2.852 /2.011 /1.737x those floors | The bound's raw table adds replay overhead. Routing floors infer mean routes (~1.8/token), not the profiled invocation's actual counts. The model still charges PV at BF16; allowed FP8 PV can lower its attention term. H200 has no FP4 MMA; expert products remain BF16 | `sim/bound.py`, `problems/muse_prefill_layer/dev/rank_v11.py` |
| dg_w8 at 16K, per stage (H200) | Lab `13a69b9e66504e20` (T=16384, 34,646 local routes over 13 touched experts, padded to 35,712 rows): gather+norm 111, dequant W13 300, GEMM W13 2,888, SwiGLU 272, dequant W2 153, GEMM W2 1,452, combine 150 us = 5,479 us; raw17 7,442, raw_v11 5,855 | The grouped BF16 GEMMs are 79% of dg_w8; dequant + glue 986 us is the rest. Serving-only until the close | `problems/muse_prefill_layer/lab/profiles/lab_prefill_dg_decomp.13a69b9e66504e20.log` |
| Expert banks are a resource | Canonical V6: W13 `[32,512,36864]`, W2 `[32,256,36864]`;905,969,664bytes/layer,50.625GiB across60layers/GPU | Separate FP32 gate/up/down scales. Board selection has no bank-count or scratch objective: retain every useful lossless layout and choose by speed. Serving can also retain copies where their throughput gain outweighs displaced KV capacity; the current canonical port is an implementation, not a one-bank rule | R-20261024-2; `docs/reference/expert_layout.md` |
| Canonical reader comparison | Cold-L2 sm120 RS128 / V6 pipeline:753=.979/3.934ms,3012=2.352/4.218,16384=9.805/18.073; RS128 also wins one-hot routing | Local MMA.sync emulation cannot choose an H200 WGMMA winner. RS128 adaptive is the provisional serving reader. Independent FP32 refs, unequal scales and fresh routing replay pass; M8/128/256 zero/all-local boundaries pass; unnormalized intermediate/down errors .003%/.029% prevent norm cancellation from hiding scale errors | `problems/muse_prefill_layer/dev/canonical_comparison_cold.json`, `canonical_comparison_hot_cold.json`, `canonical_edges.json` |
| FP8 long-context attention | Local16K-query prefixes32K/64K: BF16 Triton20.257/38.839ms, FP8 direct13.357/25.066, gatheredBN12812.890/23.721; not FA3/H200 ratios | Every key is processed; FP8 QK/PV, FP32 online softmax, probability scale448. Fresh full16K graphs have .86–1.19% sampled FP32-reference L2. Main gatheredBN128 has no SM90 stack; BN256 has424B/thread. H200 and serving agreement remain open | `problems/muse_prefill_layer/dev/attention_fp8_bench.json`, `attention_fp8_verify_long.json`, `attention_fp8_compile_tiles.json` |
| Expert port boundary | `srt/layers/muse_spark_moe.py`: `prepare_layer`, `run_layer`, `_grouped_call`; called from `MuseSpark1xFp4Experts._forward_marlin` in `srt/layers/muse_spark_fp4_moe.py` | Share canonical bank/GS sidecars; retain dense-slot/finalize interface and EP collectives. Prefill metadata can reuse router counts/positions. Preserve BF16 effective norm, pre-expert norm and BF16 layers0/61. JIT multiline source under `python/sglang/`; no cubins or board-file giant string lines | `serving/sglang/python/sglang/srt/layers/muse_spark_moe.py`, `docs/reference/expert_layout.md` |
| Attention/dense port boundaries | `FlashAttentionBackend.forward_extend` after cache writes/full-SWA selection; fused QK prep at `MuseSpark1xAttention.forward`; dense dispatch at `bf16_gemm_dispatch` / `quantization/unquant.py` | Serving needs per-request `cu_seqlens_q`, query/context lengths, padding, reserved KV slots and head strides; apply the 5.473 QK factor after E4M3 query rounding; K/V descales follow QK/PV. Residual/norm helpers map to `FusedPostNormResidualPreNorm` and `FusedRMSNormGate`; preserve collective completion | `serving/sglang/python/sglang/srt/layers/attention/flashattention_backend.py`, prefill sources |
| Historical portfolio queue folded | The deleted synthetic-tuned expert, dense/router, attention and canonical portfolio sources, their admissions, and their old hashes are historical evidence in `3bdbe49^` | The live queue is the three routing-aware complete portfolios listed below; no deleted source should be resent | `3bdbe49^`, `problems/muse_prefill_layer/dev/NEXT_CHECKS.md` |
| Tuner rejection contract | Every new/rebuilt tuner catches/logs graph-build, oracle, timing and finalist errors, drops the variant and continues; explicit failure injection covers all4 phases. Shared `builder_safety.py` updates both older builders without touching queued outputs. Coordinator's queued fused source is5241415, distinct from earlier c18fa82 admission | Independent FP32/BF16-semantic math checks use public3% global floor/5x row formula and exact public routing allowances, replacing arbitrary incumbent-L2 gates. FA3 disagreement is diagnostic; only the judge's query-dependent full-layer bound owns attention admission. Graphs, temporary FP32 oracles and synthetic input caches never survive prepare | `problems/muse_prefill_layer/dev/safe_score_tuning.py`, `builder_safety.py`, `tuner_rejection_validation.json`; `checks.py`, `reference.py` |
| Routing-aware prefill expert tuning (sm120 diagnostic) | The new tuner routes fresh random BF16 hidden rows through the prepared layer router hi+lo plus bias, chooses the busiest 32-expert range, and builds exact local slots/counts. Synthetic → realistic route totals/distinct: 753: 1,506/32 → 1,872/11; 3,012: 6,024/32 → 7,634/14; 16,384: 32,768/32 → 41,784/15 | Latest portable fused run changed `fused_bf16_64`→`raw_v11` at 753 (5,130.2→2,200.7 us), kept `raw_v11` at 3,012 (6,417.4→4,757.5 us), and kept `raw_v11` at 16,384 (16,293.9→19,169.3 us). A repeated sm120 run can flip close choices, so these are route/timing evidence rather than an H200 claim; H200 prepare repeats both fixtures with cold-L2 paired timing and drops failures | `problems/muse_prefill_layer/dev/compare_routing_local.py`, `dev/routing_local_sm120.json`, `cand_fused_expert_portfolio_routing.py` |
| Routing-aware portfolio sources | Three immutable complete files retain every existing expert, dense/router and attention choice, but dispatch experts from the router-derived fixture: score (32 choices), fused (45), and dense-fusion (45 plus structural front/back). Every graph/oracle/timing/finalist failure is logged as `SCORE_REJECT` and removed | All three pass 5/5 official tests and 8/8 sm120 graph shapes; latest local prepare is 4.2–4.4 s with Hopper tuning dormant. H200 execution and selection remain unverified until orchestrator sends a frozen source | `runs/screen/prefill_routing_round8/`, `dev/build_routing_aware_portfolios.py` |
| First-turn stage rates | Scored 16K weighted profile: expert GEMM+dequant 917.137 ms over a 573.812 ms BF16-equivalent floor = 618.8 TFLOPS; dense/router 781.297 over 607.427 = 768.9 TFLOPS; attention 264.493 over 114.422 = 427.8 TFLOPS, using H200 dense BF16 989 TFLOPS | Expert has the largest excess, 343.325 ms; dequant and GEMM must overlap before chasing dense or FA3 fixed work | `problems/muse_prefill_layer/dev/v11_profile_ranking.json`, `V11_PROFILE.md`, scored profile receipt |
| First-turn overlap candidate (sm120) | `cand_firstturn_overlap_grid4.py` adds grid-4 persistent grouped BF16 expert variants and a grid-4 overlapped W2 expansion variant to the routing-aware fused portfolio | 5/5 official tests pass; prepare 8.2 s; local sm120 timing is only a screen and leaves H200 choice unverified | SHA256 `a97aab65efeedad63f5d8c82aa47a0f85bb87f0d841dabb0136de1a2d1c283f5`, `runs/screen/muse_prefill_layer.cand_firstturn_overlap_grid4.json` |
| Expert TMA/grid candidate (H200) | `cand_expert_tma_cluster.py`, SHA `00d91b14…`, ran as **d318167eedfd4c19**, scored **1.419x**, turn1/turn4/first-turn **1.226/1.453/1.603x**; its re-send was a judge duplicate | Lost to a30's 1.434x. The extra raw17 choice changes the persistent launch grid; it does **not** implement a 2-CTA activation multicast or a producer-dequantized BF16 smem tile. Its local pass did not establish those mechanisms. Keep this distinct from the pending mixed-input kernel | Corrected ledger/receipt `d318167eedfd4c19`; source SHA `00d91b14b74f899639b9978dde63b9d42bbca967051f98c0f231ea5bf90b784c` |
| Prefill expert H200 sweep (measured, prepared fixture) | Job `b964cb5d180a463b`:753 raw17 **577.760us**, grid4 **581.504**, masked **635.776**;3012 masked **1254.912**, mid **1277.568**;16384 dg_w8 **5167.232**, masked **5191.328**, hinted **5425.312**. Routes **1451/5925/31771**, touched experts **13/14/15** | **Warm eager minimum-of-two event samples, not cold paired graph evidence.** Both this lab and board0d738 use the same generated router-fixture algorithm, but RNG history and potentially the run's router fields differ. V9 never ran: the serving-only import is absent from the image; do not infer a V9 time. New candidate defaults and paired fresh-route selection are recorded below | Receipt `runs/muse_prefill_layer/*b964cb5d180a463b.json`; payload SHA `e827828470e0ff3c`; immutable source SHA `eb0ccfc82bb78c1d03c7ac92b67859cdcb6ee246336af25829d8f8cf810b3501`; comparison `dev/sweep_dispatch/receipt_comparison.json` |
| Exact-route Humming mid receipt / sweep candidate | `cand_mid_experts2` receipt **0d738799802340fd** passed at **1.438402x**. Prepare selects masked Humming at753/3012 and dg_hint_w8 at16K. Rounded cold-stage us: masked **486/960/4360**, raw17 **523/1090/5010**, dg_w8 **879/1300/4350**, mid **529/1010/4350** | Exact-M and TMA mid plans do not beat masked at the two smaller prefill loads. Do not port decode's synthetic-weight Humming timings; this is the separate prefill full-layer receipt. The combined sweep candidate remains queued as owner-fixed **47b38920…** after five closure-`del` removals, 5 tests/8 graphs/static pass | `runs/muse_prefill_layer/20261024T043516Z_submit_0d738799802340fd.json`; `cand_prefill_expert_sweep.py`, SHA `47b38920e76e5f97f280164a3fab96bd64d6d2bbb96a7c170e55c55bef1afb45` |
| Serving first-turn expert path | Serving case `70206a2a501047eb` over the resident canonical V9 bank measures raw17/DG-W8 at **399.6/532.7us (M512), 532.4/614.9us (M753), 1454.8/1374.3us (M3012), 7811.9/6107.4us (M16384)**; Humming is not installed without a measured canonical-to-Humming repack margin | Dispatch raw17 at <=753 and per-call canonical unpack/dequant + DG-W8 at >=3012; no expanded BF16/FP8 bank is retained | `tools/h200_lab.py read 70206a2a501047eb`, `serving/sglang/python/sglang/srt/layers/muse_prefill_expert.py` |
| Serving-owned expert adapter | `PrefillExpertPlan.__call__` is the single producer entry point: BF16 pre-expert-normalized `[T,4096]`, FP32 scores `[T,8]`, global IDs `[T,8]` -> BF16 dense slots `[T,8,4096]` plus int32 local map (`-1` for remote); existing finalizer owns route weighting/post norm | Eager and CUDA-graph stock-Marlin agreement passes at **T=753/3012/16384** with relative L2 **0.00544/0.00536/0.00535**; graph replay is identical. Canonical bytes/scales are unpacked and dequantized only in the timed call | `serving/sglang/python/sglang/srt/layers/muse_prefill_expert.py`, `serving/tests/test_muse_prefill_expert.py` |
| P9 note fold | Historical `RESULTS.md`, `PORTING.md`, and `dev/LIVE_STATUS.md` facts are represented here: v1–v17 local/graph admission, v11 H200 control, expert/attention/dense serving hook boundaries, resident-bank and scratch caveats, and no serving agreement run | Old per-problem notes are intentionally absent from the working tree; `FOCUS.md` is the ≤30-line lane entry point. The orchestrator owns sends/commits; no promotion handler or judge process is used | `problems/muse_prefill_layer/FOCUS.md`, `dev/NEXT_CHECKS.md`, `docs/reference/expert_layout.md` |
| Serving KV need for prefill bank choices | Serving flags fix TP8/EP8 FP8 KV at 15,872 B per cached token/GPU, static budget 119.85 GB (`--mem-fraction-static 0.85`): c1/4/16/32 at 80K consume 1.270/5.079/20.316/40.632 GB/GPU; at the 131,072 cap consume 2.080/8.321/33.286/66.572 GB | A full alternate expert bank costs 54.358 GB/GPU (3,424,785 target-KV-token equivalent), so c32 already cannot retain one extra full bank before other allocations; partial per-shape copies remain a score resource only where the actual KV pool permits them. Chunk 16,384 adds 0.260 GB/request and 220 generated tokens add 3.492 MB/request | `serving/RULES.md`, `docs/reference/expert_layout.md`, `problems/muse_prefill_layer/dev/kv_budget.json` |
| Prefill glue inventory /portfolio | Direct a30fb411/99820fa6 named profiles: first-turn boundary pair851.7–855.2us/call against671.1us traffic floor (159ms /125ms per load). Separate dequant633.6–651.3us, SwiGLU245.5–245.9us, combine133.7–134.0us, gather+norm97.8–99.5us, routing42.0us. Omitted top12 entries remain N/A | Parked by Jihoon under `dev/parked/` after~34ms ideal first-turn headroom (~1.5%), local fused-GEMM losses and a~2h queue. Full-layer `cand_glue_prefill.py` retains prior H200 controls and times last-CTA GEMM/residual/norm, pooler-norm with norm-free expert consumers, fused router GEMM/top8, vector boundary/topk choices. All78 trials/36 finalists reached in48.111s sm120;5/5 tests +8/8 graphs; forced all-fusion5/5 and60 direct configs pass;60 SM90 specializations compile. Local GEMM fusions lose; small vector/topk gains only, H200/serving gain unverified. O fusion cannot cross serving TP all-reduce | SHA2562a2e3914ea26756fc2df828f8c58cc88d0157171dcff880dec50c55290ebdbef; `docs/reference/prefill_glue.md`, `problems/muse_prefill_layer/dev/glue/README.md`, `runs/screen/glue_prefill/` |
| Dense attribution corrected; targeted experiment built | Direct H200 `dafbcad024fe4dcd` role timings at M16384: qkv766–806us (~807TF/s,82%peak), pooler1.36–1.47ms, unpooler1.34–1.45ms, o338–406us, hi+lo router178–204us. These supersede the old438TF/s /44% qkv and146.5ms-gap inference | Identical kernel names aggregate different GEMMs; no causal cache claim follows. **d318167eedfd4c19 ran cand_expert_tma_cluster, not cand_v11_gemm** (okc_run attribution bug, fixed ccbb37a). The a30 dense choices are now exposed through the validated serving GEMM/dense adapters; their full-prefill contribution is only ~1.4–1.6%, so attention/glue remain unbound | `runs/muse_prefill_layer/*dafbcad024fe4dcd.json`, `docs/reference/prefill_gemm.md`, `dev/dense_a30/`, `serving/sglang/python/sglang/srt/layers/muse_prefill_{dense,gemm}.py` |
| Dense/router 16K decomposition | **Measured a30** profile dense GEMMs total **739.043 ms**, compared with the older V11 **781.297 ms**; selected cold-stage timings total **741.508 ms** versus a **607.427 ms** tensor floor. Cold-stage excess: pooler **46.018 ms**, unpooler **43.464**, QKV **25.874**, O **11.410**, hi+lo router **7.314**. These are separate profiler/tuner observations, not an additive full-layer prediction | Pooler/unpooler own 67% of the selected-stage excess; prioritize their Lt layouts/schedules, then QKV. FP32 hi+lo router arithmetic is 4.5% of dense time and 5.5% of dense excess. Norm/residual boundaries are separately accounted: **159.077 ms** total (two calls combined per layer), versus **124.822 ms** byte floor; expert dequant is also separate (**117.848 ms**, two passes combined). The reported dense 781 ms did not include this glue. No memory-counter claim is made | `dev/dense_a30/decompose.py`, `decomposition.json`; immutable a30 receipt/profile and `docs/reference/prefill_glue.md` |
| Dense/router candidate (local admission) | `cand_dense_router_a30.py` starts from scored a30, retains its H200 expert controls (753/3012 masked Humming, 16K dg_w8), and appends the source-only serving GEMM plan for 16K. Search prioritizes pooler/unpooler and mixed pooler/router; Lt layouts, persistent/Stream-K, and QKV layouts are checked against the already-tuned a30 incumbent. A separate 20-pair real-consumer gate admits the whole mixed/separate group | Final source passes **5/5 tests, 8/8 graphs, F821–F823 and sm90a expert-source compile**, prepare **4.56s** on SM120. **168** forced fresh dense/route graph replays pass, the actual Lt wrapper compiles SM90a and executes its ABI locally, and an injected consumer-graph failure rejects. Expert dequant stays per-call; no expert BF16 bank is created in prepare. This queued candidate was superseded by mid_dense and withdrawn by the owner; serving remains unbound | SHA256 `f4968408357c569d257142f3a6be34d5ad8d2cae247e1a35c8bd78ad40842670`; `runs/screen/muse_prefill_layer.cand_dense_router_a30.final.json`, `dev/dense_a30/admission.json`; sweep `b964cb5d180a463b` is now measured above |
| Mid-experts2 + dense merge (local admission) | `cand_mid_dense.py` is based on exact **0d738799802340fd / 1.438402x**, retaining all34 expert choices and all attention paths, with the gated 16K dense/router search from `cand_dense_router_a30`. The new base's per-load speedups are **1.2412/1.4829/1.6169x**; named first-turn dense GEMMs total **731.502 ms**. Starting expert controls match that receipt: masked753/3012, dg_hint_w8 at16K | **5/5 tests,8/8 graphs, all3 static checks** (undefined names, call signatures, sm90a expert CUDA source); local prepare **6.310s**. Actual SM90a Lt wrapper/ABI and **168** forced fresh dense/routing graph replays pass. Optional exact-route build failures reject. Scheduling selections remain prepare-only and correctness/speed gated, including the20-pair real routing consumer. Sweep `b964cb5d180a463b` has landed; its new defaults and paired expert selection are folded into mid_sweep below. H200 dense gain/prepare and the direct16K CUTLASS lab remain unverified | SHA256 `1a341fc725dece4677ecbacbc399643fcc4d03f426e590387498765ed88d11ad`; `dev/mid_dense/{README.md,admission.json,base_profile.json}`; `runs/screen/muse_prefill_layer.cand_mid_dense.json` |
| Prefill sweep dispatch (local admission) | `cand_mid_sweep.py` on frozen mid_dense defaults to **raw17 at753 / masked Humming at3012 / dg_w8 at16K**. It gates/compares the sweep's top two per T, retaining the prior masked753 and hinted16K winners too:3/2/3 contenders. Three fresh full route maps (IDs/probabilities/counts/positions/pooled values),64 spread FP32 oracle rows,20 cold alternating graph repeats, then fresh post-timing gates select entirely in prepare; metadata prints in `EXPERT_SWEEP_LAST` near the receipt tail | **5/5 tests,8/8 graphs,all4 static checks** (F821–F823, call signatures, GROUPED_SRC and PG_RAW compiled SM90a), local prepare **7.458s**. Forced Hopper prepare runs all eight shortlisted paths using real SM120 math; six fresh full-layer graphs at753/3012/16384 pass with the resulting local selections. Launch/fixture exceptions with empty messages and budget rejection pass. Only expert dispatch/prepare/error logging changed; dense/attention and both CUDA source strings are identical to mid_dense. No H200 ranking is inferred from the local selections | SHA256 `1de6c32c7a5337dc539a5b6feb24a479bed217cb794a92a32760f9fcda82e075`; `dev/sweep_dispatch/{README.md,admission.json,receipt_comparison.json,scope_check.json}`. Old0d738 chose masked/masked/hinted: expert SCORE_SELECT was truncated, final tables retained the exact choices. Both runs use generated router fixtures;753 is confounded by changed RNG/possible run router plus warm eager vs cold graph protocol.16K old leaders were within0.16%, compatible with paired-timing noise. CUTLASS queue key63aa77388930a37e had no receipt at the single check; no dense recipe merged. New full-layer H200 and serving results remain unverified |
| A30 sweep-backed dispatch candidate | `cand_prefill_a30_h200.py` starts from scored `a30fb411b0c74ffb` and installs the H200 lab winners directly: raw17 at753 (**577.760us**), masked Humming at3012 (**1254.912us**), and dg_w8 at16K (**5167.232us**). It also installs the a30 H200 dense Lt choices, materializes their heuristic tables, and keeps expert NVFP4 expansion per-call | `tools/img python tools/local_eval.py ... --json` passes **5/5 tests, 8/8 graph shapes, F821–F823, call signatures, and every SM90a CUDA-string compile**; sm120 prepare is 5.1s. A `WeightScale2Type`/`WeightScaleType` compatibility alias fixes the current image import, and five closure-invalidating tuner `del`s were removed. No dequantized expert bank is retained in prepare | Candidate SHA256 `df5073b032d3eeddf10993dcb49a9d38d849146578a9cce84c97c5bc511c62b3`; H200 sweep receipt `b964cb5d180a463b`, `dev/dense_a30/decomposition.json`, and the local screen output |
| a30_overlap (scored) | `af7065a73e5f4abf` 1.428x (CI 1.407-1.450) vs board a30_h200 1.451x: turn 1 1.256/1.177, turn 4 1.530/1.329, 16K 1.650/1.690/1.490/1.403 | No gain over a30_h200; prefill is paused until the close (PLAN P2) | `problems/muse_prefill_layer/profiles/cand_prefill_a30_overlap.af7065a73e5f4abf.txt` |
| cand_v11_gemm versus 0d738799 | The newer `471645dc` per-shape speedups are **1.231/1.143/1.435/1.236** at the four 753-token loads and **1.683/1.666/1.400/1.327** at first-turn 16K ctx0/32K sliding/32K full/64K full; board `0d738799` is **1.285/1.188/1.522/1.349** and **1.704/1.654/1.531/1.431** on the same ordering | V11 wins only the 32K-sliding 16K case; it loses the other seven, so its dense/expert choices are not reused over the 0d738/a30 dispatch | `problems/muse_prefill_layer/profiles/cand_v11_gemm.471645dc489f4b26.json`, `cand_mid_experts2.0d738799802340fd.json` |
| Prefill CUTLASS dense search (closed) | H200 lab `d67168de5b1a4e0e` tested 2,208 CUTLASS recipe/shape cases at M753/3012/16384 for pooler, unpooler, QKV, O and router. **No recipe beat the incumbent**: selected controls were already about 760–780 TFLOPS at M16384 (77–79% of the 989 TFLOPS reference); best candidate ratios remained 1.01–1.36x slower. Pooler/unpooler remain the largest dense excess, but this route has no measured fix | Close the CUTLASS dense path for prefill. Retain incumbent Lt/cuBLAS/DG dispatch and focus the lane on expert execution; no dense/router or serving change is justified by this lab. The router is also near its practical ceiling, so do not spend another prefill run on speculative dense fusion without a new mechanism | Receipt `runs/muse_bf16_proj/20261024T081916Z_check_d67168de5b1a4e0e.json`, payload `prof/lab/prefill_dense_cutlass.py`, `dev/cutlass_dense/{README.md,admission.json}` |
| Serving prefill stage triage | At the a30 first-turn profiles, dense/router GEMMs are **3981us vs ~4243us stock** (ctx0), **3971 vs ~4230us** (ctx32 sliding), **3983 vs ~4358us** (ctx32 full), and **3974 vs ~4356us** (ctx64 full): 260–382us saved, only **1.4–1.6% of whole prefill**. FA3-contiguous saves **31/22/194/363us** at those same loads, **0.2–1.4%**. Glue remains roughly 1.1–2.1ms and its boundary/dequant/SwiGLU/combine/gather changes are below 3% of the layer | Experts are the only remaining stage above the handoff threshold. `muse_prefill_dense.py` and `muse_prefill_gemm.py` are validated serving entry points for stock-compatible dense/router GEMMs, but no attention/glue module is justified by these receipts; the serving lane should wire dense only if its integration cost is acceptable | a30 profile `problems/muse_prefill_layer/profiles/cand_score_portfolio_routing.a30fb411b0c74ffb.json`, baseline profile `runs/calibration/profiles/muse_prefill_layer_sglang.499cbe8c9d784741.txt`, decomposition `dev/dense_a30/decomposition.json` |
| Dense fusion serving adapter | In `models/muse_spark.py`, `MuseSpark1xMoE._forward_experts` computes `pre_expert_proj` (pooler) and `gate.forward_split2`; combine these with the lossless mixed-output bank, preserving existing topk/expert dispatch and pre-expert norm. Serving already overlaps router/pooler on a side stream; the board's parallel path adds no new serving mechanism | `MuseSpark1xAttention.forward`/O projection and `MuseSpark1xDecoderLayer` fused boundaries are the projection+norm hooks. Keep attention TP all-reduce before nonlinear norm. `_reduce_and_unpool` owns expert reduction and `post_expert_proj`; expert/unpooler chunk overlap requires the same ordered TP/EP collectives per chunk, not the board's identity reductions. Keep aux inputs alive through join and give concurrent Lt calls separate workspaces. Serving agreement/reentrancy remains unverified | `serving/sglang/python/sglang/srt/models/muse_spark.py`; board `_dense_front`, `_dense_boundary`, `_dense_back` |
| Local compiler/tuner failures | Large16K split4 constexpr capacity triggered a sm120 ptxas SIGSEGV; runtime capacity compiled and passed the identical case. A later local tuner OOM requested8GiB for uint8 torch.sum's int64 cast while split-K graph pools were resident | Runtime64-bit pointer arithmetic avoids giant immediates and reduces specializations. A persistent read-only1GiB L2 flush with small per-SM reduction scratch replaces only the synthetic tuner's reduction, outside events. Full admission rerun passes; judge timing is untouched | `problems/muse_prefill_layer/dev/fused_ptxas_failure.log`, `fused_expert_paths.py`, `round6_validation.json` |
| Fused/overlap serving adapter | Current live hook is `srt/layers/muse_spark_moe.py` (`prepare_layer`, `_grouped_call`, `run_layer`), called by `MuseSpark1xFp4Experts._forward_marlin`; the old `_v4.py` file is gone. New kernels replace the grouped expert producer inside `_grouped_call` | SGLang applies pre-expert norm before the hook: replace the board gather+norm by gather-only. Export sorted `y` before `_combine_kernel`, scatter to dense `[T,8,4096]`, preserve separate gate/up/down GS and BF16 rounding; raw-DG down needs GS2 applied before export. Keep `_finalize_dense_expert_outputs` and EP collectives, avoiding duplicate norm/top-k weighting. Prepare selected lossless raw/interleaved banks from loaded ModelOpt tensors, budget duplicates against measured KV residency. Per-call stream forks must join before down/finalize; concurrent serving graph replay/reentrancy still needs validation | `serving/sglang/python/sglang/srt/layers/muse_spark_moe.py`, `muse_spark_fp4_moe.py`; new candidate helpers |
| Humming configuration failure | A forced TMA configuration differed25% from raw WGMMA while ordinary Humming agreed within.23% on the same753-token fixture | `get_heuristics_config` returned a cached mutable dict. Updating TMA flags polluted later standard/stand-in configs and changed their compiled kernel IDs. Copy the dict before setting flags; current tuners use the independent semantic oracle and reject/log each bad variant. This is a multi-variant construction bug, not evidence that H200 TMA loses | `problems/muse_prefill_layer/dev/diag_score_humming_forced.py`, `diag_score_humming_forced.log`, corrected builder and forced full-portfolio validation |
| Direct canonical dequant (local only) | Read-once/scattered output 5.325 ms; coalesced K1024/K2048 3.253/3.201 ms; raw-v11 3.113 ms on sm120, both W13/W2, cold-L2 graphs | Sparse scalar stores were the concrete slow recipe; output coalescing recovers 1.66x locally. These local measurements favor raw dequant; bank compatibility does not justify a slower board choice. The new score portfolio measures both on H200. Both tile sizes preserve exact represented BF16 values over signed finite E4M3/E2M1 probes and fresh count graphs | `problems/muse_prefill_layer/dev/coalesced_dequant_benchmark_local.json`, `coalesced_dequant_validation.json`, `coalesced_dequant_2048_validation.json` |
| Research workspace | Original16K DG capacity135168rows: each4096-wide BF16 route buffer1.107GB; W13 BF16 expansion2.147GB, W2 1.074GB | Canonical DG sequential aliases model 5.60GB versus v11's8.89GB scratch, but this is capacity evidence only. Both allocation schedules compete on speed in the score portfolio; no memory-saving preference. Every live range is overwritten per call; full checkpoint/KV/allocator peak remains unmeasured | `problems/muse_prefill_layer/cand_v11.py`, score portfolio source |
| Pending expert mechanisms | v14 Hopper TMA indexed Humming; v15 shared/shared WGMMA; v17 bit-plane decoder | v15 gate/down emulation passes; v15/v17 SM90a builds have no spills. v17 exhaustive signed finite E4M3/E2M1 GEMM probe has0.159% local L2. Forcing Hopper TMA flags on unsupported sm120 failed; the actual guarded v14 passes locally. No H200 conclusion until receipts land | `problems/muse_prefill_layer/dev/verify_fast_decode.log`, `compile_hopper_ss.log`, `compile_hopper_fast.log`, immutable candidate sources |
| Other landed recipes | v6 Humming indexed1.328x; v8 masked-small/DG-large1.292x (turn1 1.240x); v9 contiguous/unpackedFA31.185x; v10 route-indexed SwiGLU1.190x; corrected native-dense v13 1.237x (first-turn1.574x) | Whole-layer performance owns the choice. Small FA3-stage improvements or native-dense labels do not establish a complete gain | Jobs `1961a539127a4184`, `a2702eaedce34844`, `f13e757f4c224e06`, `583a1b97cc544f53`, `362f63065c634281` under `runs/muse_prefill_layer/` |

The frozen source order and local evidence live in `problems/muse_prefill_layer/dev/NEXT_CHECKS.md`;
`problems/muse_prefill_layer/FOCUS.md` is the lane entry point. Earlier queued sources stay unchanged.
The obsolete promotion handler was stopped and deleted: `send` chooses a scored run while a slot
remains, otherwise a check, and the board retains our best. The orchestrator sends the ready queue.

Current TP8/EP8 residency and KV-capacity accounting, including stock comparison,
active projection layouts, graph/scratch measurements and the bounded-SWA fix, is in
[Serving track: KV residency and long-context capacity](#kv-residency-and-long-context-capacity-2026-10-24).
The earlier77.8GB weight subtotal omitted selected transposes and router scratch;
it was not a full non-KV residency measurement.

### Projection evidence and next experiment (G3, G2)

Current priority is **put projection winners inside decode and serving**. The fitted
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
| Owned NVFP4 weight-bank audit | Owned dense serving modules contain BF16 projections only. `cand_v11_gemm` retains packed expert bytes/E4M3 scales in prepare, expands in each timed expert call. Raw17 grouped path permutes NVFP4/E4M3 bytes; FP32 diagnostic reference is not returned in runtime state | No prepared BF16/FP8 expanded NVFP4 weight bank found. Inherited V9 prefix in the withdrawn/parked grouped candidate recodes scales to uint8 exponent/mantissa and folds sign/zero into FP4; exact but not a permutation of stored E4M3 bytes. Do not reuse that encoder under the stricter byte-layout instruction; the MoE lane owns its incumbent | `problems/muse_moe_w4a16/dev/tma_grouped/README.md`; new CUTLASS block handles BF16 dense only |
| Stock CUTLASS verify recipe set closed | H200 lab45f58e7009c440eb:144 recipes/key, every choice loses.20-pair best M256 control/new: qkv23.792/31.024us (1.306x), pool/router33.648/41.856 (1.254x), unpool28.528/30.752 (1.082x), o13.312/15.664 (1.183x); M128 qkv20.416/27.200 (1.322x), pool28.352/35.072 (1.244x), unpool23.632/27.264 (1.142x), o11.728/13.920 (1.186x). The often-quoted1.07 unpooler ratio is coarse, not the20-pair finalist | Standard image CUTLASS Hopper cooperative TMA schedules, including split-K/Stream-K, multicast, swapped AB, rings/raster/grid/reduction choices do not beat cuBLAS/Lt at these exact M128/256 shapes. Close this **recipe set**, not the family. No native recipe is installed. Cold compile71.61s; lab80.64s; total prepare88.7s | `runs/muse_bf16_proj/20261024T063237Z_check_45f58e7009c440eb.json`; payloadSHAe43ed9b225d98cf1… |
| Verify weight-wide WGMMA (local admission; H200 pending) | New TMA kernel uses tokens on M64, weight rows on N64/128/256, 1–4 consumer warpgroups sharing 2–8-stage rings and one-launch split-K reduction. **744** local checks/**90** fresh graphs/**415** native descriptors; full lab **407** passing recipes at12 keys, zero rejections; forced decode **13/13 tests,12/12 graphs,15/15 static**; prefill host **5/5 tests,8/8 graphs,3/3 static**. Native15 kernels have no spills, max154 registers | Exact best6 `1b80ffc0` cuBLAS/Lt controls at M8/128/256, real loaded BF16/router weights,3 fixtures,20 cold rotated finalist pairs. Pooled capture streams cause a replay failure; distinct native streams fix the full reproducer (cuBLAS workspace lifetime is inferred). M128/256 dense bank is197.13MB/50.47GFLOP at256 (41.07us read floor,63.08us at800TFLOPS);138.41MB applies to M8's unpool slice. No prepared NVFP4 expansion; no H200 winner or dispatch installed | `problems/muse_bf16_proj/dev/verify_streaming/README.md` and JSON admission; payloadSHA`7b67602f1d4b77c0…`; merged with proj glue on prefill, send key **be5d6aa5ae1ac0bd**, priority0.5; detached log reports waiting for a slot, no rejection |
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

| Serving verify adapter | `muse_verify_attention.py` consumes the fused QK-norm op's FP8 query and physical paged FP8 K/V cache, walks page tables with exact causal/window limits, and dispatches only fixed q_len=8 GQA shapes; FA3 remains the fallback. | Local SM120 graph replay matches an independent FP32 page-table reference (relative L2 <=3e-5 for full/window). The local image lacks an FA3 SM90 cubin, so H200 adapter-vs-FA3 timing and graph parity remain open. | `serving/tests/test_muse_verify_attention.py`, `flashattention_backend.py`; local only |

### Prefill attention (2026-10-24)

The H200 FA3 prefill profile in `cand_v11.6a9cd943fe404d74` is the current
attention state of the art. Counting QK and PV as `4 * Q * K * 16 * 64` gives
the following effective dense-BF16 utilization (989 TFLOPS peak):

| load | FA3 kernel us | effective TFLOPS | peak share |
|---|---:|---:|---:|
| turn1 753 sliding/full | 25.8 / 414.8 | 244.8 / 597.7 | 24.8% / 60.4% |
| turn4 753 sliding/full | 61.7 / 1618.0 | 409.5 / 621.5 | 41.4% / 62.8% |
| first 16K, ctx0 full | 900.3 | 610.7 | 61.7% |
| first 16K, ctx32K sliding/full | 309.3 / 4531.4 | 444.4 / 606.6 | 44.9% / 61.3% |
| first 16K, ctx64K full | 8431.4 | 586.8 | 59.3% |

The short sliding loads are launch/metadata limited. Long full attention sustains
about 587–622 TFLOPS, leaving 37–41% of dense peak. The FP8 gather recipe is
closed on H200: `cand_v11_attn` scored 1.356768x, but at 16K full attention
FA3 was 707.584/706.080 us over 32K/64K context versus gather256
1352.640/1342.880 us, gather128 1468.544/1465.152 us, and gather64
2646.944/2637.216 us. FA3 is therefore retained for prefill attention; the
standalone block remains development evidence and has no serving dispatch.
The current prefill winner is `99820fa6e5c14fe1` at 1.428625x, selecting
`fa3_contiguous` on most shapes (about 7.62 ms for 16K over 64K and 418 us
for 753 over 80K full). Evidence: `cand_v11_attn.e789ab194eb7496c.txt` and
`cand_fused_expert_portfolio_routing.99820fa6e5c14fe1.txt`.

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
