# S14 pipe, dependency and TMA-flight identification

Current nomination gate: every TASK-gated per-shape point estimate below1 goes to P4, with ratio,SE
and seeds. Confidence/A:A summaries below are descriptive only. P4 confirms the held-out six-load R.

Receipt **`16c180784f414576`** is complete:16 routed seeds,96 pipe probes,66 work rows,6 latency rows.
Official host and correct S14/phase gates pass; PDIA is deliberately wrong math. `h200_pipe.txt` is the
losslessly decoded matrix; `read_reset.py` verifies it and records decisions in `h200_results.json`.

Measured classification at M8/16/32:

- Shared ILP2/W8 node-time ratios .98991/.98145/.98272, work-normalized .98973/.98177/.98275,
  with the same2 CTAs/16 active warps. This small contrast supports a dependency hypothesis.
  Microprobes have no independent A:A; it does not prove a full-kernel gain or one dominant bottleneck.
- Shared W16/W8 work-normalized ratios1.01796/1.02019/1.01646:32 active warps do not help.
- Cold→hot math node ratios .87069/.94082/.96439, but depth4/depth2 copy ratios
  .98253/1.07285/1.14805 do not isolate a flight shortage; depth4 also reduces actual residency to1 CTA.
- Stable ILP4 LOP3/SHF/PRMT rates are about1.96 warp instructions/SM/cycle; BF16 HMUL2 reaches3.57,
  HMMA .658, LDS128 .249, LDS32 .996, four-way LDS32 .250. These are operational same-op limits.
  The dependent SHF/PRMT cap8 clocks1.336/1.447GHz are anomalous; their >2 issue rates are not used.
  No integer/PRMT saturation is established in the real mixed kernel.

The ILP contrast triggered the already-prepared four-arm paired lab, sender4124752, merged wrapper
`de47736b9f588673`, on MoE. `followup_queue.json` records dispatch. FLIGHT/K16/LUT travel in that
packed batch but are not identified fixes; no rebuilt kernel is promoted. PAIR's separate epilogue
receipts nominate it to P4; TREE also has small M8/M16 point wins and remains an alternative. Whole-kernel and whole-layer controls still decide.

That full-kernel follow-up is now measured: **`dd539517bc8344dd`**,32 seeds/M,all native/official gates pass.
FLIGHT/S14 is1.07662/1.05285/1.03119; K16 is1.20394/1.21153/1.23270; LUT is1.59959/1.62315/1.67379.
Close those exact recipes, leaving other issuing schedules, K/occupancy layouts and lossless decoder
methods open. ILP/S14 is .99055/.99041/.99046; A:A .487/.316/.314us. Its upper95%+A:A is
1.000762/.996981/.996081, but **M8/M16/M32** now nominate by point estimate. This confirms a small dependency lever
in the full kernel; it does not establish one dominant hardware pipe. The old M8 uncertainty veto is removed.
`pipe_s14_ilp.patch` is the exact admitted patch. Fresh-seed direct ILP/PAIR comparison is queued as
sender479906, wrapper`af7b4526022c8d75`; direct result1b135b70835f4634 favors PAIR over ILP at all3 M; whole-layer and combined gains remain unmeasured.

The earlier58.7 us math/activation clock is ONCE, not S14. Counts28.56/34.30/45.43 per unique group
are S14's repeated-block model. They must not be compared as the same schedule. The new lab measures
matched S14 math/B-fill clocks and ONCE-style gathered-activation contrasts.

## Per-pipe envelopes

A group is16 weights/lane across32 lanes,288 packed bytes including scale codes. At4TB/s,132SMs,
1.98GHz (assumed), one unique group gets18.8179 SM cycles. Public limits describe PTX operation classes;
PRMT and several SASS routing details are unpublished. The synthetic same-op probes measure effective
limits and dependent-chain rates on the judge. These are operational instruction groups, not a claim of
an exact physical port map. Do not add independent-pipe utilization percentages when resources overlap.

| Class | Per requested group | M32 per unique group | Nominal warp instructions/SM/clock | M32 load at4TB/s |
|---|---:|---:|---:|---:|
| Integer/address math+finite epi |17.5833|27.9666|2|74.3%|
| PRMT |1|1.5905|about1.96 measured operational|about4.3% at assumed1.98GHz|
| Packed BF16 HMUL2 |8|12.7241|4 F16x2 envelope; BF16 calibrated|16.9% envelope|
| HMMA16816 |2|3.1810|1 dense-TC upper bound; actual mma calibrated|16.9% upper-bound load|
| LDS128 weights+scale |.5625|.8947|.25 shared-bandwidth envelope|19.0%|
| Shared bank service, bytes (incl epi writes/reads) |448 B|712.55 B|128 B/SM/clock envelope|29.6%|
| FP32 finite epi |.47917|.76212|4|1.01%|
| MUFU finite epi |.02083|.03314|.5|.35%|
| F2FP BF16 finite epi |.01042|.01657|.25 conservative packed conversion envelope|.35%|
| SHFL norm epi |.04167|.06627|1|.35%|

`pipe_model.py`/`pipe_model.json` retain calculations. Repeated-block factors from the32 local routing
seeds are1/1.20098/1.59052 atM8/16/32. The table excludes outer scheduler/setup and producer/control
instructions. Activation B-cache fill adds16 LDG128/warp/segment (one/group divided by run length),
plus its address work. Norm atomics are address-contended, so aggregate instruction count does not
model their latency. Rare division helpers are excluded from the finite-path epi count.
Input activations are already BF16; their B-cache loads require no conversion. The W13 packed BF16
round is the F2FP row; the W2 BF16-to-FP32 path uses shifts. Packed F2FP is charged conservatively as
two scalar conversions against16 scalar results/SM/clock; its exact SASS port limit is unpublished.

Primary references: [NVIDIA operation throughput](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-c-best-practices-guide/index.html#arithmetic-instructions),
[NVIDIA shared banks](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-programming-guide/index.html#shared-memory-5).
The nominal tensor roof is4096 FLOP/SM/clock and HMMA m16n8k16 computes4096 FLOP. The actual same-op
limit can be lower: [Hopper microbenchmarks](https://arxiv.org/html/2501.12084v2) measure backward-compatible
mma below WGMMA's peak. H800 results are context, not our H200 calibration. Likewise, the F16x2 table
is an envelope; the lab explicitly measures BF16 HMUL2.

## Little's law

4TB/s/132 =30,303.03 B/SM/us. Required outstanding bytes are this rate times measured completion latency.
A36,864-byte tile, two CTAs and two raw slots give147,456 B (144KiB) allocated capacity. Excluding one
consumed/current slot per CTA leaves73,728 B (72KiB) as a steady lookahead upper envelope. Capacity is
not an observation of time-average outstanding bytes; slots may already be ready or not issued yet.

| Completion latency (illustrative) | Required bytes/SM, unique4TB/s | If all M32 repeated-block requests miss L2 |
|---|---:|---:|
|.25us|7,576|12,049|
|.5us|15,152|24,099|
|1us|30,303|48,197|
|1.5us|45,455|72,296|
|2us|60,606|96,395|

The72KiB envelope covers2.433us at unique4TB/s, or1.530us under the1.59052 all-cold reload assumption.
That factor describes logical TMA traffic; L2 hits can make HBM traffic smaller. The measured cold
36KiB completion is .676/1.096/2.000us at1/132/264 CTAs; hot is .320/.568/1.056us.
At264 CTAs the unique4TB/s requirement is60,606 bytes/SM, giving1.217 steady-capacity coverage.
An all-miss1.59052 reload model would reduce that to .765, but actual HBM misses and average outstanding
bytes are unmeasured. Copy depth does not confirm the shortage. Exposed wait and issue-to-late-consume
time are not completion latency.

## Diagnostic payload

`build_pipe_lab.py` -> `prof/lab/moe_nt1_pipe.py` contains:

-16 weight/router seeds atM8/16/32: exact S14 versus PDIA, A:A, readflush AB/BA, separate consumer nodes.
 PDIA retains TMA ranges, B-cache/weight LDS and syncs, removes decoder/MMA and leaves a small finite
 checksum tail and the scalar epi. Every B operand and loaded weight word contributes to the checksum.
 Its wrong math is explicitly ungated and cannot be promoted. Native flags69 has120regs and zero spills.
-8 paired S14 phase seeds per M: same routed graph, B-fill, W13/W2 math, store/barrier, expert-ready and
 grid clocks; trace/off observer tax. Every correct S14/phase arm passes the native TASK gate.
-4 seeds of matched expert byte ranges/block multiplicity: pure TMA stream at1/2/4 stages, weight LDS,
 full math from HBM or a4.7MiB L2 working set, shared-only math,4/8/16-warp CTAs, varied resident CTAs,
 two independent MMA chains, and NB2/NB4 activation gathers where the M shape permits them.
-LOP3/SHF/PRMT/HMUL2/FFMA/HMMA/LDS128/LDS32/4-way gathers/SHFL/MUFU and shared-address chains.
 ILP1/4 and occupancy targets8/16/32/64 warps. Actual grid/residency/regs/local bytes/active SMs and
 clock64/globaltimer are printed. Same-op throughput is distinct from its single-chain dependency cost.
-TMA36KiB latency with immediate wait, cold unique/hot small-pool addresses and1/SM/2SM CTAs. The final
 cold replay has a read flush too. Timed microkernels are diagnostics; shared-only rates count equivalent
 packed work, not HBM bytes. Pure copy has no expert-ready dependency.

`audit_pipe.py` checks full native instruction/resource fingerprints and that pure-copy kernels have no
HMMA/HMUL2 and math/gather probes retain their prescribed MMA count. It also checks PDIA retains the
activation/weight loads and has no spills. Extra-block results are all observable; this prevents dead
code elimination from turning a gather/MMA probe into a different workload.

The complete final payload passed16 local seeds, all96 same-op probes,66 work rows and6 latency rows.
Local D4 copy uses a test-only D2 stand-in because147KiB exceeds sm120's per-block shared capacity;
the original D4 compiles sm90a and fits H200. `pipe_lab_validation.json` records36 forced ABI calls.
`compact_pipe` losslessly packs the full probe matrix to stay below the9000-character receipt cap;
`read_pipe.py` decodes the local log or H200 receipt. The final local receipt uses5832 characters.
Local timings cannot select the H200 bottleneck. The receipt and classification above select below.

Completed: MoE sender1805785, wrapper875038058dfb252d, payload `d5319603`,
judge **`16c180784f414576`**. Both epilogue receipts are also complete (README).

## Rebuild decision rule and sent response

All derive from S14, preserve original M1, and cover2<=M<=32 when the small-M path is called.
`patch_s14.py --kind flight|ilp|k16|lut -o <candidate.py>` emits cell+patch+manifest; no variant is promoted.

| Evidence after reset | Ready response | Reason / rejection condition |
|---|---|---|
| Cold math loses to hot math; copy depth/flight scales materially and Little-law coverage is short | FLIGHT (`pfly_cell.py`) | One per-CTA bulk-L2 prefetch beyond the two staged slots; reject if depth doesn't help or it competes with useful loads |
| Shared-only math improves with independent chains, without a saturated pipe | ILP (`pilp_cell.py`) | Alternate two MMA accumulator chains; reject if new registers or final sum erase the benefit |
| Shared-only math improves at32 versus16 active warps, after normalizing work and residency | K16/r64 (`pk16_cell.py`) | K256/warp,16 math warps/CTA,64 regs, last eight B operands cached in timed shared; two H200 CTAs fit |
| Integer/PRMT throughput is close to the measured same-op envelope while shared has headroom | FP4-only LUT (`plut_cell.py`) |1KiB code-pair table, scale math unchanged; moves integer decode work to shared. Reject if predicted or measured bank service becomes the tighter roof |
| Shared gather/epi serialization dominates | Existing PAIR/TREE cells | Completed tests nominate PAIR; TREE has small point wins versus PAIR at M8/M16; no rerun is needed for nomination |

If evidence conflicts or no diagnostic isolates a limit, none of these is sent as a purported solution.
A classified branch must still beat S14 in a paired kernel lab and then the six-load whole-layer score.
The paired follow-up `prof/lab/moe_nt1_pipe_rebuild.py` (`caa59593`) is now sent: all four arms
in one run,32 seeds/M, ABBA read flush, A:A and consumer node rates. It passed every local gate and
all32 seeds; K16's local stand-in recorded192 ABI calls. The small independent-chain contrast is the
classified ILP lead; full-kernel measurement must establish whether that lead survives. Other arms are
packed comparisons, not claimed solutions. `pipe_rebuild_validation.json` pins the unchanged payload;
`followup_queue.json` records completed job`dd539517bc8344dd`; the measured outcome is above.

The K16 shared cache makes the two-stage kernel110,736 dynamic bytes atM32: it fits two228KiB H200
CTAs, but exceeds local sm120's per-block capacity. The explicit test-only `k16_standin.py` runs native
math with one raw stage and checks the unchanged two-stage ABI; it is absent from candidates. Flags69
has64 regs,16-byte stack and zero spill loads/stores; trace325 has observer spills. Local timing cannot
rank its intended32-warps/SM H200 form. A discovered512-thread finalizer overflow was fixed by retaining
256 combine owners; all component/replay/offset checks were rerun after the fix.

The LUT keeps two lossless code layouts, allowed by CONSTRAINTS. `lut_audit.py` independently compares
all decoded integer code pairs and all scale bytes across the full banks, then enforces9bytes/16 per
layout. It replaces the old aggregate cap that incorrectly rejected legal duplicates. No wider weight
values are retained in prepare. This differs from the closed joint weight/scale lookup recipe: only an
FP4 code pair indexes a fixed1KiB table; scaling stays in the timed HMUL2 path.

## c49 composition and next cuts

`c49dbdffe92d491a` confirms the SHARD2/PAIR/PAIR+BLOCK per-M composition is TASK-correct on16 fresh
seeds/M. COMBO/S15=.997261/.988757/.970993/.979561;direct COMBO/PAIR at M16/M32=.995504/.993125.
A:A mean absolute .225/.480/.309/.362us. Every point winner goes to P4 with SE,seeds and exact
`review/combo/combo.patch`6b3990b2;only P4's held-out six-load R promotes the whole layer.

The new cuts are on that measured parent,not a synthetic math clock. `review/next_issue/sass_cost.json`
compiles actual CUDA and marked fill bodies with sm90a,-O3,-lineinfo:

- CANCEL/S15 has the same125regs/zero spills,but adds16 shared expert-map loads,23 static global loads,
  sign/range predicates,and FP64 exceptional tails. Main-kernel text50688→65152B;ordinary finite epsilon
  takes the FP32 path. Static FP64 text does not prove ordinary FP64 execution or a unique bottleneck.
- FAST represents the same norm identity by bounded FP32 metadata and one fma/rsqrt/multiply with a
  zero-norm guard. Sign is in its numerator. No runtime FP64/sign/exponent branch;224 native extreme
  helper cases and16 captured full-MoE scale cases pass. New PAIR+BLOCK FAST uses124regs,zero spills.
- INDEX hoists row-valid/pass branches outside8 fills and uses one bounded32bit vector index per
  segment. Marked integer/address/control counts W13 31→13,W2 26→8,while each retains16 vector loads.
  Save18/(16*L)=1.125/L per16 requested weights at actual segment length L. In the narrower original
  SHF/LOP3/LEA/IMAD class alone,the reductions are15(W13) and17(W2) per segment. The core per-tile
  decoder remains17 instructions in that class; this amortized setup cut does not close that gap.
- FLUSH1 deletes the second consecutive norm flush after the first has resetnv0=0. Every necessary
  flush/store/fence/publish/acquire remains. Count-build PAIR+BLOCK text3352→3296 instructions,
  ATOMG7→6,125regs/zero spills. Its measured latency benefit is still open.

These are three independent arms,not a combined gain claim. All9 variants pass810 native TASK/replay
checks. Normal/forced screens and the packed paired follow-up are under `review/next_issue/`.
The restricted same-mask3-pure-rotations-per-word+one-LOP3 search has no layout with zero rotation in
both source words;that rejects only this cheap one-SHF-saving repack,not other lossless decoders.
No new throughput/saturation claim is inferred from the static instruction cuts.

Instruction-cut follow-up accepted as MoE check`d7fb2d8bddae4b8c`,sender1971066,wrapper`76a7f192a080d0b6`. All6 full screens and the exact16seed/64AA local payload pass. H200 timings are pending;no new cut is promoted from static counts alone.
