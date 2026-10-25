# S14 pipe, dependency and TMA-flight identification

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
| PRMT |1|1.5905|unpublished; calibrated|pending|
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
That factor describes logical TMA traffic; L2 hits can make HBM traffic smaller. There is no measured
H200 latency yet. The queued lab substitutes its cold/hot, lightly/fully loaded36KiB completion
latencies, reports required bytes and coverage, and measures throughput against depth1/2/4 and actual
CTA residency. It does not label exposed wait or issue-to-late-consume time as completion latency.

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
Local timings cannot select the H200 bottleneck. The H200 result, not this static table, selects below.

Queued: MoE sender1805785, wrapper875038058dfb252d, priority1.2, not-before1792972800. Payload SHA256
`d5319603`; no judge ID yet. The older epilogue senders3360739/1005738 remain unchanged.

## Ready variants and decision rule (unsent)

All derive from S14, preserve original M1, and cover2<=M<=32 when the small-M path is called.
`patch_s14.py --kind flight|ilp|k16|lut -o <candidate.py>` emits cell+patch+manifest; no variant is promoted.

| Evidence after reset | Ready response | Reason / rejection condition |
|---|---|---|
| Cold math loses to hot math; copy depth/flight scales materially and Little-law coverage is short | FLIGHT (`pfly_cell.py`) | One per-CTA bulk-L2 prefetch beyond the two staged slots; reject if depth doesn't help or it competes with useful loads |
| Shared-only math improves with independent chains, without a saturated pipe | ILP (`pilp_cell.py`) | Alternate two MMA accumulator chains; reject if new registers or final sum erase the benefit |
| Shared-only math improves at32 versus16 active warps, after normalizing work and residency | K16/r64 (`pk16_cell.py`) | K256/warp,16 math warps/CTA,64 regs, last eight B operands cached in timed shared; two H200 CTAs fit |
| Integer/PRMT throughput is close to the measured same-op envelope while shared has headroom | FP4-only LUT (`plut_cell.py`) |1KiB code-pair table, scale math unchanged; moves integer decode work to shared. Reject if predicted or measured bank service becomes the tighter roof |
| Shared gather/epi serialization dominates | Existing PAIR/TREE cells | Their already-queued tests reduce bank requests and add-chain depth; do not rerun them as a new recipe |

If evidence conflicts or no diagnostic isolates a limit, none of these is sent as a purported solution.
A classified branch must still beat S14 in a paired kernel lab and then the six-load whole-layer score.
The paired follow-up is prebuilt as `prof/lab/moe_nt1_pipe_rebuild.py` (`caa59593`), unsent: all four arms
in one run,32 seeds/M, ABBA read flush, A:A and consumer node rates. It passed every local gate and
all32 seeds; K16's local stand-in recorded192 ABI calls. Send this batch only when the diagnostic
isolates a limit addressed by at least one arm. Otherwise hold it, rather than spending a run on an
unidentified mechanism. `pipe_rebuild_validation.json` pins the exact payload and local receipt.

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
