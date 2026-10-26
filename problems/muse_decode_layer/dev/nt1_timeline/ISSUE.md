# S14 small-M issue model and second-round cuts

H200 decisions: `d650c04ead114119` and `29ad388de976432c` pass all native gates.
PAIR/S14 is .97493/.97262/.97932 (16 seeds) and .97830/.97462/.97778 (32 seeds) at M8/16/32,
above A:A uncertainty. TREE/PAIR .99904/.99811/1.00007 does not establish a benefit from the shorter
add chain. FOUR, LOCAL and FOUR_LOCAL lose versus PAIR: fewer modeled bank requests or zero norm
atomics are insufficient to predict a win. LOCAL's traced M32 export/gather is .234/1.110 us and its
combine is2.112 us versus PAIR1.498; this supports the tail-cost explanation, with observer tax and
overlapping clocks preventing additive attribution. PAIR is nominated to P4, pending whole-layer pairing.
Exact ratios, controls and remaining hypotheses are in README.md and `h200_results.json`.

The 4 TB/s target counts unique packed expert bytes, including scale codes (9 bytes/16 weights).
It is a logical rate, not a measured HBM counter. Exact base: S14 `f1c77ce6`, flags69 at M8/16/32.

Reproduce in the judge image with GPU0:

```sh
CUDA_VISIBLE_DEVICES=0 tools/img python3 prof/compile_static.py problems/muse_decode_layer/cand_r3_best7_s14_final.py --out runs/screen/nt1_round2_exact
CUDA_VISIBLE_DEVICES=0 tools/img python3 problems/muse_decode_layer/dev/nt1_timeline/issue_audit.py
python3 problems/muse_decode_layer/dev/nt1_timeline/issue_budget.py
python3 problems/muse_decode_layer/dev/nt1_timeline/reduction_depth.py
```

`compile_static.py` extracts the exact embedded strings without importing the candidate. The marked SASS
probes copy its reduction bodies and the reuse builder's identical decoder/MMA bodies. They compile sm90a
locally. Counts and assumptions are in `issue_counts.json`, `issue_budget.json`, and `reduction_depth.json`.

## Issue budget

Hopper has four schedulers and64 INT32 cores per SM; each scheduler can issue one eligible warp instruction.
This describes aggregate peak issue, not the throughput of every opcode or the number of eligible warps.
[NVIDIA architecture and scheduling](https://docs.nvidia.com/cuda/archive/12.9.1/cuda-c-programming-guide/index.html#compute-capability-9-0).

A group is16 weights/lane across32 lanes, hence288 packed bytes. At4 TB/s, the132-SM issue budget is
`4*132*clock/(4e12/288)` instructions/group:75.272 at1.98 GHz,69.569 at1.83,53.603 at1.41.
Those clocks are assumptions; the reset trace measures effective clock64/globaltimer.

| Blocks sharing one decode | Instructions/warp tile | Per16 weights/lane | Decoder/scale group | MMA | Extra activation global loads |
|---|---:|---:|---:|---:|---:|
| 1 (S14) | 457 | 28.5625 | 26 | 2 | 0 |
| 2 | 532 | 33.2500 | 26 | 4 | 1 |
| 3 | 618 | 38.6250 | 26 | 6 | 2 |
| 4 | 693 | 43.3125 | 26 | 8 | 3 |

One warp tile is256 weights/lane; divide instructions by16. The decoder/address/scale group has416
instructions including128 HMUL2; all paths have32*blocks HMMA. The first B0 cache fill costs46 warp
instructions (16 LDG128) per expert/block segment, amortized over its tiles. BF16 MMA operands need no
conversion. W13 output uses BF16 conversion; W2 postnorm BF16-to-float uses shifts. A scale LDS128 before
the math marker adds0.0625/group. NB3/NB4 isolated probes have2/7 spill instructions: the nonspilling full
ONCE image differs, so their totals are a model, not an exact dynamic count of that image.

The H200 f58cc7a6 route mix has81/17/17/8 experts with1/2/3/4 blocks,123 experts and198 blocks/16seeds.
It normalizes S14 math to45.979/group of unique weights and ONCE math to31.560. S14's scalar epilogue
adds2.842; a partial store/barrier adds0.4375 per requested group; the producer's no-retry branch envelope
adds0.734–1.656 per requested group, plus cache fill/setup. At a high clock aggregate issue leaves room,
but that alone does not rule out an individual pipeline limit. The S14 math integer/address envelope is
29.076/group of unique weights versus a hypothetical two-warp/cycle budget37.636 at1.98 GHz or26.801 at
1.41. This includes LOP3/SHF/LEA/PRMT/IMAD, not a measured routing of those opcodes onto one pipe, and omits
outer integer work. We cannot call it purely memory bound from the static model.

## Reduction latency and work

At M32 old ONCE's H200 critical CTA has58.7 us math/activation,28.5 us epi,3.6 us exposed weight waits
(f58cc7a6). Epi includes stores, barrier, producer, activation/output and norm atomics. Its time is not an
atomic-only measurement. The first batch (`d650c04ead114119`) separates these clocks and tests bank-safe loads,
warp7 production, and run-based norms.

Scalar gathers map four distinct words to a shared bank. The request model follows
[NVIDIA's shared-bank rules](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-and-memory-banks).
The scalar reduction models128 read wavefronts/CTA versus32 for PAIR; partial writes remain32.
These are address estimates, not measured conflict counters. FOUR keeps32 reads by issuing LDS64 from
four half-active warps instead of two full warps. It spends more warp instructions to expose more writers.

| Epi per CTA output tile | W13 warp instructions | W2 warp instructions | Per16 weights/lane W13/W2 |
|---|---:|---:|---:|
| Scalar (admitted) |185|308|1.445 /2.406|
| PAIR (measured winner) |148|129|1.156 /1.008|
| TREE |144|125|1.125 /0.977|
| FOUR |314|270|2.453 /2.109|

Marked static sets include guarded branches, addressing and output work; W2 discounts one synthetic
observation store per writer. PAIR/TREE flush eight scalar norm atomics per full token run; FOUR flushes
sixteen, since each writer owns half a token's channels. TREE's measured SASS FADD chain is8->3, with no
additional spills. It tests dependency latency despite a small instruction-count reduction.

LOCAL uses two shared FP32 slots per possible pair (2 KiB atM32), with disjoint parity/channel-half ownership.
Every replay zeros them inside the timed kernel. Run flushes update these CTA-local slots; all threads join
before exporting the sums to `ss[npairs+pos*grid+cta]`. There are no global norm atomics. Every exported
position is written even when no expert routed locally, so replay does not reuse old values. Before the
grid release, each exporting thread fences its writes. After release, finalizer warp j gathers the CTA
partials for top-k slot j coalescently across lanes, reduces them and shares one scale. It then runs the
original weighted combine. This trades per-run global contention for270336 B of export atM32/grid264,
plus540672 B of finalizer reads and one extra CTA barrier. FOUR_LOCAL gives each half a separate slot;
no cross-warp shared atomic or unsynchronized += occurs. Scratch contains only current FP32 sums,
allocated in the call; prepare retains exactly the original packed weights.

The new trace separates local export (slot61) and the final gather (62), in addition to first-batch
store/barrier, producer, writer0 and norm flush clocks. Producer/body overlap makes the subclocks
non-additive. Full flags69 sm90a images retain128 HMUL2/32 HMMA each and zero spills; measured H200
residency is2 CTAs/SM for every arm. Aggregate issue counts cannot predict barrier skew or lost
parallelism; use the paired whole-kernel and then whole-layer result to select the recipe.
