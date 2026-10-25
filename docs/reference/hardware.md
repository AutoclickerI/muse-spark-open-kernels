Read when sizing Hopper kernels or inspecting memory, tensor, register and synchronization limits.
Serves: G2.

# H200 and Hopper hardware

Judge-specific measurements and protocol: [FINDINGS](../FINDINGS.md#the-judges-h200).
Local sm_120/sm_121 timings do not establish Hopper execution or rank.

## H200 facts

| Parameter | Value | Source |
|---|---:|---|
| HBM3e | 141 GB, 4.8 TB/s | [NVIDIA H200](https://www.nvidia.com/en-us/data-center/h200/) |
| L2 | 60 MB, driver-reported | [DeepGEMM issue 107](https://github.com/deepseek-ai/DeepGEMM/issues/107) |
| Shared memory | 228 KB per SM, 227 KB per block | [Hopper tuning guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html) |
| Registers | 64K 32-bit per SM, up to 255 per thread | same |
| Dense tensor rates | BF16 989, FP8 1,979 TFLOP/s (half the sparse figures NVIDIA lists) | [NVIDIA H200](https://www.nvidia.com/en-us/data-center/h200/) |
| Scalar issue, cc 9.0, per SM per clock | 32-bit integer add, bitwise AND/OR/XOR, shift, compare: 64; FP32 FMA: 128 | [CUDA guide throughput table](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-c-programming-guide/index.html) (checked 2026-10-25) |

H200 is sm_90; WGMMA and TMA need `sm_90a` builds, which the judge runs. Architecture-specific code doesn't port
across architectures ([Hopper compatibility guide](https://docs.nvidia.com/cuda/hopper-compatibility-guide/)).

## Memory requests and cache policy

HBM rate is a request-concurrency problem: aligned, coalesced rows, several independent loads per thread and
enough resident CTAs. A 128-bit load sets request width, not memory-level parallelism (measurements: FINDINGS "The judge's H200"). TMA pays off when a block reuses a tile; for a
one-pass skinny stream its barrier and descriptor cost can lose to plain loads (H800 study: vector loads reach
90-91% of bandwidth, TMA helps mainly large matrices; [arXiv:2501.12084](https://arxiv.org/abs/2501.12084),
[arXiv:2402.13499](https://arxiv.org/abs/2402.13499)).

PTX cache controls (`L1::no_allocate`, `L2::evict_first/last`, prefetch-size hints, `createpolicy`,
`prefetch.global.L2`) are hints the hardware may ignore ([PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)).
Stream one-pass weights so they don't evict reusable inputs; keep L2 reuse for tiles several tokens read. L2
persistence across the per-replay flush is forbidden (CONSTRAINTS "Boundary cases"); the access-policy window is
probabilistic anyway ([L2 cache control](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/l2-cache-control.html)).
H800's near/far L2 partitions are a locality model, not an H200 bank map.

## TMA, WGMMA and pipelines

TMA moves 1D-5D tiles without per-thread register staging. A reused grouped-MoE or prefill tile can use a producer
warpgroup, a TMA ring, `mbarrier`, consumer WGMMA and `wgmma.commit_group/wait_group`. `setmaxnreg.dec/inc` gives
producer and consumer warpgroups different register budgets; every warp of the group must execute the same aligned
instruction, and an increase can block on the CTA's register pool
([Hopper tuning guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html),
[CUTLASS Hopper tutorials](https://github.com/NVIDIA/cutlass/tree/main/examples/cute/tutorial/hopper)).

`wgmma.mma_async` is an sm_90a warpgroup operation: RS takes A from registers and B from shared memory, SS both from
shared memory. RS saves a shared round trip at the cost of live registers and occupancy. WGMMA reaches peak only
at N >= 64; small-N performance depends on operand path and residency (FINDINGS "The judge's H200"). WGMMA
has no INT4 mode, and its FP8 form needs both operands FP8; an FP8 `mma.sync` on sm_90 is emulated (FINDINGS "Our kernels").

## Format and decode

Hopper has no E2M1 (FP4) conversion: PTX restricts the packed E2M1 forms to sm_100+. NVFP4 W4A16 therefore needs
integer/logic (LOP3) or LUT (`prmt`) reconstruction, E4M3 block scales, BF16 activations and accumulation points,
and an FP32 global scale in the epilogue. Packed FP8 conversion exists (`cvt.rn.f16x2.e4m3x2`), and the query may
round to E4M3 (CONSTRAINTS 6). Every finite E2M1 x E4M3 product is exact in BF16, so a product table
or in-register decode is lossless. The issue cost of decode is in [bound](bound.md) "NVFP4 decode issue".

## Graphs, PDL and persistent grids

CUDA Graphs move launch setup to instantiation; they don't remove GPU-side tails or let a consumer see data before
its producer ends ([CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)).
Programmatic Dependent Launch can start a consumer early, but it must execute `griddepcontrol.wait` before reading
dependent data and overlap isn't guaranteed
([PDL](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html)):
it hides independent setup or prefetch only. A full-grid kernel leaves no SMs for a side-stream branch, so a
prefetch must be issued from inside the running kernels (FINDINGS `d5a90258`). Persistent grids and clusters help
only when their residency and shared-tile assumptions hold; a resident grid can block dependent work.

With page size 1, each logical page is one token row: coalesce a whole aligned K/V row and reuse it across the
query heads of its KV head. The logical page table says nothing about physical pages, TLB reach or bank mapping;
test address permutations instead of assuming a layout.

## Registers and binary inspection

Read `ptxas` resource output and SASS for spills (LDL/STL), load groups (`LDG`, `LDGSTS`, TMA), GMMA/HMMA, LOP3/PRMT,
barriers and dependency chains (`cuobjdump -sass`, `nvdisasm`:
[CUDA binary utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/)). On these memory-bound shapes a spill
usually costs more than one fewer resident warp; unroll only far enough to create independent requests.
`-maxrregcount`, `-dlcm` and similar flags are experiments, not gains. A large fused kernel can outgrow the
instruction cache, whose H200 size no primary source gives: compare SASS sizes rather than assume a threshold.
CuAssembler is unofficial and unvalidated for sm_90a; a patched cubin is allowed (CONSTRAINTS "Boundary cases") but
needs its own H200 correctness and timing.

H200 near/far memory latency, bank mapping, TLB reach and clock behaviour under load remain
unestablished here. External H100/H800 figures do not establish those H200 constants.
