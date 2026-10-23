Serves: G1, G2, G3. Read when sizing a kernel or interpreting a H200 bound.

# H200 and Hopper hardware

Only judge receipts measure H200 kernel or serving performance; current values and profiles are in [FINDINGS](../FINDINGS.md). This file keeps hardware specifications and laws that remain valid across candidates. Local sm_120 timing is for correctness and loose ranking, not H200 evidence; machine access is documented in [RESOURCES](../RESOURCES.md).

## H200 facts

| Parameter | Value | Source |
|---|---:|---|
| HBM3e | 141 GB, 4.8 TB/s | [NVIDIA H200](https://www.nvidia.com/en-us/data-center/h200/) |
| Hopper die | 132-SM H200 SXM configuration; six-stack/full-GH100 descriptions explain the architecture | [Hopper architecture](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/), [DeepGEMM issue 107](https://github.com/deepseek-ai/DeepGEMM/issues/107) |
| Shared memory | 228 KB per SM, 227 KB per block | [Hopper tuning guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html) |
| Registers | 64K 32-bit registers per SM; up to 255 per thread | [Hopper tuning guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html) |
| Dense tensor rates | 989 TFLOP/s BF16 and 1,979 TFLOP/s FP8 are the model's H200 rates | [NVIDIA H200](https://www.nvidia.com/en-us/data-center/h200/), [FINDINGS](../FINDINGS.md) |
| Serving fabric | 8x H200 HGX uses NVLink/NVSwitch; TP8 all-reduce evidence is receipt/source dependent | [Hopper architecture](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/), [FINDINGS](../FINDINGS.md) |

H200 uses Hopper sm_90; WGMMA/TMA paths requiring `sm_90a` are available on the judge. The local RTX PRO 6000 sm_120 and GB10 sm_121 machines have different memory systems and execute neither Hopper WGMMA/TMA nor FA3; do not transfer their timings.

## Format and execution laws

Hopper has no hardware E2M1 (FP4) conversion and no INT4 WGMMA mode. NVFP4 W4A16 therefore needs integer/logic or LUT reconstruction, E4M3 block scales, BF16 activation/accumulation points, and an FP32 global-scale epilogue. Hopper supports packed FP8 E4M3 conversion such as `cvt.rn.f16x2.e4m3x2`. The precision and activation contract is in [CONSTRAINTS](../CONSTRAINTS.md). Sources: [PTX conversion and WGMMA](https://docs.nvidia.com/cuda/parallel-thread-execution/), [Hopper SASS study](https://arxiv.org/abs/2501.12084).

HBM rate is a request-concurrency problem: aligned, coalesced loads and enough independent requests matter more than a single wide load. TMA avoids per-thread register staging and is appropriate for reused tiles; it can lose to ordinary global loads for a one-pass skinny stream. WGMMA RS uses register A/shared B; SS uses shared A/B. H800 measurements show N-dependent RS/SS behavior but are not H200 Muse timings: [Hopper microbenchmarks](https://arxiv.org/abs/2501.12084), [Hopper tuning guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html).

CUDA cache modifiers, prefetch sizes, PDL, graph replay, register allocation, and cluster behavior are performance mechanisms with no universal gain. Their detailed constraints and unresolved H200 facts are in [research_hw_lowlevel](research_hw_lowlevel.md). The judge's fixed replay adjustment and all candidate-specific bandwidth/timing claims belong in [FINDINGS](../FINDINGS.md), not here.

## Sources and unresolved values

Primary sources: [H200 product page](https://www.nvidia.com/en-us/data-center/h200/), [Hopper architecture](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/), [Hopper tuning guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html), [CUDA throughput table](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-c-programming-guide/index.html), [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html), and Hopper microbenchmarks [arXiv:2501.12084](https://arxiv.org/abs/2501.12084).

H200 HBM latency, L2 near/far latency and bandwidth, bank mapping, TLB reach, Hopper PDL saving, cluster launch cost, LOP3/PRMT throughput, and clock locking remain unverified for the judge unless [FINDINGS](../FINDINGS.md) records a receipt.

Additional primary measurements and hardware references retained from the audit:

- [Ujeniya et al., arXiv:2604.11391](https://arxiv.org/abs/2604.11391) is the source for H100/H200 power-cap and clock observations; values read from its figures are not a product specification.
- [NVSHMEM on 8x H200, arXiv:2606.05951](https://arxiv.org/abs/2606.05951) and [SiFAR all-reduce on 8x H200, arXiv:2607.08973](https://arxiv.org/abs/2607.08973) are the published interconnect comparisons; they do not replace current serving receipts.
- [H100 SXM global-load concurrency, arXiv:2608.15764](https://arxiv.org/abs/2608.15764) supplies H100 load-concurrency evidence, not an H200 latency specification.
- The H200 NVL deviceQuery record ([cluster-gpu wiki](https://deac-wiki.readthedocs.io/en/latest/cluster-gpu.html)) and the frontiers graph-boundary issue ([issue](https://github.com/vectorware-inc/frontiers-feedback/issues/24)) are corroborating measurements, not NVIDIA product specifications.

The retained external measurements provide bounds, not portable constants. The H100/H200 power-cap study ([arXiv:2604.11391](https://arxiv.org/abs/2604.11391)) is the source for the observed memory-bound versus tensor-heavy clock ranges. H100 global-load-concurrency work ([arXiv:2608.15764](https://arxiv.org/abs/2608.15764)) supports the unloaded-memory-latency/concurrency model; its roughly 350–400 ns latency is H100 evidence, not an H200 specification. The H200 NVL deviceQuery record ([cluster-gpu wiki](https://deac-wiki.readthedocs.io/en/latest/cluster-gpu.html)) corroborates the driver-reported 60 MB L2 observation.

For serving interconnect, the NVSHMEM 8x-H200 study ([arXiv:2606.05951](https://arxiv.org/abs/2606.05951)) reports roughly 2.4–3.1 us for its best small all-reduces, while SiFAR's 8x-H200 results ([arXiv:2607.08973](https://arxiv.org/abs/2607.08973)) and NCCL/NVLS measurements support about 5–7 us for 8–32 KB TP8 reductions and roughly 276 GB/s algorithm bandwidth for large NVLS reductions. These are implementation measurements, not a guarantee for the serving harness. The graph-boundary issue ([frontiers issue](https://github.com/vectorware-inc/frontiers-feedback/issues/24)) reports about 1.0 us empty-kernel and 1.3 us graph versus 2.1 us stream launch behavior; it is H100-era evidence for why graph tails matter, not an H200 fixed cost.
