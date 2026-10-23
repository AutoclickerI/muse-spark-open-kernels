# Hardware: the judge's H200 and our machines

Reference for kernel design and for `sim/hw.py`. Tags: **SPEC** vendor peak; **MEAS** measured (by
whom, on what); **EST** arithmetic or estimate; **MEM** from memory, unchecked. Research done
2026-10-23 (web agent, primary sources listed at the end); numbers our own runs measured on the
judge are in [FINDINGS](FINDINGS.md).

## The mismatch

| | Judge (kernel track) | Judge (serving) | This box | capp DGX |
|---|---|---|---|---|
| GPU | 1x H200 SXM, sm_90 | 8x H200 SXM, NVSwitch | 2x RTX PRO 6000 Blackwell Max-Q, sm_120 | 4x GB10, sm_121 |
| HBM | 4.8 TB/s HBM3e | same | ~1.8 TB/s GDDR7 (MEM) | ~273 GB/s LPDDR5x (MEM) |
| FP4 tensor cores | no | no | yes | yes |
| wgmma, sm90 TMA paths, FA3 | yes | yes | no | no |

Local runs check correctness and graph capture; their timings are not H200 timings. Only `okc check`
measures the judge's GPU.

## H200 SXM (GH100, the same die as H100)

| Parameter | Value | Tag, source |
|---|---|---|
| SMs | 132 | SPEC [2], MEAS [9] |
| SM clock | 1980 MHz on memory-bound kernels at 700 W (~550 W drawn); 1.78-1.85 GHz on tensor-heavy work | MEAS H200 [8] |
| HBM3e | 141 GB, 6 stacks, 6144-bit, 4.8 TB/s | SPEC [1] |
| HBM achievable | ~4.45 TB/s read-heavy (~92%); 4.3 TB/s copy (SGLang `hopper_bf16_gemv.py` note); 3.6 TB/s 1R:1W | MEAS H200 [8] |
| HBM latency | ~350-400 ns unloaded | MEAS H100 [7], EST for H200 |
| Bytes in flight to saturate | >= 13 KB per SM by Little's law; target >= 32 KB per SM, >= ~60 SMs streaming | EST [6] |
| L2 | **60 MB** (driver-reported on H200; H100 SXM has 50 MB), two partitions, far hit ~1.6-1.9x near | MEAS [9][10], [6][7] |
| L2 hit latency | ~145 ns near, ~270 ns far | MEAS H100 SXM [7] |
| Shared memory | 228 KB per SM, 227 KB per block | SPEC [3] |
| Registers | 64K x 32-bit per SM, <= 255 per thread | SPEC [3] |
| BF16 tensor, dense | 989 TFLOP/s (judge's floor); wgmma reaches ~96% of peak | SPEC [1], MEAS H800 [6] |
| FP8 tensor, dense | 1979 TFLOP/s | SPEC [1] |
| mma.sync efficiency | ~63% of peak | MEAS H800 [6] |
| wgmma at small N (tokens on N) | N=8: 22-30%; 16: 39-60%; 32: 65-97%; >=64: ~99% of peak (SS / RS) | MEAS H800 [6] |
| CUDA-core issue | 128 FP32 FMA, 256 HFMA2 results, 64 INT/LOP, 16 "other conversions" per SM per clock | SPEC [4] |

### What Hopper lacks for this model's formats

- No FP4 (E2M1) conversion and no FP8-to-BF16x2 conversion: `cvt.rn.bf16x2.{e4m3x2,e2m1x2}` need
  sm_100f+; only `cvt.rn.f16x2.e4m3x2` exists on sm_90 [5]. NVFP4 dequant is integer/logic work
  (LOP3/PRMT/shifts) plus HFMA2/HMUL2.
- Dequant budget at line rate (EST): NVFP4 is 4.5 bits per weight, so 4.4 TB/s is ~30 weights per clock
  per SM against 64 INT ops per clock: ~2 INT ops per weight, little left for address math. Marlin's
  FP4 decode is ~1.25-1.5 INT ops per weight (agent's count from source).
- E2M1 x E4M3 has at most 5 significant bits, so the BF16 product is exact; the FP32 per-expert global
  scale can go in the epilogue (EST).

## Launch and synchronization

| Parameter | Value | Tag, source |
|---|---|---|
| Kernel boundary inside a CUDA graph | ~1.0 us (empty kernels); 1.3 us graph vs 2.1 us stream launch gap | MEAS H100 [13][14] |
| Dependent memory-bound kernel, no PDL | ~1.5-3 us per boundary (gap + tail + HBM ramp) | EST |
| `grid.sync()` | ~1.3 us | MEAS H100 [13] |
| PDL saving | ~0.5 us per boundary on B200; Hopper unmeasured (EST 0.3-1 us). The bigger win: issue the next kernel's weight prefetch before `griddepcontrol.wait` | MEAS B200 / EST |
| The judge | subtracts a graph replay's fixed cost (~4 us; `replay_overhead` in each run's records) from both sides | MEAS, our runs |

## Interconnect (serving: 8x H200 HGX)

| Parameter | Value | Tag, source |
|---|---|---|
| NVLink 4 | 18 links, 900 GB/s bidirectional, 450 GB/s per direction; NVSwitch with multicast and SHARP (NVLS) | SPEC [2] |
| TP8 all-reduce, 8-32 KB | ~5-7 us (NCCL ring 4.7-8.9, NVLS 5.6-5.9, vLLM one/two-shot ~5-6.6, research 2.4-3.1) | MEAS 8x H200 [17][18] |
| TP8 all-reduce, large | NCCL NVLS ~276 GB/s algorithm bandwidth; 128 KB-1 MB on H200 unmeasured (EST 6-10 us) | MEAS [17] / EST |
| Decode payloads | [tokens, 8192] BF16 = 16 KB per token: 128 KB at 8 tokens, 4 MB at 256 | static |

## Sources

1. https://www.nvidia.com/en-us/data-center/h200/
2. https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/
3. https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html
4. https://docs.nvidia.com/cuda/archive/12.6.0/cuda-c-programming-guide/index.html (arithmetic throughput table)
5. https://docs.nvidia.com/cuda/parallel-thread-execution/index.html (cvt, wgmma)
6. arXiv 2501.12084 (Luo et al., Hopper microbenchmarks, H800 PCIe)
7. arXiv 2608.15764 (H100 SXM global-load concurrency; preprint)
8. arXiv 2604.11391 (Ujeniya et al., H100 and H200 under power caps; values read off figures)
9. https://github.com/deepseek-ai/DeepGEMM/issues/107 (H200: 132 SMs, 60 MB L2)
10. https://deac-wiki.readthedocs.io/en/latest/cluster-gpu.html (H200 NVL deviceQuery)
13. https://github.com/vectorware-inc/frontiers-feedback/issues/24
14. https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles
17. arXiv 2606.05951 (NVSHMEM on 8x H200)
18. arXiv 2607.08973 (SiFAR all-reduce, 8x H200; values read off figures)

Not verified: H200 HBM latency and L2 bandwidth, PDL on Hopper, all-reduce at 128 KB-1 MB on H200,
PRMT/F2FP throughput, whether the judge locks clocks.
