# H200 low-level levers for Muse Spark Open

This note extends HARDWARE.md; it does not replace the measurements in FINDINGS.md. The target is one H200 SXM (Hopper sm_90/sm_90a, 132 active SMs in the judge) running graph-replayed kernels. The relevant cases are skinny BF16 GEMV (m 1--256), grouped NVFP4 W4A16 MoE (bs 8--256), paged FP8 GQA with page size 1 (q_len 1 and 8), and whole decode/prefill layers. A cold-L2 read precedes every scored replay.

Evidence labels used below:

* **MEAS** is a measurement already in this repository. It is the best evidence for the actual judge, but is not a general hardware specification.
* **SPEC** is NVIDIA documentation or a primary paper. A result measured on H800/H100 is labelled as such and is not silently promoted to an H200 result.
* **INFER** is a calculation or expected range from the above evidence. It is a test hypothesis, not a measured gain.
* **MEM/UNVERIFIED** is a lead from memory or an implementation detail for which no H200 measurement was found. It must not be reported as a result.

## Ranked levers

The gain column is an engineering expectation for these workloads, not a published H200 number. It is deliberately a range so that a diagnostic check can falsify it.

| Rank | Lever | Best target | Evidence and expected gain | Effort / confidence |
|---:|---|---|---|---|
| 1 | Keep more global-load requests in flight; use aligned 128-bit loads and enough CTAs | BF16 GEMV, layer dense weights, attention KV streams | **MEAS:** our 52--61 MB H200 stream reached 4.04--4.25 TB/s; best was 132 CTAs × 1024 threads with four 16-byte loads in flight per thread. **SPEC:** an H800 study reached 90--91% of theoretical global bandwidth with vectorized loads and sufficient blocks ([paper](https://arxiv.org/abs/2501.12084)). **INFER:** 0--8% for the already-best stream, 5--20% for kernels that expose too little memory-level parallelism. | Low / high |
| 2 | Use TMA only where a tile is reused; use a producer/consumer pipeline and wgmma for grouped GEMM | NVFP4 MoE, decode/prefill expert tiles | **SPEC:** TMA removes register traffic and most per-thread address instructions ([Hopper tuning guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)); the H800 study measured roughly 600 vs 450 TFLOP/s for large FP16 GEMM with vs without TMA, but little difference below n<2000 ([paper](https://arxiv.org/abs/2501.12084)). **INFER:** 8--25% on large MoE/prefill tiles, near zero or negative for a one-pass skinny stream. | High / medium |
| 3 | Prefer wgmma register-source A (RS) when dequantized activations are already in registers; choose N >= 64; pipeline commit_group/wait_group | MoE W4A16 and large prefill grouped GEMM | **SPEC:** on H800, dense wgmma throughput was near peak for N>=64; at N=8/16/32, RS was materially better than SS (for example FP16 RS 215/426/634 TFLOP/s versus SS 158/287/477 in the study table) ([paper](https://arxiv.org/abs/2501.12084)). The ISA defines RS as register A plus shared-memory B and requires sm_90a ([PTX wgmma](https://docs.nvidia.com/cuda/parallel-thread-execution/)). **INFER:** 5--20% for expert tiles whose N can be 64+, less for decode-sized N. | High / medium |
| 4 | Keep one-pass weights out of L1; use L2 prefetch/eviction hints only for data with reuse | Dense projections, expert weights, attention K/V | **SPEC:** PTX exposes L1::no_allocate, L2::evict_first/last, 64/128/256-B prefetch-size hints, and createpolicy; all are explicitly performance hints ([PTX load instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/)). **MEAS:** H200 L2 re-read was 8.2--9.6 TB/s for 4--16 MB and 4.5--5.1 TB/s at 32 MB. **INFER:** 0--12% when a kernel currently pollutes L2 or misses a tile reuse opportunity; zero on the judge cold-L2 first read. | Low--medium / medium |
| 5 | Make page-1 KV gathers coalesced by row and test address/page coloring | FP8 paged attention and whole decode layers | **SPEC:** H100/Hopper has partitioned L2 with near/far latency; the H800 study measured near/far groups and says access placement matters ([paper](https://arxiv.org/abs/2501.12084)). CUDA reports that device allocations prefer 2 MiB physical pages ([CUDA Programming Guide, unified memory](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html)). There is no public H200 TLB-reach number for this exact page table. **INFER/UNVERIFIED:** 3--15% on random page-1 attention if gathers defeat coalescing or TLB locality; 0% if FA3 already achieves the same layout. | Medium / low--medium |
| 6 | Fuse short stages and use persistent work only when it removes real graph-node tails; test PDL for overlap of independent preambles | Attention decode, decode layer, router/glue | **MEAS:** the judge reports about 4.2 us fixed graph replay overhead and very short sliding attention baselines. **SPEC:** CUDA Graphs amortize host submission setup ([graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)); Hopper PDL can schedule a dependent grid after griddepcontrol.launch_dependents before the producer exits, with a required dependent wait ([PDL guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html), [PTX](https://docs.nvidia.com/cuda/parallel-thread-execution/)). No H200 PDL saving was found. **INFER:** 2--10% for many-kernel layer glue; no effect when one fused node dominates. | Medium--high / low |
| 7 | Use packed FP8 conversion for allowed query rounding; avoid scalar conversion and repeated unpack/pack | FP8 KV attention | **SPEC:** cvt.rn.f16x2.e4m3x2 and the reverse packed FP16 conversion are supported for sm_90 ([PTX conversion](https://docs.nvidia.com/cuda/parallel-thread-execution/)); Hopper FP8 tensor cores accept E4M3/E5M2. No public sm90 conversion throughput number was found. **INFER:** 1--5% of attention time when conversion is visible; often hidden by KV loads. | Low / medium |
| 8 | Decode E2M1 with integer bit logic and pack once; do not seek a Hopper FP4 converter | NVFP4 MoE | **SPEC:** current PTX target notes restrict E2M1 conversion to Blackwell family targets (sm_100f/sm_110f/sm_120f), while E4M3 conversion supports sm_90; Hopper wgmma has no INT4 mode ([PTX conversion targets](https://docs.nvidia.com/cuda/parallel-thread-execution/), [Hopper study SASS table](https://arxiv.org/abs/2501.12084)). **INFER:** LOP3/shift/PRMT plus BF16 multiply is the required Hopper path; 5--20% is possible only if current dequant is instruction- or register-bound. Exact LOP3/PRMT issue cost for H200 is **UNVERIFIED**. | Medium / high for premise, low for gain |
| 9 | Control register budget and inspect SASS for spills, dependency chains, issue slots, and code size | All custom CUDA; especially fused layers and warp-specialized MoE | **SPEC:** Hopper has a 64K 32-bit-register file per SM; setmaxnreg can redistribute registers between warpgroups within a CTA ([Hopper tuning guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html), [PTX setmaxnreg](https://docs.nvidia.com/cuda/parallel-thread-execution/)). ptxas supports -O3, --allow-expensive-optimizations, -maxrregcount, -dlcm, and verbose resource output ([nvcc/ptxas](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html)). **INFER:** 0--10%; a spill or a long dependency chain can dominate a small kernel, but changing limits can also reduce occupancy. | Low--medium / medium |
| 10 | Use clusters/DSM only for demonstrable cross-CTA reuse; otherwise avoid cluster launch and synchronization overhead | Prefill tiles, possible expert-weight multicast | **SPEC:** Hopper clusters guarantee concurrent CTAs and expose distributed shared memory/TMA multicast ([Hopper blog](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/), [tuning guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)). Public sources give no H200 cluster-launch-cost number. **INFER/UNVERIFIED:** useful only when many CTAs consume the same tile; likely negative for page-1 attention and skinny GEMV. | High / low |

## 1. HBM3e is limited by request concurrency, not by one load instruction

NVIDIA rates H200 at 141 GB HBM3e and 4.8 TB/s ([H200 product page](https://www.nvidia.com/en-us/data-center/h200/)). The judge measured 4.04--4.25 TB/s, or about 84--89% of that specification. The current best configuration is a useful target: 132 CTAs, 1024 threads per CTA, and four outstanding 16-byte loads per thread. That is a repository measurement, not a claim that every kernel should launch 132 CTAs.

The H800 microbenchmark is the closest public Hopper result. It uses vectorized global reads and reports 90--91% of theoretical global-memory bandwidth; larger load sizes and more launched blocks raise throughput ([Luo et al., 2025, memory throughput](https://arxiv.org/abs/2501.12084)). The earlier Hopper study also benchmarks FP8 tensor cores, asynchronous copies, and instruction throughput ([Luo et al., 2024](https://arxiv.org/abs/2402.13499)). NVIDIA's Hopper documentation says TMA can move large blocks with one issuing thread and avoid register traffic ([Hopper tuning guide, TMA](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)), but the same paper measured TMA about 170 cycles slower than a regular access in its latency test and recommends non-tensor 1D transfers when a tensor map would constrain the load size. Thus:

* BF16 GEMV should start with coalesced global loads equivalent to LDG.128, several independent addresses per thread, and enough CTAs to keep all HBM partitions busy. A 128-bit load is a request-width choice; it does not by itself create enough outstanding requests.
* TMA is a candidate for a tiled GEMM where a block consumes a tile repeatedly. It is not a default replacement for a streaming 1D weight read.
* Measure bytes transferred and the actual kernel time together. A wider load can lower instruction count while making register pressure, tail predication, or overfetch worse.

The repository's prior Little's-law estimate (about 13 KB of outstanding data per SM as a minimum, with about 32 KB per SM a safer target) is **INFER**, not an H200 counter. The direct H200 evidence is the four 16-byte loads per thread in the best stream; a follow-up probe should report both bytes outstanding and the number of independent load instructions per SM.

The HBM channel/bank mapping is not documented at the level needed to predict a particular virtual-address permutation. The portable rule from CUDA best practices is still valid: align and coalesce, use unit-stride rows, and test a stride or expert layout rather than assuming that a contiguous allocation maps ideally to every channel. HBM3e raises the rate over H100; it does not remove bank conflicts or address-translation latency.

**Probe:** in muse_bf16_proj, time the same 52--61 MB read with 16, 32, 64, and 128-bit vector loads; with 1, 2, and 4 independent loads per thread; and with 66, 132, and 264 CTAs. Record achieved TB/s, registers, and spill bytes from ptxas. In MoE, repeat for a contiguous expert tile and the actual routed gather. The question is whether the candidate is HBM-request-limited or already instruction/register-limited.

## 2. L2 sectors, promotion, and two-partition locality

PTX exposes cache controls on global loads. The ISA lists 64/128/256-byte prefetch-size hints, L1::no_allocate, L1::evict_first/last, L2::evict_first/last, and L2::cache_hint; createpolicy constructs an eviction policy passed to a load. The separate prefetch.global.L2 form and cp.async.bulk.prefetch.L2.global both initiate cache prefetches without returning data to the issuing thread. The ISA states that prefetch and cache-policy values are performance hints and may be ignored, so they are safe experiments rather than contracts ([PTX ld.global and prefetch](https://docs.nvidia.com/cuda/parallel-thread-execution/), examples and target notes).

For this challenge the policy should follow reuse:

* A dense projection or expert weight stream is mostly one-pass after the judge's cold-L2 flush. Try ld.global.nc with L1::no_allocate or a streaming/evict-first hint only if it prevents useful input/output data from being displaced. ld.global.nc uses a non-coherent read-only cache and is not automatically faster; the ISA says its longer latency needs sufficient parallelism.
* An attention K/V tile, an expert tile used by several tokens, or a weight tile reread inside a grouped GEMM can justify L2::evict_last or a bounded cp.async.bulk.prefetch.L2. Do not mark a whole 28 MB expert as persisting when the reusable tile is much smaller.
* A 128- or 256-byte prefetch hint can fetch adjacent sectors, but it can overfetch page-1 gathers. Test 64/128/256 separately.

The H800 paper finds a dual-partitioned L2, distinct near and far hit latencies, and an H800 L2 throughput of 4.23 times its global-memory throughput. It reports that 8 MB accesses can be near or far hits and 40 MB accesses expose near/far misses ([paper, partitioned L2](https://arxiv.org/abs/2501.12084)). Treat the two partitions as separate locality domains for a probe; the paper assumes equal capacity, but the physical address hash is not public. Our H200 probe sees 8.2--9.6 TB/s for a 4--16 MB reread and only 4.5--5.1 TB/s at 32 MB. That is consistent with capacity/partition pressure, but it does not identify the H200's physical hash function.

CUDA's runtime L2 access-policy window can reserve a set-aside for persisting data, but the guide says the default maximum is at most 75% of L2 and that the selected accesses are probabilistic according to hitRatio ([L2 cache control](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/l2-cache-control.html)). The scored graph begins cold, and a Python submission normally cannot edit the judge's graph-node attribute, so PTX hints inside the kernel are the practical lever.

**Probe:** for each dense and expert stream, bundle ordinary global, L2::128B, L2::256B, nc.L1::no_allocate, and cp.async.bulk.prefetch.L2.global variants. Use a 4, 16, and 32 MB reread and a cold 52 MB stream. For attention, compare contiguous page rows with an address permutation that changes the 128-byte row's alignment. The question is whether a hint changes the judge's cold-first replay or only an artificial warm reread.

## 3. TMA, cp.async.bulk, and warp specialization

TMA is a Hopper asynchronous copy engine. NVIDIA documents global-to-shared and shared-to-global 1D--5D transfers, address generation by a tensor map, no register staging, and producer/consumer warp specialization ([Hopper tuning guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html); [Hopper architecture blog](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)). PTX also has cp.async.bulk and a non-blocking cp.async.bulk.prefetch.L2.global; the prefetch size must be a multiple of 16 bytes and the operation is only a hint ([PTX bulk prefetch](https://docs.nvidia.com/cuda/parallel-thread-execution/)).

TMA's useful design is therefore:

1. One producer warpgroup issues a deep ring of TMA copies into shared memory.
2. Consumer warpgroups execute wgmma against the previous tile while the next tile is in flight.
3. mbarrier and wgmma.commit_group/wait_group expose only the dependency that is actually needed.
4. setmaxnreg.dec gives producer warps back registers and setmaxnreg.inc gives consumer warps a larger accumulator budget (subject to the PTX rules that every warp in the warpgroup must execute it and values must be multiples of eight).

This maps to large grouped MoE and prefill tiles, where one weight tile serves many token rows. It maps poorly to a single skinny GEMV, where the tile has little reuse and TMA/barrier setup becomes visible. The H800 paper's application test reaches close to 600 TFLOP/s with TMA versus about 450 without only for large matrices; it explicitly reports little difference below n=2000 and recommends a non-tensor path for constrained 1D loads ([paper](https://arxiv.org/abs/2501.12084)). These values are H800 FP16 GEMM measurements, not expected Muse gains.

The implementation pattern in [CUTLASS's Hopper CuTe tutorials](https://github.com/NVIDIA/cutlass/tree/main/examples/cute/tutorial/hopper), [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM), [FlashAttention-3's Hopper kernels](https://github.com/Dao-AILab/flash-attention/tree/main/hopper), and [ThunderKittens](https://github.com/HazyResearch/ThunderKittens) is useful as source-level corroboration: producer warps issue asynchronous copies, consumers issue WGMMA, and barriers delimit tiles. These projects are not H200 measurements and their tile choices are not automatically correct for page-1 attention or Muse's routed skew.

**Probe:** in muse_moe_w4a16 and muse_prefill_layer, compile one kernel family with (a) scalar/vector global-to-register loads, (b) cp.async/shared-memory stages, and (c) TMA producer/consumer stages at two depths. Time bs 8, 128, 256 and the 16K-token prefill shape. Record TMA bytes, barrier wait time, registers, and achieved WGMMA throughput. The question is whether data movement or WGMMA occupancy is the limiting stage.

## 4. Paged FP8 attention: row coalescing, TLB reach, and partition placement

The scored cache has page size 1, so a logical page is one token row. Each K/V row has many contiguous bytes, but requests from different query heads and requests can land on unrelated rows. The first-order optimization is to have a warp (or a small warpgroup) read a complete 128-byte-aligned row segment, keep K/V traversal in the same order for all query heads sharing a KV head, and avoid scalar gathers that turn one row into many sectors. For page sizes 64 in tests, a tile copy can recover regularity; a page-size-dependent branch is allowed because it depends on configuration, not input values.

Hopper's L2 is partitioned. The public H800 measurements show that a working set below half L2 can remain in one partition and that accesses to a far partition have higher latency; the study observed near/far groups at 8 MB and near/far misses at 40 MB ([paper](https://arxiv.org/abs/2501.12084)). This supports testing row alignment and a stable address permutation, but it does not justify skipping keys or assuming a particular hash.

CUDA's memory guide says GPU-resident allocations prefer 2 MiB physical pages or larger and explains that TLBs cache virtual-to-physical mappings ([unified memory and physical pages](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html)). It does not publish H200 TLB entry counts or reach. A page-size-1 logical cache therefore does not imply 1-byte or 1-page physical mappings, and a claim that TLB misses dominate is UNVERIFIED until a probe varies the number of distinct physical rows.

**Probe:** in muse_attention_decode, hold Q, context, and arithmetic constant while varying (i) page-size 1 versus 64, (ii) contiguous versus randomized physical row order, (iii) 64/128/256-byte row-aligned loads, and (iv) K/V row reuse across query heads. Include context 2K, 16K, 80K, and 1M. Time sliding and full attention separately. The question is whether the bottleneck is row-sector traffic, address translation, or softmax/launch overhead.

## 5. Graph replay, PDL, tails, persistent work, and clusters

CUDA Graphs move much of launch setup to graph instantiation and launch the executable graph repeatedly with little host submission work ([CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html)). The competition already captures each shape, and FINDINGS.md records about 4.2 us of fixed graph replay cost that the judge subtracts. That makes graph-node count and the final tail of a fused layer more relevant than ordinary stream-launch microbenchmarks. A Hazy Research H100 measurement found about 2.1 us for a dummy stream launch and about 1.3 us with CUDA graphs, with the residual GPU-side launch/teardown cost still visible ([Look Ma, No Bubbles](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)). This is an H100 engineering measurement, not a promised H200 per-node cost.

Programmatic Dependent Launch is available from compute capability 9.0. A producer issues griddepcontrol.launch_dependents; the dependent can begin before the producer exits, but the runtime does not guarantee overlap, and the dependent must use griddepcontrol.wait before consuming producer results ([CUDA PDL guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html), [PTX instruction](https://docs.nvidia.com/cuda/parallel-thread-execution/)). CUDA Graphs can encode programmatic dependency edges. This can hide a dependent kernel's zeroing, address setup, or weight prefetch; it cannot make a dependent result available early.

No primary source measured a Hopper PDL saving in the Muse shape regime. The Hazy measurement says PDL's coarse cudaGridDependencySynchronize can itself introduce stalls when only chunks are ready; their megakernel avoids that with chunk counters. The HARDWARE.md estimate of 0.3--1 us should remain an estimate. Persistent kernels can remove inter-kernel tails, but they must still make progress and return all graph outputs; a persistent grid that monopolizes SMs can prevent a dependent graph node from launching. Cluster launches add a scheduling guarantee and DSM/TMA options, but a cluster is useful only if CTAs actually share a tile. There is no public H200 cluster-launch-cost number.

**Probe:** in muse_decode_layer, bundle (a) the current multi-kernel graph, (b) the same operations fused into one kernel, (c) a persistent single-grid version, and (d) a two-node graph with PDL-enabled prefetch/setup. Use batch 1, 16, and 32, sliding/full, and verify q_len=8. Compare raw replay time, judge-adjusted time if available, and a kernel trace showing whether the dependent starts before the producer tail. The question is whether launch/tail time or memory/compute work dominates.

## 6. WGMMA versus mma.sync, and register-source A

The PTX ISA requires wgmma.mma_async to target sm_90a; the operation is asynchronous and warpgroup-wide. Its RS form supplies A from registers and B from a shared-memory descriptor; its SS form supplies both A and B from shared memory. The ISA examples and operand rules are in [PTX wgmma.mma_async](https://docs.nvidia.com/cuda/parallel-thread-execution/). The Hopper microbenchmark paper independently describes RS versus SS and reports that dense WGMMA is near peak for N>=64; with smaller N, RS has higher throughput because it avoids exposing as much shared-memory latency ([paper, WGMMA](https://arxiv.org/abs/2501.12084)).

For NVFP4 W4A16, the natural RS experiment is to decode a small A tile (activation or dequantized values) into registers, while TMA loads B/weights into shared memory. This saves a register-to-shared round trip but increases live registers and can lower occupancy. If dequantization is done for B, RS cannot directly make B register-sourced; the tile must be placed in shared memory for the standard RS/SS forms. The best choice is therefore a layout question, not a slogan that WGMMA is always faster.

The same paper's H800 table reports dense FP16 WGMMA throughput around 719--732 TFLOP/s at N=64--256, but only 158--634 TFLOP/s for N=8--32 depending on RS/SS. Those are instruction microbenchmarks on H800 with different clocks and power behavior; they are evidence for the shape trend, not a H200 result. mma.sync remains a portable fallback and can be preferable when N is tiny, when a kernel cannot afford WGMMA's warpgroup barriers, or when compilation must support non-Hopper GPUs.

**Probe:** for MoE and prefill, hold the tile shape fixed and compare mma.sync, WGMMA-SS, and WGMMA-RS at N=8, 16, 32, 64, and 128. Use two accumulator budgets and a setmaxnreg producer/consumer split. Disassemble each cubin and record GMMA, HMMA, LDGSTS, barrier, and wait instructions. The question is whether the candidate is tensor-core issue-limited, shared-memory-latency-limited, or register/occupancy-limited.

## 7. FP8 conversion and the Hopper FP4 boundary

Hopper supports packed E4M3/E5M2 conversion instructions. PTX documents cvt.rn.f16x2.e4m3x2 (two E4M3 values to packed FP16) and the reverse conversion as requiring sm_90 or newer; the PTX target notes distinguish this from newer BF16-to-FP8 variants ([PTX conversion section](https://docs.nvidia.com/cuda/parallel-thread-execution/)). This is the appropriate primitive for the attention query rounding that the challenge permits. Pack two values per instruction, keep conversion adjacent to the load/use, and do not round queries more coarsely than E4M3.

The current PTX manual lists E2M1 conversion syntax, but its target table restricts the E2M1-to/from-FP16/BF16 and FP32 forms to Blackwell-family targets (sm_100a/sm_100f and later), not sm_90. This agrees with the repository's HARDWARE.md: Hopper has no hardware E2M1 converter. The Hopper study's SASS table also shows INT4 mma becoming CUDA-core IMAD sequences and WGMMA having no INT4 mode ([paper, SASS table](https://arxiv.org/abs/2501.12084)).

Consequently, NVFP4 decode on H200 is an integer/bit-manipulation path: extract nibbles, map E2M1 codes to BF16-compatible bit patterns or small LUT entries, apply the per-block E4M3 scale, and feed BF16/FP16 tensor operations. LOP3, shifts, PRMT, and table lookup are candidate sequences; their exact issue rates, register-bank behavior, and cost per weight on H200 are UNVERIFIED. A compiler-generated conversion that silently widens through FP32 is not a valid replacement for the required precision or bandwidth.

**Probe:** compile a standalone Hopper-only decoder with one E2M1 nibble per lane and compare (i) LOP3/shift, (ii) PRMT plus lookup, (iii) a 16-entry shared-memory/constant LUT, and (iv) the existing Marlin decoder. Measure decoded weights per integer issue cycle, integer pipe utilization, registers, and spills for 4K, 28 MB, and routed expert tiles. The question is whether the present MoE candidate is spending more time on decode logic or on weight traffic/tensor math.

## 8. Register allocation, issue behavior, and instruction-cache pressure

Hopper's register file is 64K 32-bit registers per SM, with at most 255 registers per thread in the normal CUDA model ([Hopper tuning guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html)). setmaxnreg is a warpgroup instruction, not a free occupancy knob: every warp in the group must execute the same aligned instruction, inc can block waiting for the CTA register pool, and the requested count is a multiple of eight ([PTX setmaxnreg](https://docs.nvidia.com/cuda/parallel-thread-execution/)).

Use ptxas -v and SASS to establish three limits before tuning:

1. A spill (LDL/STL or compiler spill diagnostics) is usually more damaging than one fewer resident warp on these memory-bound shapes.
2. A long RAW dependency in a load/decode/accumulate chain can leave issue slots idle even when occupancy is high. Unroll only far enough to create independent requests; excessive unrolling increases register count and instruction-cache footprint.
3. A large fused decode/prefill kernel can exceed the instruction-cache working set. There is no public H200 instruction-cache capacity/penalty table in the primary sources used here, so any claim of a particular threshold is MEM/UNVERIFIED. Inspect SASS size and compare a compact loop with a fully specialized shape kernel.

The CUDA arithmetic-throughput table gives Hopper's broad issue rates for integer, FP16/BF16, and conversion classes, but it does not provide a per-instruction schedule for every SASS opcode ([CUDA C Programming Guide, throughput table](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-c-programming-guide/index.html)). SASS control fields and nvdisasm output are therefore the practical way to see whether a load, LOP3/PRMT, conversion, or tensor instruction is waiting on a dependency; PTX source alone cannot establish a dual-issue pairing. Use the table to identify a possible pipe bottleneck, then measure a dependent and an independent chain. The H800 paper reports WGMMA power-induced frequency changes under sustained random inputs, so tensor-heavy kernels should also log the judge's reported SM clock rather than assuming the nominal frequency ([paper, power/frequency](https://arxiv.org/abs/2501.12084)).

**Probe:** for each candidate, collect ptxas -v, cuobjdump -sass, and (where available) nvdisasm output. Compare maxrregcount values around the first spill, with and without unrolling, and with two setmaxnreg allocations in the warp-specialized version. The question is whether the proposed optimization removes a real dependency/spill or merely trades one resource for another.

## 9. Binary-level inspection without an H200

The CUDA Binary Utilities are usable on this host even when no Hopper GPU is present. NVIDIA documents cuobjdump -sass for extracting SASS from a cubin/fatbin and nvdisasm for disassembling a cubin ([CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/)). Build explicit compute_90a/sm_90a code for WGMMA/TMA experiments; NVIDIA's compatibility guide says architecture-specific sm_90a code is not forward- or backward-compatible with other architectures ([Hopper compatibility](https://docs.nvidia.com/cuda/hopper-compatibility-guide/)). A portable sm_90 or host-architecture fallback can remain for local correctness, but only sm_90a can exercise architecture-specific WGMMA behavior on the judge.

Inspect at least:

* global-load width and cache modifiers (LDG.E.128, LDG.64/32, LDGSTS/TMA paths), address arithmetic, and predication;
* GMMA/HMMA, LOP3, PRMT, conversion, barrier, and wait instructions;
* local-memory LDL/STL spills and repeated reloads;
* dependency chains between load, decode, tensor issue, and accumulator writeback;
* SASS instruction count and duplicated shape-specialized blocks that may pressure the instruction cache.

ptxas options are a controlled experiment, not a guarantee. -O3 and --allow-expensive-optimizations can change scheduling; -maxrregcount caps registers and can create spills; -dlcm changes the default global-load cache modifier. NVIDIA documents these semantics in the [nvcc compiler guide](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html). Inline PTX can force an instruction form or cache modifier, while CUDA intrinsics let ptxas schedule and select variants; compare the resulting SASS rather than assuming either is faster.

CuAssembler is an unofficial SASS assembler/patcher ([repository](https://github.com/cloudcores/CuAssembler)). Its README claims broad-generation coverage, but public support and encoding quality for sm_90a are not guaranteed by NVIDIA and were not validated here. Treat it as a read-only research tool until a tiny patched cubin passes H200 correctness and timing. NVIDIA's cuobjdump/nvdisasm output is the reliable inspection path; no unsupported SASS patch should be part of a scored submission without a reproducible check.

## 10. What is actually H200-specific

NVIDIA's H200 page specifies 141 GB HBM3e, 4.8 TB/s bandwidth, and up to 700 W TDP ([H200](https://www.nvidia.com/en-us/data-center/h200/)). NVIDIA's Hopper architecture page describes the GH100 full die as six stacks/controllers and 60 MB L2, while the H100 SXM5 product configuration is 132 SM, five HBM3 stacks, and 50 MB L2 ([Hopper architecture](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)). The H200 uses the same Hopper compute generation but a different HBM3e configuration; the product page does not publish a separate H200 L2 table.

This resolves one apparent tension in HARDWARE.md:

* The report's 60 MB H200 L2 is a MEAS/driver observation recorded in HARDWARE.md and corroborated by the full-GH100 architecture description; it is not an H200 product-page specification.
* The public microbenchmark numbers (50 MB L2, 2.039 TB/s H800 memory, 90--91% achieved) are H800/H100-family measurements, not H200 measurements. The H200 probe's 4.04--4.25 TB/s is the authoritative result for this judge.
* H200's higher memory rate makes dense and NVFP4 weight traffic more likely to be HBM-bound, while compute throughput and ISA choices remain Hopper. It does not make H100/H800 WGMMA or L2 percentages portable without rechecking clocks and power.
* The judge reports 1980 MHz SM, 3201 MHz memory, and a 700 W limit. Sustained WGMMA or conversion-heavy code can alter clocks; log clock and power in any diagnostic check before attributing a gain to an instruction.

No primary source found here documents H200 bank mapping, exact H200 TLB reach, H200 L2 near/far latency, PDL savings, cluster launch latency, or LOP3/PRMT throughput. Those rows in HARDWARE.md remain estimates or measurements from the repository, and the probes below are the way to turn them into H200 evidence.

## Probe plan

Bundle these variants into one diagnostic check per problem. Each check should compile all variants once, warm up outside the timed loop, time each with CUDA events inside the allowed diagnostic harness, and print only scalar timings and resource counters. Keep the scored kernel itself graph-capturable; no input-dependent host decisions belong in kernel_fn.

### muse_bf16_proj (skinny BF16 GEMV and router)

1. **Global-load ILP:** 16/32/64/128-bit loads × 1/2/4 independent loads per thread × 66/132/264 CTAs on 52--61 MB weights. Question: what request concurrency reaches the H200 plateau without spills?
2. **Cache policy:** ordinary global, L2::128B, L2::256B, nc.L1::no_allocate, and evict-first on cold 52 MB and warm 4/16/32 MB rereads. Question: do hints help the actual cold-L2 GEMV or only a warm microbenchmark?
3. **Skinny GEMV path:** one row per CTA, multiple rows per CTA, and a persistent grid for m=1,16,32,256. Question: is the SGLang gap launch/tail, HBM latency, or arithmetic/address issue?
4. **Router conversion:** FP32 accumulation with original FP32 weights versus the allowed BF16 hi+lo split, and packed BF16/FP8 conversion only where legal. Question: can conversion and router error stay below the 0.01% router bound while reducing instruction count?

### muse_moe_w4a16 (decode and grouped NVFP4)

1. **Decoder cost:** LOP3/shift, PRMT/LUT, and the current Marlin decode for 4K and 28 MB expert tiles. Question: decoded weights per integer issue cycle and spill count?
2. **WGMMA matrix shape:** mma.sync, WGMMA-SS, and WGMMA-RS at N=8/16/32/64/128, with identical BF16 accumulation. Question: where does RS overtake SS and does it match the routed expert shapes?
3. **Pipeline:** global/shared stages versus TMA producer/consumer with depth 2/3/4 and setmaxnreg producer/consumer budgets. Question: does TMA hide weight latency or add barrier/descriptor overhead?
4. **Routed layout:** contiguous expert tiles versus the actual skewed expert order, with 64/128/256-byte load sectors and optional L2 prefetch. Question: is the busiest-GPU routing pattern causing partition or sector waste?

### muse_attention_decode (paged FP8 KV)

1. **Row transaction width:** scalar/32/64/128/256-bit loads, page size 1 and 64, contexts 2K/16K/80K/1M. Question: which width and page mode minimizes sectors per K/V row?
2. **Physical locality:** contiguous, stride-2/4, and randomized row addresses with the same logical cache contents. Question: does a page-1 gather expose TLB or L2 near/far behavior?
3. **FP8 path:** packed cvt.rn.f16x2.e4m3x2 query rounding versus the existing conversion, with conversion hoisted beside Q/K loads. Question: is conversion visible after KV traffic is saturated?
4. **Attention scheduling:** one CTA per request, multiple requests per CTA, and a persistent query tile; compare sliding/full and q_len 1/8. Question: is the short sliding case dominated by tails/fixed work while full attention is bandwidth/softmax-bound?

### muse_decode_layer (whole decode layer)

1. **Fusion boundary:** current graph, fused dense+norm/router stages, and one persistent layer kernel. Question: how much of the measured 169--614 us is graph-node/tail/glue time?
2. **PDL:** encode a programmatic dependency between dense output and the next prefetch/setup; compare with a normal graph edge and verify griddepcontrol.wait. Question: is there observable overlap, and is it safe under graph replay?
3. **Cross-stage cache policy:** mark only reused expert/KV tiles evict-last; mark one-pass dense weights no-allocate. Question: does a fused layer retain useful tiles between stages without harming cold-L2 reads?
4. **Clock/resource log:** capture SM/memory clocks, power, registers, spills, SASS bytes, and barrier waits for batch 1/16/32, sliding/full, q_len 1/8. Question: did a nominal instruction gain cause a clock drop or occupancy loss?

### muse_prefill_layer (large grouped work)

1. **Expert GEMM:** compare the same RS/SS/WGMMA and TMA pipeline variants at q_len 753 and 16,384. Question: does the large-N regime reach the H800 paper's N>=64 behavior on H200?
2. **TMA depth and tile size:** 64/128/256-byte sectors, 2/3/4 stages, and tile shapes that keep the expert's K/V or weight reuse in shared memory. Question: where does barrier overhead stop being hidden?
3. **HBM versus reuse:** cold first read, 4/16/32 MB weight reread, and a full expert working set. Question: is the prefill bound HBM bandwidth, L2 capacity/partitioning, or tensor throughput?
4. **Instruction footprint:** compact loop versus shape-specialized unrolling; inspect SASS size and instruction-cache counters if available. Question: does a larger unroll improve overlap or evict the instruction stream?

The expected outcome of these probes is a ranked set of H200 measurements, not a promise that every ISA feature belongs in the final kernel. A lever should be kept only when it survives the cold-L2 graph replay, correctness tolerance, nearby-shape tests, and the judge's reported clock/power envelope.

