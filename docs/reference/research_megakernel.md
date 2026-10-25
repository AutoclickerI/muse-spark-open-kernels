Read when deciding whether to fuse Muse decode stages into a persistent H200 kernel, selecting donor mechanisms, or interpreting the failed dense-chain experiment.

# Decode megakernels: evidence and a Muse H200 design

## Decision

A competitive decode megakernel needs **competitive inner loops, tile-level readiness, and operand storage whose lifetime crosses operation boundaries**. One launch and software grid barriers do not supply these properties. Our failed dense chain replaced fast projections with a poorly matched, low-occupancy register-MMA implementation; it did not test the stronger design.

For S11, the explicit fusion-only scenario below saves about **5–8 us per layer** at decode 1/16/32 and verify 1, for **+2.80% on the six-load score**, from 1.70677 to approximately **1.75451x**. This is a conditional engineering budget, **not a performance prediction or H200 result**. Inner-loop parity is unproved and could erase it. The measured evidence does not justify importing Hazy's 1.5x whole-model gain into this layer, or calling the approximately 48 us decode-1 streaming bound a megakernel forecast.

The most useful new donors are Flare/b12x's exact scale codecs and scale-staging choices, Hazy/Forge/MPK's cross-instruction operand lifetime, and Event Tensor's distinction between static and dynamic scheduling. There is also strong negative evidence: our newer NT1 H200 experiments remove large waits yet lose overall. The next experiment must identify a net critical-path reduction while preserving tile throughput.

## Evidence contract and scope

**Measured external** means the authors' result, not independently reproduced here. **Measured local** means an existing H200 receipt. **Static** means source inspection or compiler-resource evidence. **Modeled** means arithmetic under stated assumptions. Missing batch-specific results are marked unavailable; a paper's best result is not silently assigned to batch 16 or 32.

Primary sources read: [ForgeMegakernel v1][forge], [AutoMegaKernel v1][amk], [Ada-MK v1][ada], [Fleet v1][fleet], [Event Tensor v2][event], [MPK v2][mpk], [FlashFormer v2][ff], the official [Hazy low-latency post][hazy], [tensor-parallel follow-up][hazy-tp], [Fluffy Kittens follow-up][fluffy], [Retire the Abstractions][retire], [Cohere's implementation report][cohere], and [AMD's MI355X report][amd]. Hazy's [ThunderMLA][mla] is included as the attention-interpreter precursor. No performance claim below relies on recollection or a secondary summary.

Flare code is pinned to `NVIDIA/flare-local-inference` tree **bce032c014bae34502ba22d3304fa084d5b40afe**. Its [`pyproject.toml`][flare-pin] pins the kernel dependency to `local-inference-lab/flashinfer` **259648d983fdd1488bb6b8df8e5ea6134901a4d4**; b12x citations refer to that revision. Hazy code is pinned to **7309cec801537b61fea3b50d7dfe454a6cde578e**. Dates in the question identify the requested releases; code observations describe these inspected revisions, not an asserted reconstruction of release-day contents.

Downloads are isolated by source under the task's specified `scratchpad/research/` directory. Only data/text inspection was performed; no downloaded code was executed, built, installed, or imported. No GPU work, okc commands, Docker operations, or process management were performed. `sync_upstream.sh` was inspected but not run because it invokes `okc init`, conflicting with this task's explicit prohibition. The local authoritative task and constraint files were read. This note is the only intended repository change.

## What the literature actually measures

“Per-kernel launch” has several baselines: eager host dispatch, CUDA-graph operator execution, and a single persistent launch containing stage-wide barriers. They answer different questions. Unless identified as a controlled ablation, a serving-system ratio includes differences in operator implementations and runtime work.

| Source | Hardware, workload and measured gain | Coverage within batch 1–32; limitations |
|---|---|---|
| **ForgeMegakernel** | One H100 80GB HBM3, BF16 weights/KV. Fourteen cells, 0.6B–13B, contexts 96–4096, B1–16: **1.10–1.61x over graph-enabled SGLang 0.5.18**, geomean **1.21x**; **1.54x** geomean over MPK. Llama-3.2-1B B1/context96: **1.37x SGLang**, **1.50x vLLM**, 74.0% MBU. [§5.1–5.2][forge] | B16 is tested; no B32. Serving, separately: Qwen3-0.6B B16/context1024 **1709/1457 = 1.173x**, context4096 **1688/1322 = 1.277x**; B8/context2048 **1576/1406 = 1.121x**. These Table 2 numbers are serving TPOT, not isolated-step Figure 3 gains. Serving geomean is 1.11x. [Table 2, §5.3][forge] |
| **AutoMegaKernel** | B1, position zero/empty KV. On H100 an earlier SmolLM2-135M FP32 path takes **14.30 ms vs 8.65 ms** graph-enabled vLLM: speedup **0.605x**, or 1.65x slower. Against eager vLLM it is 1.06x slower. Tuned W8A16 vs BF16 graph-cuBLAS wins on L4 **1.18–1.33x**, L40S **1.25–1.27x**, A10G up to **1.08x**, RTX 5090 Laptop **1.19–1.23x**; it loses on A100/H100. [Tables 10–11, §6][amk] | No B2–32 performance evidence. Quantization comparisons change precision and do not isolate fusion. H100 W8A16 ratios are about **0.72 down to 0.60x** as model size grows; datacenter whole-decode rows predate its newer GEMV. Equal-precision RTX BF16 remains below graph-cuBLAS. [§5–6][amk] |
| **Ada-MK** | NVIDIA **L20**, GPTQ W4A16 Qwen3-1.7B/Qwen2.5-1.5B. Input64/output12: Qwen3 B1 **1.236x vanilla TensorRT-LLM**; B1/B2 **1.502/1.645x vLLM**, **1.719/1.493x SGLang**. Qwen2.5 improvement over TRT is **1.067–1.156x** across tested batches. [§5.2–5.5][ada] | B1,2,4,8,16; no B32. On HumanEval B16: **1.195x TRT**, **1.016x vLLM**, **1.035x SGLang**. On CSL B16 it trails vLLM by 3.5%. These are complete offline generation throughput measurements including the engine's prefill path, not isolated decode launch ablations. The introduction's “10–50% latency” description must not replace the tables' throughput definitions. [§5][ada] |
| **Fleet** | **MI350X**, dense BF16 Qwen3-8B; 64 input, 1024 output. B1 vLLM eager **10.51 ms**, chiplet-unaware MPK **7.83**, Fleet M-tile/M-split **6.82/6.73**: **1.54/1.56x vLLM**, **1.15/1.16x MPK**. B32 M-tile **12.35 ms**, M-split **13.37**, MPK **15.62**: **1.27x MPK**, and authors report **1.06x vLLM**. [§6.1–6.2][fleet] | B1–16 improvement over MPK is **1.13–1.16x**, attributed to fewer dispatches, without cooperative M-tile weight reuse. B32 activates reuse. vLLM was run with `--enforce-eager`; the B1 1.56x is not a CUDA-graph baseline result. B64 loses to vLLM. [§6.1–6.4][fleet] |
| **Event Tensor / ETC** | **B200**. Qwen3-30B-A3B B1, prompt512/output100: **1.48x graph vLLM**, **1.20x graph SGLang**. Dense Qwen3-32B B1: up to **1.15x vLLM**. Complete MoE layer: up to **1.23x** best baseline at **1024 tokens**, outside our decode batch range. [§4.2–4.3][event] | Serving sweep B1–128; prose does not give exact B16/32 serving ratios. Controlled Qwen3-32B **TP4 on four B200s**, same task code, against a barrier-serialized persistent kernel: static scheduling **1.09/1.06/1.07x at B1/16/32**; dynamic **0.83/0.82/0.85x**. Thus the 6–9% result isolates fine-grained scheduling, but does not compare one launch with many. [Table 3, §4.5][event] |
| **MPK / Mirage** | **A100, H100, B200**, BF16, five models including Qwen3-30B-A3B. B1 end-to-end serving **1.0–1.7x** over the best vLLM/SGLang baseline. Qwen3-8B A100 example **14.5 → 12.5 ms = 1.16x**. Qwen3-1.7B TP8 on eight H100s: **1.1–1.4x** vLLM/SGLang. [§6.2–6.5][mpk] | Maximum batches 1,2,4,8,16; no B32 in that sweep. The 1.0–1.7x range spans GPUs and models, not each H100 cell. Includes GPU-resident request admission and page management, benefits absent from our already supplied layer inputs. [§6.1–6.3, Appendix A][mpk] |
| **Hazy “No Bubbles”** | Llama-3.2-1B BF16, **B1**, prompt32/output128, no speculation. **H100: >1.5x SGLang, almost 2.5x vLLM**, 78% bandwidth; **B200: >1.5x SGLang, >3.5x vLLM**. Baselines use graphs and compilation. [official post][hazy] | No B16/32 result. Its B200 accounting still spends 250/600 us on activation handoffs/consistency and 40 us on warp sync. The 5 us-per-boundary example combines tails, launch, and memory latency; it is not a transferable launch constant. [runtime breakdown][hazy] |
| **FlashFormer** | One **H100 SXM**, Llama-3.1-8B BF16, B1, contexts128–6144: **184/182/180/176 tok/s**, vs GPTFast **168/159/158/143**, and vLLM **146/144/141/138**. Derived **1.095–1.231x GPTFast**, **1.260–1.277x vLLM** for the BF16 rows. Both graph-enabled. [Table 6][ff] | B2/B4 also measured; no B16/32. Table 7 ratios vs GPTFast: B2/context128 **328/308=1.065x**, B4/context128 **616/608=1.013x**; context1024 **648/556=1.165x**, **600/504=1.190x**. The abstract/prose's up-to-61% vLLM result includes INT4, not the BF16 table alone. [§5.1–5.2][ff] |
| **AMD MI355X blog** | One **MI355X gfx950**. Qwen3.8-Flash-Next, same MXFP4 weights and eight-bit activations on both paths: **3.4–3.5x** regular ATOM without speculation on real prompts; **3.1–3.3x** with MTP; **3.1x per user at B4**. Qwen3.8-27B's complete speculative setup is **2.0–2.1x** regular ATOM. GPT-OSS-120B **3.6x** stock vLLM without speculation, **1.9x** comparing fastest speculative setups. [results/configuration][amd] | No B16/32 evidence. Qwen3.8-27B changes LM-head/drafter precision; GPT-OSS changes precision, drafter and GPU partition mode. Even the closer-precision controls are not pure scheduling ablations. Qwen-Flash's sparse top-512 attention changes the workload relative to Muse's mandatory every-key attention. [test configuration][amd] |
| **Flare / b12x** | SM120/121 local inference; Python preparation, Rust graph replay, native W4A16/MoE and CSF paths. **No controlled B1–32 decode-megakernel-over-per-op gain established by the inspected code/docs.** [README][flare-readme], [core design][flare-design] | A research prototype and kernel donor, not evidence for a particular H200 speedup. Its 6.9 GB/token / 1.79 TB/s ≈3.9 ms example is a bandwidth bound; “native runtime within 5% of replay” is a qualification requirement, not a measured fusion result. [design, serving-performance section][flare-design] |

FlashFormer's often-cited **62 → 48 us/layer** is the **8K-wide, 32-layer deep-linear microbenchmark**: Table 4 gives 1.998 ms separate, 1.543 ms fused, 1.513 ms fused+overlap, i.e. **62.44 → 48.22 → 47.28 us/layer**. Fusion gives 1.295x; overlap adds only 1.020x beyond fusion. It is not an entire transformer layer with attention, norms and MoE. [§3, Table 4][ff]

### Hazy follow-ups and the most relevant MoE implementation

The [September 2025 Llama-70B TP follow-up][hazy-tp] uses eight H100s and reports **22% higher end-to-end throughput** on 65,536 ShareGPT requests (output 9043 vs 7387 tok/s). Its target is throughput, with examples at batch8192, not B1–32 latency. A detailed instruction transition shows the compute gap shrinking **10.2 → 3.4 us** with shared-page pipelining; whole-workload pipeline benefits are about **2–6%**. A global work queue helps large batches but costs more than it saves at very small batches. Tokasaurus manages KV pages. These results support the lifetime mechanism, not a 22% Muse claim.

[Fluffy Kittens, November 2025][fluffy] demonstrates fused collectives/GEMMs and expert dispatch+GEMM on H100/B200. Its expert example is top8/256 experts, hidden7168/intermediate2048, eight H100s. Crucially, **NVFP4 support and multi-node MoE megakernels are listed as future work in that post**; they are not demonstrated decode B1–32 gains. [Retire the Abstractions, August 2026][retire] says an MoE megakernel was built with agents without the prior abstraction, but supplies no controlled GPU/batch performance table. Neither statement supplies an NVFP4 W4A16 H200 result for Muse.

[ThunderMLA][mla] is an attention-only precursor, reporting **20–35% over FlashMLA** for DeepSeek MLA. It provides an on-device instruction stream and motivates removing attention/combine boundaries. MLA dimensions and cache representation differ from our GQA d64/page1 problem; no whole-layer score is assigned to it.

[Cohere's North Mini Code megakernel][cohere] is useful additional primary evidence: one H100, BF16 30B-A3B, 128 experts/top8, context8K, **1.58x vLLM at B1**, **1.32x at B8 with real routing**, **1.14x at B8 with uniform routing**; maximum released batch8. Its overall serving gains are **1.25–1.41x**. It explicitly recommends matching cuBLAS/FA3 standalone before integration. Attention and MoE run as parallel branches; **Muse's router consumes the post-attention residual, so this branch overlap does not transfer**. A sparse active-expert set and local dynamic queues do transfer structurally.

## Mechanisms: what disappears and what remains

| Work | Scheduling and dependency mechanism | Operand prefetch / HBM continuity | KV and MoE; SM90 transfer |
|---|---|---|---|
| Forge | Per-SM typed instruction streams; producer counters replace intermediate grid-wide barriers. Milestones explicitly separate single launch, streams, counters, and buffer pooling. Removes launch/ramp/drain and unnecessary all-SM waits. | Shared buffer pool follows operand lifetimes, allowing instruction k+1 loads during k stores. Streams alone hide no latency. | Dense-model study with BF16 KV, not a demonstrated NVFP4 MoE runtime. Scheduling transfers to H200; numeric operators must use Muse formats. [§4.1][forge] |
| AMK | Static task DAG, monotonic uint32 counters, release/acquire handoffs; cooperative entry/exit grid sync. Validator rejects cycles, unsatisfiable waits, wrong per-SM order and buffer read races. A shared multi-producer count must mean an all-join, not an arbitrary first-k subset. | Scheduler optionally hints next GEMV weights into **L2**, not retained operand pages. Depth2 ablation was **0.98x** on RTX 5090; later int8 staging-ring tests also lost. | No demonstrated dynamic MoE/sliding attention; position-zero measurements cannot establish KV streaming. Validator ideas transfer; replacing BF16 weights by int8/int4 is forbidden here. [Algorithms 1, §4–6][amk] |
| Ada-MK | Offline fine-grained data-movement/compute DAG search, serialized trace, warp roles. Removes runtime page-state decision instructions and false resource dependencies; paper describes semaphore/role synchronization but does not expose a fully auditable cross-SM counter implementation. | Split K to halve per-stage smem, reuse activation pages after register reads and reuse them for weights/output. Lookahead N+2 and role balancing expose overlap. Padding-aware vector loads remove invalid I/O. | TRT retains peripheral KV management; dense GPTQ models, no demonstrated routed MoE. Static lifetime/padding ideas transfer; INT4 decoder and Ada-specific pipeline are not NVFP4 Hopper implementations. [§3–4][ada] |
| Fleet | CU/chiplet/device tasks; one scheduler CU per XCD, local counters, last local worker publishes a cross-XCD event. Fewer tiny dispatches and fewer cross-chiplet flush/fence operations. | Cooperative M-major traversal lets concurrent token tiles share weights in local L2. Activation fusion removes buffer read/write traffic. Does not demonstrate Hazy-style next-op shared operand paging as its main gain. | Dense Qwen study; no demonstrated MoE result. H200 has one coherent L2 rather than eight private XCD L2s. Reuse order and event coalescing transfer; XCD placement, AMD cache flags and relaxed-scope protocol do not. [§3–6][fleet] |
| Event Tensor | Integer event tensors; notify decrements, wait observes zero. Static per-SM queues or a global ready push/pop queue. Dynamic route-dependent event counts/consumer ranges handle MoE. Removes false operator-wide joins and wave tails; dynamic queue adds atomics/contention. | Compiler prefetch pass starts weights before input activations arrive. No evidence that every such prefetch is free, or a measured per-boundary smem budget for Muse. | Full attention/RoPE/KV/norm/MoE pipeline; dynamic expert dispatch. Core event ideas transfer to SM90, B200 task code must be ported. The tiny-M dynamic-MoE ablation loses: **0.95x at one token**, static **1.03x**. [§2–4, Table 2][event] |
| MPK | SM-task graph; event-triggered workers and decentralized scheduler warps. AOT tasks wait locally; JIT tasks enter queues after readiness. Hybrid choice reduces both imbalance and worker→scheduler→worker latency. | Shared pages are acquired/released at task granularity; start next preload after current task has issued its data transfers and freed pages. Also prefetch 352-byte task descriptors. Cross-task pipelining gives **1.2–1.3x** on Qwen3-8B's final linear layer on **B200**, a task-level ablation rather than a whole-model gain. | GPU page allocation/admission plus paged attention; route metadata and fused token-gather/GEMM. H200 supports mechanisms. Four reserved scheduler SMs cost **3.03% of 132 SMs** and are not automatically worth spending for one small layer. [§5–6, Figure 12][mpk] |
| Hazy | Host-built per-SM instruction streams, global dependency counters, per-chunk consumers. Down projection waits on intermediate chunks, not the whole MLP. | H100 uses **13 × 16 KiB** shared pages plus metadata. Load, compute, store can overlap across instructions; still pays global activation consistency and counter costs. | Original dense Llama and attention instructions, not routed MoE. H100 device loops transfer architecturally; BF16 KV and model semantics need adaptation. [post][hazy]; [`page_allocator_loop`, `include/controller/page_allocator.cuh`][hazy-pages] |
| FlashFormer | Compile-time Cheetah partitioning; atomic-based **global synchronization** plus per-thread fences. Three attention global joins remain. Demonstrates that some global joins can coexist with useful weight overlap. | Fixed-byte matrix chunks flow through a **unified shared-memory pipeline across operations**, including future weights while current consumers drain. No inter-op grid barrier forces loaders to stop. | Per-KV-head KV splitting and in-kernel softmax combine; current-token K/V loaded separately to avoid stale speculative preloads. Fuses up/gate/SiLU with matching down chunks; no routed-MoE demonstration. Hopper-native BF16 design ideas transfer; GPTQ INT4 representation does not. [§4][ff] |
| AMD MI355X | Resident CU workers, data+sequence tag published together in one 64-bit word; consumers poll only needed values. Per-chiplet release words localize polling. Whole-forward or one-kernel-per-layer, depending on model. | Idle CUs preload future weights. Partition mode places weights near their consumer die. Removes broad waits and remote-HBM traffic, not merely launches. | Real MoE/speculation/state management; Qwen-Flash's top-512 sparse attention is inadmissible for Muse. H200 has no MI355X die-affinity lever; tag/data publication concept transfers only with correct CUDA atomic ordering. MXFP8 activations and converting weights to MXFP4 are forbidden. [mechanisms/configuration][amd] |
| Cohere | Static waves and per-SM descriptors with dynamic attention/expert drains, global counters, common 12-warp ABI. Named worker barrier excludes independently running controller. | Weight stage ring is kept alive across same-op tiles; load weights before waiting for activations. Fixed shared layouts won over general paging in that implementation. | Continuous batching/paged attention; both expert GEMMs plus SiLU and reductions. BF16 H100 is close hardware evidence; no NVFP4 decoder. Its 384-thread ABI is a donor design, not a free replacement for our tuned 256-thread NT1. [implementation sections][cohere] |

All these methods pay real producer→consumer latency. **Counter != data**: observing a counter is safe only after producer stores are published and acquired at the correct scope. TMA completion, async-proxy ordering, shared-page ownership and global visibility are separate conditions. [AMK §4][amk], [Cohere barrier listing][cohere], [CUDA PTX memory model][ptx]

“Keep HBM busy” means use otherwise idle HBM slots. If the predecessor already saturates HBM, successor reads displace it. A megakernel can shift compulsory reads earlier; it cannot overlap two independent 4.58 TB/s streams on the same H200 or remove NVFP4 dequant instructions by rescheduling them.

## Flare and b12x: precise donor audit

### Preparation, execution and fusion boundary

[`README.md`][flare-readme] and [`docs/core-design.md`][flare-design] distinguish Ignition's Python preparation/Inductor compilation/capture from Helios's Rust execution of prepared graphs. Opaque b12x calls remain native operations; pointwise PyTorch work is visible to Inductor fusion. Captured auxiliary-stream forks overlap independent projections, shared/routed experts, and sparse-attention selection; branches use disjoint scratch and rejoin before consumers. This is **graph-level overlap**, not proof that Rust executes a decoder in one persistent GPU kernel. Python-vs-Rust host dispatch supplies no modeled gain to our already captured layer.

[`ignition/components/moe.py::PreparedRouter`, `PreparedRoutedExperts`][flare-moe] plans top-k routing, scale expansion and expert execution with caller-owned scratch. `PreparedRoutedExperts` can integrate router inputs and can choose inline CSF reconstruction or a separate expansion operation. [`ignition/components/linear.py::UnquantizedProjection`, `QuantizedProjection`, `ParallelProjections`][flare-linear] delegates projections and captured independent-stream work to b12x. **No new H200 timing is inferred from these wrappers.**

FP8 E4M3 KV is explicit in [`ignition/recipes/qwen38/state.py::HybridStateLayout.build`][flare-state] (`kv_dtype=torch.float8_e4m3fn`, main page64), and attention uses a separate paged KV writer in [`attention.py`, lines 275–284][flare-attn]. This recipe uses QSA and head dimensions256. Its sparse selection, page64 and model-specific recurrence are not implementations of our full/page1/d64 GQA attention. The reusable idea is **page-index and KV staging**, not its skipped-key semantics.

### CSF is an encoding of scale bytes, not a new arithmetic format

[`_lib/quant/csf_encode.py::encode_csf_source`, `_choose_bases`, `_pack_rows`][csf-encode] operates on native scale bytes without floating-point conversion. A row chooses a base covering the largest number of scale bytes in a **16-value window for NVFP4 (4-bit offsets)** or **two-value window for MXFP4 (one-bit offsets)**; outliers are stored verbatim as exceptions. It is lossless, but exceptions/indexes can make it larger than the input. One-bit MXFP4 scale coding must not be advertised as NVFP4's compression ratio.

[`_lib/quant/csf_blocks.py::BlockStorageLayout`, `upload_csf_block_payload`][csf-blocks] stores bases, codes, page/group offsets and exception records. [`csf_inline.py::CsfInlineReader.stage_compressed`, `expand_compressed`, `expand_warp_rows`, `prefetch_column_bounds`, `prefetch_row_bases`][csf-inline] stages compact records, reconstructs native scale words in shared memory, and retains row bases/bounds across a K sweep. These are useful instruction/address-locality ideas even if the external encoding loses on our weight distribution.

The [fused-MoE README][b12x-readme] explains the trade-off: expansion reconstructs each selected expert once per call, while inline reconstruction repeats for each consuming compute tile. Inline can save traffic at moderate routing density; few experts or many token tiles can favor an expanded scratch buffer. Complete-call tuning includes expansion, routing, resets and output reduction. **Expansion must occur in the timed call** under our rules. Prepare-time scratch allocation and lossless code construction are allowed; persistent widened numeric scales/weights are not.

Our S11 NT1 bank already contains **32 KiB FP4 payload + 4 KiB scale indices per 36 KiB tile**, and a recoded scale bank; see [`dev/nt1_timeline/README.md`](../../problems/muse_decode_layer/dev/nt1_timeline/README.md) and [`_encode_scales`, `_repack_recoded`](../../problems/muse_decode_layer/cand_r3_best7_s11_final.py:1498). Thus an ideal halving of scale bytes saves **1/18 = 5.56%** of original expert payload, before CSF bases/exceptions/metadata. It does not halve MoE weights or halve the layer. Using the S11 byte attribution gives at most **0.86/2.59/3.09/2.58/6.33/7.33 us per layer averaged S/F** at D1/D16/D32/V1/V16/V32, and **+1.52% score** if all saved HBM time is exposed and decoding is free. This is a deliberately optimistic **byte ceiling**, not a codec forecast. See the exact S/F calculation below.

**Precision trap:** [`docs/design_docs/b12x_nvfp4_packed_scale_contract.md`][b12x-scale-contract] says the packed W4A16 preparation normalizes scales and **discards nonzero normalized E4M3 values below 1/64**. CSF faithfully preserves those already packed bytes; it cannot restore discarded subnormals. The donor's audited GLM checkpoint happens to retain all scales, but that says nothing about Muse or fresh held-out weights. Require exhaustive byte reconstruction over every finite source scale/global-scale contract and actual Muse weights. A codec preserving raw scale bytes is admissible; importing this narrowing unchanged is not.

### GEMV, dequantization and persistent MoE

* [`gemm/bf16_gemv/_kernel.py::SmallNGemvKernel`, `MixedVectorGemvKernel`][b12x-gemv] implement vector/coalesced paths; [`_tensor_core.py::TensorCoreGemvKernel.kernel`][b12x-tcgemv] uses small token tiles, clamped padded loads, register fragments and BF16 `m16n8k16` MMA. Hopper supports this arithmetic, but the wrapper/runtime is not a drop-in SM90 backend or evidence that it beats our balanced split-K dispatch.
* [`moe/_shared/kernels/w4a16/kernel/gemm.py::W4A16GemmKernel._run_mma_pipeline`, `_dequant_and_accumulate_bundle`, `_scaled_dequant_b_fragment`][b12x-gemm] pipelines `cp.async`, packed scales, register dequant and BF16/FP16 MMA. Its packed [`packed_dequant_e2m1x4_to_bfloat2x2`][b12x-packed] uses bit operations to construct BF16 fragments and scale multiplication. This **software packed path** can be ported to Hopper. Other native conversion/FP4 and activation-quantized b12x paths depend on newer architecture/PTX features and cannot be presumed to run on SM90.
* [`kernel/direct.py::MoEMicroKernelW4A16SmallMDirect`][b12x-direct] delegates small-M native-ModelOpt-layout execution to the micro backend, with BF16 activation mode. It is a candidate for studying slot routing/padding, not proof of faster Muse NVFP4 GEMV.
* [`kernel/fused.py::W4A16FusedMoeKernel._moe_body`, `_grid_barrier`][b12x-fused] contains a **persistent FC1→activation→FC2 kernel with software grid barriers**. Its shared pipeline is not evidence of barrier-free overlap across expert stages. Porting that barrier structure alone would repeat part of 87dc's problem. The scale mode is a selectable pipeline property in [`W4A16GemmKernel`][b12x-gemm].
* The [MoE README][b12x-readme] explicitly separates decode/verify A16 from optional prefill A4. A4 activation quantization, native W4A4/W4A8 paths, MXFP8 dense projections, and any widened prepare-time weight representation earn **zero Muse gain** because they violate our format contract. Different model/dense-branch semantics also earn zero.

No inspected primary benchmark establishes b12x's per-technique latency gain on H200 for our shapes. Our current one-LOP3 decoder already improved the predecessor; repeating bit packing, slot routing or scale-bank recoding is not an additional improvement. A donor must beat that current instruction sequence and whole-call timing.

Static decoder comparison: b12x's [packed E2M1→BF16 helper][b12x-packed] expresses nine bit operations per four values before scale multiplication and output moves. This is source-level arithmetic, not a SASS count: ptxas may fuse operations, and the fragment layouts differ. Our [lane note](../../problems/muse_decode_layer/dev/nt1_timeline/README.md) and [joint-decoder counting script](../../problems/muse_decode_layer/dev/nt1_timeline/table_kill/count.py) report about 26 instructions per 16 weights for S11's LOP3 sequence, excluding MMA. There is no demonstrated instruction-count advantage for replacing that sequence with the donor helper.

## Reconstructing 87dc944b

### What was actually timed

The [H200 receipt `87dc944bfd7944c7`](../../runs/muse_decode_layer/20261024T144347Z_check_87dc944bfd7944c7.json) names an embedded best7-reset source with SHA256 **b4cc76dc5302c4572951cf56163cb1bd1c8f1db5d3c133c667fbbe208ba6c98a**. The hash matches the present [`cand_r3_best7_reset.py`](../../problems/muse_decode_layer/cand_r3_best7_reset.py). Its `DENSE_CHAIN_CUDA` literal has SHA256 **81d1859f3608a08e1b2c81aed6f34a21fac0f8c2deb6255c481737d8a0b253b3**. These identities are also recorded in [`prior_sm90a.json`](../../problems/muse_decode_layer/dev/megakernel/prior_sm90a.json).

The immutable [submitted lab wrapper](../../runs/muse_decode_layer/20261024T144347Z_check_87dc944bfd7944c7.py), payload functions `_segment`, `_state`, `_paired`, compares **O→boundary1→pool/router**, not the full megakernel, on real dense weights with alternating cold-read graph pairs. The reference is best7's selected projection family, with its library plans initialized in `_state`; it is not S11. The segment errors are around 4–8e-5 and full-layer route flips are zero in the two changed-input checks. These checks establish the tested math, not a broad held-out correctness guarantee.

| M | best7 segment us | persistent segment us | Paired slowdown | Full-layer evidence |
|---:|---:|---:|---:|---|
| 1 | 38.29 | 57.67 | 1.5068x | 82.74 → 104.00 us, +21.26 us |
| 8 | 39.17 | 62.68 | 1.6014x | 158.17 → 181.74 us, +23.57 us |
| 16 | 39.39 | 65.29 | 1.6587x | Segment only |
| 32 | 39.76 | 71.23 | 1.7929x | Segment only |
| 128 | 43.56 | 198.87 | 4.5678x | Segment only |
| 256 | 56.30 | 365.55 | 6.4940x | Segment only |

### Source-level causes, and what cannot be identified

1. **Wrong token tile at tiny M.** [`_dense_chain_call`](../../problems/muse_decode_layer/cand_r3_best7_reset.py:6233) calls `island(... nt=4, grid=SM_count, prefetch=16MiB, prefix=True, pdl=False, route=False)`. In `DENSE_CHAIN_CUDA::dense_tile`, `BT=8*NT`, the accumulator is `[TR][NT][4]`, and padded token addresses clamp to `M-1`; outputs alone are masked. Thus M1/M8 perform four token-fragment groups rather than NT1's one. This multiplies token-side MMA, activation loads and state by four **relative to NT1**, not immutable weight bytes. It is a static explanation, not a measured fourfold H200 time penalty.

2. **Register pipeline and inadequate resident work.** The [SM90a compiler audit](../../problems/muse_decode_layer/dev/megakernel/prior_sm90a.json) reports **248 registers/thread for NT4**, 63,488 registers/256-thread CTA, **one CTA/SM**, only **8 resident warps** with the actual 132-CTA grid. NT1 uses124 registers and allows two CTAs by register count, but the same 132 grid still supplies only one. The dense tile has register double buffering within its K loop, then restarts/drains it for each row task. It is not a cross-task shared operand ring. This weakens latency hiding even though the code uses tensor cores.

3. **Wave tails plus three all-grid joins.** `dense_island` runs O's **512 row16 tasks** over132 CTAs (four trips for 116 CTAs, three for 16); pool/router has **288 tasks**, so only 24 CTAs perform its third trip (**18.18% of SMs**). It calls `grid_barrier` after O, after boundary1 and after pool. At M1/NT4, `C=16/NT=4`: only **four CTAs** compute the norm boundary while 128 wait. These joins serialize low-occupancy phases. Some reduction joins are mathematically required; making all 132 CTAs stop at each phase is not. [`DENSE_CHAIN_CUDA::dense_island`, `grid_barrier`, literal at line6208](../../problems/muse_decode_layer/cand_r3_best7_reset.py:6208)

4. **L2 hints do not retain operands.** `prefetch_weights` issues bulk **L2** hints for O at entry and pool before the first grid barrier. The first prefetch is inside the measured call with PDL disabled, so it has no predecessor overlap to exploit; it adds instructions before O's real loads. The later hint may overlap some tail but still requires L2→register loads and can compete with O/HBM. It has neither Hazy's shared-page handoff nor a bounded next-task operand. The code's activation-load comment mentions L1 reuse, but `ld_act` actually uses **`ld.global.cg`**, bypassing L1; the comment is not evidence of L1 activation reuse. [`dense_tile`, `prefetch_weights`][chain-source]

5. **Large-M rereads and lost GEMM reuse.** `groups=ceil(M/32)` wraps the complete O and pool task loops. At M128/256 it issues **4/8 logical reads** of each bank and repeats per-row pipeline/reduction work. Some rereads may hit L2, so these are not measured HBM transaction multiples. Nevertheless pool/router is75.50 MB, larger than H200's50 MB L2, and the whole-bank traversal is unfavorable to preserving reuse. A constant barrier cost cannot explain the 4.57→6.49x large-M slowdown. A properly tiled GEMM exploits token reuse without serial full-bank passes.

6. **Fusion removed fewer things than its name implies.** O's FP32 partial buffer, boundary reads/writes, and pool FP32 output are still global. The call separately zeroes counters. It does not fuse attention, routing (`route=False`), experts or unpool. It mostly replaces launch boundaries with software barriers while changing the expensive inner loops. Source inspection therefore does not support “barriers alone caused the entire loss.”

The unchanged-body [sm120 local experiment](../../problems/muse_decode_layer/dev/megakernel/prior_local.json) supports caution: at M1, NT4/prefetch16MiB is88.83 us vs NT1/no-prefetch80.47, but at M32 NT1/no-prefetch267.44 vs NT4/no-prefetch98.60 because NT1 now rereads the weights four times. Fixing `nt=4` globally to `nt=1` is not a solution. These are Blackwell observations with a132-CTA grid, **not Hopper timing decomposition**.

**Unidentified:** 87dc has no measured per-task H200 wait/load/math/store breakdown. Its optional `stamps` are not populated by `_dense_chain_call`. Existing local stamps are taken after barriers and include straggler waiting. The source and receipt explain the structural failure and scaling; they do not identify how many of the19.38 us M1 segment penalty belong to each cause. The prompt's “mechanism unknown” is superseded by this source audit, while that quantitative attribution remains unknown.

### Newer negative evidence constrains a replacement

In the [H200 NT1 receipt `17512d86`](../../runs/muse_moe_w4a16/20261025T045045Z_check_17512d86f3a74f27.json), M32 expert-round scheduling reduces critical W2-ready wait **11.20→2.11 us**, but W13 grows **37.41→41.09**, final barrier **4.34→12.66**, and node **79.95→85.36**. Removing most warp7 tile-barrier time (**11.89→0.91 us**) also fails to improve whole-call timing. The `AGG` results are worse at M8/16/32.

The directly S11-paired [receipt `3af06b40`](../../runs/muse_prefill_layer/20261025T045637Z_submit_3af06b40b5f944a6.json) confirms all these arms lose; interleaved W13/W2 rounds (`R41029`) are **1.102/1.128/1.093x slower** at M8/16/32, zero wins in16 seeds each. [Receipt `4dd00540`](../../runs/muse_moe_w4a16/20261025T050150Z_check_4dd00540cb8c46df.json) also finds independent writer-warp publication slower: **+2.36/+1.01/+1.11/+0.38 us** at M1/8/16/32. These receipts have landed even where lane text still says pending.

The lesson is to optimize the **whole path and work distribution**, not a selected wait counter. Repeating RND/ROT/PIPE/WPUB is not the first experiment recommended below.

## A concrete Muse design for M1/8/16/32

This is a proposed architecture, not implementation work authorized by this research task. One graph-capturable invocation computes one layer; it cannot preload another timed call or keep protected L2 state between replays. All pointer-independent schedules, packed layouts, TMA descriptors and scratch capacity can be prepared within 300 s. Input-derived routes, counters, cache updates and activations are produced in the timed invocation. [TASK](../../problems/muse_decode_layer/TASK.md), [constraints](../CONSTRAINTS.md)

### Stage graph and readiness granularity

| Stage | Work and required precision | Release condition; what can overlap |
|---|---|---|
| 0 | Initialize invocation event bank/queue metadata; input residual is FP32, attention input BF16. | Immutable first weights may load during initialization. No prior invocation's counters can satisfy a new wait. |
| 1 | BF16 QKV+gate projections, FP32 reductions followed by the incumbent BF16 boundaries. Retain S11's separate gate scheduling where useful. | Events per token block, KV group/head and K-split completion. A query/group may proceed before unrelated QKV rows; gate remains a separate prerequisite for gated attention output. |
| 2 | Q/K norm, RoPE, exact FP8 E4M3 KV append, query precision no coarser than E4M3. | Publish both new K and V before any task reads those appended entries. Historical-page index loads can run earlier; no reader races a speculative current-token prefetch. |
| 3 | Sliding/all-context attention: every key, 16 query heads/2 KV heads/d64, page1. Split by `(request, KV group, key range)`, with all 8 verify queries sharing historical KV where legal. | Publish partial `(max, denominator, output)`; combine only its own completed splits. Norm/gate each attention head, BF16 output. O K-slices can proceed for completed input-head ranges; partials reduce only after their contributing slices. |
| 4 | BF16 O projection → branch BF16 rounding → branch RMS → FP32 residual update → hidden RMS/gain/BF16. | Two 8192-wide row reductions are genuine joins. Use per-token reductions; do not wait for all tokens. Other workers fetch assigned pool/router weights during this join. |
| 5 | BF16 pool plus FP32 router implemented by BF16 hi+lo products with FP32 accumulation. | Pool's4096-wide norm and router sigmoid/top8 can run independently after their respective projected rows. Every256 router score participates. Produce pooled BF16 expert input and selected local routes. |
| 6 | Device local-expert counts, token lists and per-expert/token-block tasks. | Release one expert's W13 work only after its participating tokens' normalized inputs and metadata are ready. Keep static placement for tiny M; dynamic work only for actual route/length imbalance. No host readback. |
| 7 | NVFP4 W13 using NT1 decoder/MMA; SiLU×up and the incumbent BF16 intermediate rounding. | Events per expert/token block. First design releases W2 after that block's complete W13, preserving the current full-K inner loop; it does not recreate an all-expert W13 join. |
| 8 | NVFP4 W2, post-expert norm over4096, FP32 weighted expert combination, BF16 y boundary. | W2 weight lookahead begins after route discovery and during W13's actual tail. A token combines after all **its** local expert results/norm statistics complete. No-local tokens explicitly publish zero output/readiness. |
| 9 | BF16 unpooler:1024-row output slice only when the **call's total T≤8**; otherwise all8192 rows. | Preload assigned unpool weights during MoE's final non-HBM work, then wait for combined y. Internal8-token chunks do not authorize the slice shortcut at M16/32. |
| 10 | Post gain/branch RMS, FP32 residual update, hidden RMS/gain/BF16 output. | Final per-token joins, then terminal completion after all outputs/KV stores are visible. FP32 stream remains FP32. |

This stage graph follows the actual [task equation](../../problems/muse_decode_layer/TASK.md) and S11 [`_best7_run` / `_best6_call`](../../problems/muse_decode_layer/cand_r3_best7_s11_final.py). In particular router/MoE cannot run in parallel with Muse attention as Cohere's do. Moving pool/router before the attention residual would change the layer.

For W13→W2, a later design could release K chunks in Hazy's style and accumulate FP32 W2 partials. It changes W2 tiling, intermediate lifetimes and reduction order, so **no saving is credited until its identical-work standalone cell wins**. The landed interleaved-round losses above specifically argue against assuming this is beneficial.

### Per-SM roles and resource envelope

Start from **one resident 256-thread CTA per SM,132 CTAs**, with no dedicated scheduler SMs. Inside each CTA, retain the eight NT1 consumer warps. Thread0 owns task advance, weight TMA issue and event publication at bounded progress points; asynchronous copies continue while the consumer warps compute. This is a deliberately small instruction loop with per-opcode device functions, not a forced port to Cohere's384-thread shape. A graph task is not necessarily one full GEMV row; dense operators keep their independently selected split-K/token/output tiles.

Per-SM roles are **task roles**, not permanent SM partitions: the same CTA executes dense tiles, attention slices, norm reducers or expert tiles as its valid schedule permits. Norm tasks use only their assigned token/reduction workers; idle CTAs may fetch a bounded successor operand or claim other ready expert/attention tiles. A blocked current task must never pin the only worker capable of producing its dependency. At M16/32, use weight-major ordering among token blocks to retain short-term L2 reuse; do not replace current projection tiles with a serial full-bank sweep per8 tokens.

This choice is a major risk: S11's NT1 uses a264-CTA geometry and can exploit two resident CTAs/SM, whereas the proposed resident worker has one. Merely raising shared memory to hold lookahead can sacrifice more intra-op latency hiding than fusion saves. **Port parity is a measured prerequisite, not assumed hardware equivalence.** Alternative thread geometries must earn a separate result before replacing this design.

Proposed **212 KiB dynamic shared-memory cap per CTA**, within Hopper's227 KiB per-block opt-in limit ([Hopper tuning guide][hopper]):

| Shared allocation | Budget | Lifetime |
|---|---:|---|
| Current packed-weight operand ring | 3 ×36 KiB =108 KiB | Three NT1-format slots; dense/attention may use the same slab with their own tile layouts. |
| Next-instruction operand | 36 KiB | One retained successor tile; never overwrite until all readers and async transfers retire. |
| Activation/reduction/attention workspace | 64 KiB | Includes a32 KiB FP32 residual row and16 KiB BF16 row when running a boundary. Attention uses a different view after release; it does not coexist with a live norm row. |
| Descriptor ring, events, alignment and barriers | 4 KiB | Fixed control slab. Global queue/counters are separate prepare-allocated scratch. |
| **Total** | **212 KiB =217,088 bytes** | One CTA/SM; no claim of two-CTA residency with this allocation. |

For attention, three128-key K/V stages for one KV group need48 KiB of the operand slab; query8×8heads×64 BF16 needs8 KiB, leaving workspace for score/softmax/V-transpose and split reduction. Exact FA3-derived device loops must be compiled against the cap; an entry point in another library cannot be called from inside the kernel. The cap is a feasible storage sketch, not proof that the selected compiler layout fits or matches FA3.

Register footprint is the other admission condition:256 threads must fit the SM's64K32-bit register file with no unintended spills; per-op lifetimes must end before dispatch to the next body. Dynamic shared-memory aliasing does not reduce the compiler's maximum live register footprint. The original NT4's248 registers/thread illustrates how close this can get. Do not assume sparse warp roles lower a monolithic kernel's allocation automatically.

### Weight-prefetch schedule and capacity accounting

| During current stage | Bounded future operand | Condition and expected benefit |
|---|---|---|
| QKV's last K loads / epilogue | Historical KV page-index chunks, then selected K/V tiles | Length/page metadata comes from this call. Prefetch immutable history only; new entries wait for append publication. This removes address/load latency, not KV bytes. |
| Attention's final issued KV tiles and softmax/combine | First assigned O tile in the next-operand slot | Only after the CTA's own KV loads are issued. If other CTAs still saturate HBM, this is contention, not free overlap. O compute waits for its attention inputs. |
| O's last weight loads, branch norm and residual norm | First pool/router weight tile | Weight address is static. Pool activations wait for the full row's residual/hidden norm. Reuse the staged operand directly instead of sending an L2 hint and refetching it. |
| Pool projection's last loads and router top8 | More fixed pool work; metadata only for unknown routes | **No expert weights before their IDs are known**. Broadcasting all 32 local experts is expensive speculative traffic, not a modeled saving. |
| Route publication / expert-input RMS | Selected W13 tile on each participating worker | Distributed issue, not one router CTA reading tens of MB before exit. No redundant per-worker full expert-bank prefetch. |
| W13 last loads and epilogue | Same expert/token block's first W2 tile | Route is known; activation readiness is separate. May overlap decoder/math/store latency, but not double HBM bandwidth. |
| W2 norm/combine | Assigned unpooler tile | Particularly interesting for M16/32, where unpool reads67.11 MB. Avoid displacing active expert tiles that another token block will reuse. |
| Final boundary | None from a later invocation | The benchmark supplies a cold independent next call; no cross-call persistence credit. |

One 36 KiB next operand per 132 workers retains **4.866 MB**, only **1.062 us at 4.58 TB/s**. This can hide a first-tile latency/drain, not an entire 75.50 MB pool/router bank (**16.48 us pure stream time**). At one boundary not every SM is free or owns a useful successor tile, so even 1.062 us is a capacity-based ceiling on useful early bytes, not a guaranteed saving. The model credits **zero additional prefetch gain** in its central fusion scenario and evaluates +1 us/layer separately.

This restraint is supported by our [ramp-oracle receipt `cc718d3f`](../../runs/muse_bf16_proj/20261025T030409Z_check_cc718d3fc6da40c9.json): attention→O prefetch changes chains by **+2.12/+2.19/+2.27 us** at sliding B1/16/32; warming O outside the timed interval gives **+0.26/+0.31/+0.03 us**, and B1 full warm O saves only0.70 us. [Register/L2 lookahead receipt `5cefdbc4`](../../runs/muse_bf16_proj/20261025T021917Z_check_5cefdbc428e74ec2.json) also shows broader lookahead hurting some pool chains. **Warming is an oracle diagnostic, not an admissible timed implementation.** These results reject “hide a 2.36 us ramp at every boundary” as a forecast; a shared retained operand is a different hypothesis requiring a new control.

### Synchronization and replay correctness

Use a static feed-forward instruction order for dense phases and a bounded eligible-task list for variable expert/attention work. Global event slots are indexed by invocation epoch, token block, expert/KV group and producer slice. A producer completes all relevant ordinary/TMA stores, executes the required publication fence/proxy ordering, and performs a device-scope **release** count update. Consumers use device-scope **acquire** observations before activation loads. Async shared-copy full/empty barriers use independent slot generations. [PTX ordering rules][ptx]

Counters represent either a named tile or a proven all-join with an exact producer count. `counter>=k` over an unordered producer set cannot certify that a particular K-slice is ready. Static schedule validation must prove the task DAG acyclic, every wait satisfiable, shared-page releases after all readers, and no worker waiting for its own later producer. Dynamic expert tasks are made eligible only after their route metadata is published; no global queue of unready tasks may consume all resident workers. [AMK §4][amk], [Event Tensor §2–3][event]

Initialize invocation scratch in the timed kernel with a prepare-initialized monotonic invocation header and two event banks. Each logical worker keeps its own last-completed epoch in scratch and derives its expected next epoch from that value; simply observing a nonzero global “initialized” flag is unsafe. CTA0 initializes and publishes the next epoch, and workers acquire that exact epoch before using counters. A new bank cannot be reused while any old task or TMA operation still references it. Every invocation, including a no-local-route invocation, finishes terminal bookkeeping and updates each worker's completed epoch. Graph replays of this single scratch instance are stream-ordered; simultaneous independent invocations require disjoint scratch. Test poisoned scratch, epoch wrap handling, changing addresses, alternating none/some/none local routes, and many changed-input graph replays. This stores invocation control state, not cached input-derived outputs.

The8192/4096-wide norms, top8 selection and per-token expert combination remain true joins. They may be token-local or hierarchical reductions, but counters do not permit reading a normalized slice before the norm's scalar is complete. BF16 rounding points and FP32 residual updates remain explicit throughout.

## S11 time model, with all six loads

### Measured anchors

Use S11 final SHA256 **acda6c621b1f3b885f2111ea1b4482ae1d733e2f2493db2b7f0b852ccf66dfb7**, byte-matched to [`cand_r3_best7_s11_final.py`](../../problems/muse_decode_layer/cand_r3_best7_s11_final.py), and [H200 profile receipt `863f502c62e54d7b`](../../runs/muse_decode_layer/20261025T034235Z_check_863f502c62e54d7b.json). [`NEXT_CHECKS.md`](../../problems/muse_decode_layer/dev/NEXT_CHECKS.md) and [its machine-readable accounting](../../problems/muse_decode_layer/dev/levers/current.json) reproduce score **1.706767423x**. Those anchors supersede the reset-profile numbers in the older megakernel scope.

| Load | S11 sliding/full layer us | Projection sum S/F | Attention S/F | MoE S/F | Boundaries S/F | Route S/F |
|---|---:|---:|---:|---:|---:|---:|
| decode1 | 93.30 /102.77 | 48.9 /48.9 | 11.4 /19.8 | 34.1 /33.9 | 5.3 /5.0 | 3.7 /3.6 |
| decode16 | 154.37 /249.61 | 64.6 /109.4 | 13.8 /106.1 | 65.8 /74.1 | 15.9 /31.6 | 3.9 /4.4 |
| decode32 | 197.61 /374.17 | 60.3 /68.2 | 18.3 /196.8 | 103.8 /122.8 | 24.7 /25.6 | 4.2 /4.2 |
| verify1 | 134.91 /151.25 | 43.5 /43.7 | 21.6 /33.0 | 94.7 /73.5 | 8.3 /8.3 | 3.7 /3.7 |
| verify16 | 301.06 /423.50 | 68.7 /67.1 | ≥18.6 /≥168.8 | ≥194.7 /≥206.8 | 11.4 /11.2 | 0 /0 |
| verify32 | 399.30 /640.36 | 90.1 /89.4 | ≥24.0 /≥284.9 | ≥270.4 /≥237.0 | 17.5 /16.6 | 4.2 /3.9 |

Stage rows are profiler durations, **not additive critical-path measurements**. Repeated projection names are proportionally attributed, omitted rows remain unassigned, and PDL intervals overlap. Decode16 full O's58 us is especially unsafe to interpret as58 us of GEMV computation. The model subtracts explicit small deltas from measured whole-layer anchors; it does not sum those durations or reclaim every apparent wait. Existing S11 scale-bank, gate-split, small-M flags, LI_swap and vq8 gains are already included, not added again.

### Floors are feasibility bounds

With `W = sum(projection_bytes) + attention_bytes + moe_bytes` from `current.json`, the measured cold-stream calibration gives an **optimistic fused-stream bound**

```text
t_stream = 2.36 us + W / 4,580,000 bytes/us
```

The calibration comes from [H200 `7b6c96b5`](../../runs/muse_bf16_proj/20261025T022218Z_check_7b6c96b5f97d4c9c.json), not H100 marketing bandwidth. It counts useful logical bytes and inferred active-expert work, not transaction-level traffic, and excludes dequant instruction time, repeated tile reads, most activation/scratch traffic and serial joins. A real design also needs `max(resource compute bound, stream bound)` plus exposed dependency latency.

| Load | One-stream bound S/F, us | S11 S/F, us |
|---|---:|---:|
| decode1 | 48.15 /52.58 | 93.30 /102.77 |
| decode16 | 94.37 /164.06 | 154.37 /249.61 |
| decode32 | 104.39 /246.74 | 197.61 /374.17 |
| verify1 | 79.06 /83.49 | 134.91 /151.25 |
| verify16 | 160.89 /232.43 | 301.06 /423.50 |
| verify32 | 180.87 /322.91 | 399.30 /640.36 |

The large gap includes intrinsic arithmetic/dequant/address/softmax costs and poor inner-loop efficiency. It is not all removable launch slack. Amortizing the 2.36 us intercept once is an idealized limit; the calibrated intercept includes launch/control/tails, not a separable memory-latency token that can be hidden repeatedly.

### Fusion-only design budget

For M≤32 define, in us per layer:

```text
G = 0.40 * min(profile_boundaries, 8.0)
    + min(profile_route, 1.5)
    + (2.0 if M==8 else 0.0)  # verify1 attention glue budget
    + 3.0                    # total launch/tail/dispatch budget across layer
    - tax[M]
tax = {1: 1.5, 8: 2.0, 16: 2.0, 32: 2.5}
t_proposed = t_S11 - G
```

These are **chosen assumptions**, informed by the earlier scope's3 us transition budget and1.5–2 us resource tax, not measured savings. Capping boundary credit at8 us prevents treating the long PDL spans as pure removable glue. Keep all projection/MoE/attention core times unchanged; this assumes successful port parity and gives no automatic lossless-compression or lookahead gain. The2 us M8 term is an attention-prep/combine fusion target and may already overlap the3 us transition term in a real implementation; the combined scenario is optimistic until separately instrumented. Tax covers counters, activation publication, dispatch and resource changes, but is not a measured upper bound on the cost of losing a resident CTA.

| Load | Assumed saving S/F us | Proposed S/F layer us | S11 load us | Proposed load us | Load factor vs S11 |
|---|---:|---:|---:|---:|---:|
| decode1 | 5.12 /5.00 | **88.18 /97.77** | 5936.12 | **5620.60** | 1.0561x |
| decode16 | 5.70 /5.70 | **148.67 /243.91** | 11094.57 | **10741.17** | 1.0329x |
| decode32 | 5.20 /5.20 | **192.41 /368.97** | 15076.83 | **14754.43** | 1.0219x |
| verify1 | 7.70 /7.70 | **127.21 /143.55** | 8625.87 | **8148.47** | 1.0586x |
| verify16 | 0 /0 | 301.06 /423.50 | 20624.93 | 20624.93 | 1.0000x |
| verify32 | 0 /0 | 399.30 /640.36 | 28613.68 | 28613.68 | 1.0000x |

Load time is `46*t_sliding + 16*t_full`. Score factor is
`product_over_6_loads(T_S11 / T_proposed)**(1/6)` = **1.02797x**. Multiplying1.706767423 gives **1.75451x**. The last two loads are explicitly unchanged because this design targets M≤32, not their M128/256 heavy-WGMMA branch.

If the projected gain requires a uniform regression of only about 5–8 us/layer from the resident inner loops, it disappears. Given the 264→132-CTA MoE change and the prior 15/16 projection-port losses in [closed recipes](../FINDINGS.md), such a regression is plausible. This uncertainty is more consequential than rounding the modeled gain.

Sensitivity: removing the possibly overlapping M8 attention-glue credit gives **+2.54%** score; doubling every scheduling/resource tax gives **+1.89%**. Adding a 5% regression to projection+MoE+attention profile durations on the four targeted loads changes the central result to **−0.73%**; a 10% regression gives **−3.97%**. These are sensitivity calculations using duration attribution, not measured regression bounds. They show why preserving the inner loops comes before building a larger scheduler.

### Estimated effect of each transferable technique

The following **independent sensitivity scenarios** cover every mechanism above. Values are **saved us per complete 62-layer load**, in order D1/D16/D32/V1/V16/V32. They use the stage table and measured load denominators. They are not additive: scheduling, glue, prefetch and faster inner loops can reclaim the same interval. “Zero credited” means no demonstrated extra benefit beyond S11, not impossibility.

| ID / technique | Removes or reduces | Explicit modeled assumption | Saved load us, six loads | Score effect |
|---|---|---|---|---:|
| **F: fused glue + static fine-grained instruction schedule** | Launches, grid tails, some glue instructions/activation round trips; keeps true joins | Exactly the fusion budget above; M≤32 only | **315.52 /353.40 /322.40 /477.40 /0 /0** | **+2.80%**, conditional |
| **D: preserve/tune dense tiles and short-lived weight reuse** | Redundant loads, padding work and task restart instructions; not the compulsory bank read | Hypothetical 5% projection-duration improvement on all shapes; PDL attribution makes this an optimistic sensitivity | **151.59 /236.10 /193.25 /135.01 /211.69 /278.75** | **+1.62%** target; no donor-verified gain |
| **E: expert-local readiness + fused input gather/metadata** | Broad expert waits, route/gather instructions and extra launches | Hypothetical 5% reduction of small-M MoE durations, no larger-M change | **105.55 /210.62 /336.98 /276.61 /0 /0** | **+1.55%** target; **zero verified**, related arms lose |
| **C: inline exact CSF** | Scale bytes only, at added decode/index cost | Ideal50% scale-byte saving, scale fraction1/9, all saved bytes exposed: `moe_bytes/(18*4.58e6)` | **53.24 /161.26 /191.02 /159.70 /392.01 /454.12** | **≤+1.52%** byte-only ceiling for this assumption |
| **K: page/KV-address pipelining** | Address-dependency and memory latency; every KV byte still read | Hypothetical 5% attention-duration reduction, all shapes, not a fusion guarantee | **42.06 /116.62 /199.53 /76.08 /177.82 /283.12** | **+0.98%** target |
| **P: retained next-op operand** | First-tile load/drain latency and refetch instructions | Hypothetical 1 us exposed saving per M≤32 layer; unchanged mainloops/bytes | **62 /62 /62 /62 /0 /0** | **+0.46%** target; **zero central credit** |
| **Q: graph preparation/Rust replay, GPU admission/page allocation** | CPU and startup costs | Already outside Muse's measured GPU layer or already graph captured | **0 /0 /0 /0 /0 /0** | **0%** |
| **A: AMD chiplet placement, NVLink compute/communication overlap** | Remote-die/remote-GPU traffic and communication waits | Muse is one H200 with identity collectives | **0 /0 /0 /0 /0 /0** | **0%** |
| **N: native FP4/FP8-activation, dense weight requantization, sparse keys, speculation** | Bytes/work or number of target passes | Violates formats/every-key work or changes the defined load | **0 /0 /0 /0 /0 /0** | **Inadmissible/out of scope** |
| **V: static DAG/lifetime validation and replay epochs** | Deadlocks/races, not compulsory execution work | Correctness prerequisite; no timing credit | **0 /0 /0 /0 /0 /0** | **0% direct**, necessary for F/E/P |

The C scenario's exact per-layer S/F savings are **0.859/0.859, 2.617/2.555, 3.070/3.111, 2.576/2.576, 6.312/6.353, 7.318/7.342 us**. It uses the original one-byte-per 16 scale payload as byte basis. Exceptions, metadata, existing bank-index layout and decode overhead must be measured; the actual difference could be negative. D and K deliberately specify a5% target rather than pretending a donor's whole-engine multiplier applies to Muse.

Source-to-scenario mapping: Forge **F/P/V**; AMK **V/P/D** (its W8/W4 gains are **N**); Ada-MK **F/D/P**; Fleet **D/F**, chiplet-specific parts **A**; ETC **F/E/P/K/V**; MPK **F/E/P/K**, runtime admission **Q**, communication **A**; Hazy/FlashFormer **F/P/D/K**; Cohere **F/E/P**, parallel attention/MoE branch has no Muse credit; Flare **C/D/E/K**, preparation/runtime **Q**, reduced-precision/sparse paths **N**; AMD **F/P/E**, chiplet/precision/speculation mechanisms **A/N**.

### Reproducible arithmetic without executing repository modules

This computes the scenario directly from the saved S11 JSON; it does not import kernels, touch a GPU or query the judge:

```python
import json, math
from pathlib import Path
d = json.loads(Path('problems/muse_decode_layer/dev/levers/current.json').read_text())
assert d['source_identity']['timed_sha256'] == (
    'acda6c621b1f3b885f2111ea1b4482ae1d733e2f2493db2b7f0b852ccf66dfb7')
tax = {1: 1.5, 8: 2.0, 16: 2.0, 32: 2.5}
saved = dict.fromkeys(d['loads_us'], 0.0)
for r in d['rows']:
    m, s = r['tokens'], r['stages_us']
    delta = (0.4 * min(s['boundaries'], 8) + min(s['route'], 1.5)
             + (2 if m == 8 else 0) + 3 - tax[m]) if m <= 32 else 0
    saved[r['case']['load']] += r['case']['weight'] * delta
factor = math.prod(t / (t - saved[k]) for k, t in d['loads_us'].items())**(1/6)
print(saved, factor, d['baseline_source_speedup'] * factor)
```

## What is verified and what remains open

**Checked:** all requested primary papers/posts and the cited donor paths; 87dc's source/hash, exact six-M timings and changed-input diagnostics; S11 source identity and twelve-shape/six-load accounting; the newer landed NT1 wait-removal failures; arithmetic of all scenarios. The external speedups above are the authors' measurements and comparisons as stated. Missing exact B16/32 serving numbers are not digitized from unlabeled plot heights or invented.

**Inferred:** occupancy/wave structure and logical rereads explain why87dc is a bad implementation test of fusion. They do not identify each cause's H200 percentage. The retained-operand schedule fits a proposed shared-memory envelope but has not demonstrated inner-loop parity or a whole-layer gain. CSF admissibility depends on preserving Muse's actual native scale values, and its compressed size/speed must be audited. All numerical score improvements proposed here remain unverified on H200.

**Implementation prerequisites:** standalone task parity on the actual resident geometry; complete TASK routing/cache/residual checks on changed inputs; no-local and skewed-route liveness; batch and page variants from correctness tests; compile/prepare below 300 s; paired cold-read AB/BA and A:A whole-layer comparisons with multiple seeds. Diagnostic stamps belong in separate timing runs. A shortened wait or lower summed profiler time is insufficient; promote only an improvement in the actual six-load score.

## Ranked transferable techniques and first H200 experiments

The ranking is by the numerical conditional score effect above, with ceilings and unverified targets explicitly distinguished. It is not a recommendation to build every item or to add their gains.

1. **Static tile-level fusion with retained current inner loops — +2.80% scenario, no verified gain.** Evidence: Hazy/Forge's instruction/counter/operand-lifetime design and ETC's6–9% same-code static-scheduling ablation;87dc demonstrates the cost of changing the loops. **First experiment:** an S11-paired `O→boundary1→pool/router→route` prefix on M1/8/16/32, with the exact same projection tile bodies in unfused and resident arms, and a third resident arm with phase-wide waits. Preserve the W13 suffix for full-output correctness. Require dense task parity and total prefix benefit before widening to the layer; this directly tests F without relying on already failed NT1 round scheduling.

2. **Dense task geometry, padding and weight-reuse preservation — +1.62% for a 5% projection improvement, no donor-verified gain.** Evidence: Ada-MK's padding/vector-load mechanism, Fleet's B32 reuse benefit, and87dc's fixed32-token tile/full-bank rereads. **First experiment:** same S11 dense tiles standalone versus inside a resident dispatch with one opcode, no counter waits and no lookahead, all four dense roles and M1/8/16/32/128/256; then test weight-major token-tile order against identical arithmetic. This locates the port/occupancy tax before attributing any loss to synchronization.

3. **Expert-local scheduling and producer-owned metadata — +1.55% for a 5% small-M MoE improvement; current related recipes lose.** Evidence: ETC/MPK's route-dependent event model, plus our17512/3af06/4dd005 receipts proving that shorter readiness/publish waits alone do not win. **First experiment:** a new per-expert ready-task placement that keeps the S11 tile body, per-CTA chunk lengths and resident geometry unchanged; compare identical routes and work, explicit ready-count placement on/off, with instruction/phase diagnostics. Do not resubmit RND/ROT/PIPE/WPUB. If no placement can preserve those work properties, charge the changed work explicitly and do not call the result a synchronization ablation.

4. **Exact inline CSF scale reconstruction — ≤+1.52% under ideal half-scale-byte saving; likely smaller and possibly negative.** Evidence: b12x's raw-byte base/offset/exception encoder and inline staging, our36 KiB NT1 tile's1/9 scale fraction, and the donor's documented packed-subnormal loss. **First experiment:** first audit every raw scale byte and packed-global-scale reconstruction on CPU/prepare; then an H200 identical-S11-NT1 comparison of current scale indices, lossless CSF inline, and timed expand-to-scratch, including all metadata/decode/reset/reduction cost across all six loads and held-out routing seeds. Fail the representation on any lost native value before timing it.

5. **Page-table/KV pipeline carried across task boundaries — +0.98% for a 5% attention reduction, no assigned byte saving.** Evidence: Hazy/FlashFormer/MPK per-head splits and pipeline continuity; Flare's explicit FP8 paged-state staging is a structural donor, while its sparse selection is inadmissible. **First experiment:** current S11 attention arithmetic with only page-index lookahead and query/group readiness changed, page1 and every key retained, B1/16/32 at q1/q8, separate new-KV append hazard tests, then paired whole-layer timing. This must preserve S11's existing full-verify vq8 gains.

6. **One next-op shared operand per free worker — +0.46% if 1 us/layer is exposed; zero central credit.** Evidence: Hazy/MPK's shared-page lifetime, FlashFormer's additional2% deep-linear overlap gain, and our negative L2/ramp oracles. **First experiment:** in the parity-proven prefix, compare no prefetch, the old L2 hint, and one 36 KiB retained operand after the last current-stage load, all with identical task order. Stamp load issue/completion and publication only in diagnostic arms; demand a whole-prefix win with no increase in weight reads. This tests a different mechanism from the already closed broad L2-prefetch recipes.

[forge]: https://arxiv.org/html/2609.12379v1
[amk]: https://arxiv.org/html/2606.09682v1
[ada]: https://arxiv.org/html/2605.11581v1
[fleet]: https://arxiv.org/html/2604.15379v1
[event]: https://arxiv.org/html/2604.13327v2
[mpk]: https://arxiv.org/html/2512.22219v2
[ff]: https://arxiv.org/html/2505.22758v2
[hazy]: https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles
[hazy-tp]: https://hazyresearch.stanford.edu/blog/2025-09-28-tp-llama-main
[fluffy]: https://hazyresearch.stanford.edu/blog/2025-11-17-fluffy-kittens
[retire]: https://hazyresearch.stanford.edu/blog/2026-08-05-retire-the-abstractions
[mla]: https://hazyresearch.stanford.edu/blog/2025-03-04-thundermla
[cohere]: https://cohere.com/blog/megakernels
[amd]: https://rocm.blogs.amd.com/software-tools-optimization/megakernel-intro/README.html
[hopper]: https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html#shared-memory
[ptx]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#memory-consistency-model
[flare-pin]: https://github.com/NVIDIA/flare-local-inference/blob/bce032c014bae34502ba22d3304fa084d5b40afe/pyproject.toml
[flare-readme]: https://github.com/NVIDIA/flare-local-inference/blob/bce032c014bae34502ba22d3304fa084d5b40afe/README.md
[flare-design]: https://github.com/NVIDIA/flare-local-inference/blob/bce032c014bae34502ba22d3304fa084d5b40afe/docs/core-design.md
[flare-moe]: https://github.com/NVIDIA/flare-local-inference/blob/bce032c014bae34502ba22d3304fa084d5b40afe/ignition/components/moe.py
[flare-linear]: https://github.com/NVIDIA/flare-local-inference/blob/bce032c014bae34502ba22d3304fa084d5b40afe/ignition/components/linear.py
[flare-state]: https://github.com/NVIDIA/flare-local-inference/blob/bce032c014bae34502ba22d3304fa084d5b40afe/ignition/recipes/qwen38/state.py#L54
[flare-attn]: https://github.com/NVIDIA/flare-local-inference/blob/bce032c014bae34502ba22d3304fa084d5b40afe/ignition/recipes/qwen38/attention.py#L275
[csf-encode]: https://github.com/local-inference-lab/flashinfer/blob/259648d983fdd1488bb6b8df8e5ea6134901a4d4/flashinfer/experimental/b12x/_lib/quant/csf_encode.py
[csf-blocks]: https://github.com/local-inference-lab/flashinfer/blob/259648d983fdd1488bb6b8df8e5ea6134901a4d4/flashinfer/experimental/b12x/_lib/quant/csf_blocks.py
[csf-inline]: https://github.com/local-inference-lab/flashinfer/blob/259648d983fdd1488bb6b8df8e5ea6134901a4d4/flashinfer/experimental/b12x/_lib/quant/csf_inline.py
[b12x-readme]: https://github.com/local-inference-lab/flashinfer/blob/259648d983fdd1488bb6b8df8e5ea6134901a4d4/flashinfer/experimental/b12x/moe/fused_moe/README.md
[b12x-scale-contract]: https://github.com/local-inference-lab/flashinfer/blob/259648d983fdd1488bb6b8df8e5ea6134901a4d4/docs/design_docs/b12x_nvfp4_packed_scale_contract.md
[b12x-gemv]: https://github.com/local-inference-lab/flashinfer/blob/259648d983fdd1488bb6b8df8e5ea6134901a4d4/flashinfer/experimental/b12x/gemm/bf16_gemv/_kernel.py
[b12x-tcgemv]: https://github.com/local-inference-lab/flashinfer/blob/259648d983fdd1488bb6b8df8e5ea6134901a4d4/flashinfer/experimental/b12x/gemm/bf16_gemv/_tensor_core.py
[b12x-gemm]: https://github.com/local-inference-lab/flashinfer/blob/259648d983fdd1488bb6b8df8e5ea6134901a4d4/flashinfer/experimental/b12x/moe/_shared/kernels/w4a16/kernel/gemm.py
[b12x-packed]: https://github.com/local-inference-lab/flashinfer/blob/259648d983fdd1488bb6b8df8e5ea6134901a4d4/flashinfer/experimental/b12x/_lib/intrinsics/packed.py#L598
[b12x-direct]: https://github.com/local-inference-lab/flashinfer/blob/259648d983fdd1488bb6b8df8e5ea6134901a4d4/flashinfer/experimental/b12x/moe/_shared/kernels/w4a16/kernel/direct.py
[b12x-fused]: https://github.com/local-inference-lab/flashinfer/blob/259648d983fdd1488bb6b8df8e5ea6134901a4d4/flashinfer/experimental/b12x/moe/_shared/kernels/w4a16/kernel/fused.py
[hazy-pages]: https://github.com/HazyResearch/Megakernels/blob/7309cec801537b61fea3b50d7dfe454a6cde578e/include/controller/page_allocator.cuh
[chain-source]: ../../problems/muse_decode_layer/cand_r3_best7_reset.py
