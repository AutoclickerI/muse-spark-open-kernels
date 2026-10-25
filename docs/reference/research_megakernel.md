Read when designing cross-stage operand lifetimes, validating a persistent schedule, or auditing Flare's exact scale codec.
Serves: G2.

# Decode megakernels: evidence and a Muse H200 design

This reference owns scheduling and codec contracts. FINDINGS owns Muse measurements and the
failed `87dc944b` recipe; SOTA owns external comparisons; PLAN owns proposed experiments.
A competitive resident kernel needs competitive inner loops, precise readiness and operand
storage spanning operations. Launch count alone establishes none of them.

## Scheduling and operand lifetime

| Donor | Durable mechanism | Transfer limit |
|---|---|---|
| Forge | Per-SM typed instruction streams; producer counters replace broad joins; shared pools follow operand lifetimes across operations | Streams alone hide no latency; dense BF16 operators need Muse-format implementations. [§4.1][forge] |
| AutoMegaKernel | Static DAG validator rejects cycles, unsatisfiable waits, wrong SM order and buffer races; monotonic release/acquire counters | A multi-producer count means an all-join, not any first-k subset. L2 prefetch is not retained operand storage. [§4][amk] |
| Ada-MK | Offline movement/compute DAG, role/lifetime reuse, padding-aware loads, lookahead N+2 | GPTQ/Ada decoder is not NVFP4/Hopper. [§3–4][ada] |
| Event Tensor | Event notify decrements toward zero; static SM queues or dynamic ready queues; route-dependent counts/ranges | Dynamic queues add atomics/contention; ready queues must not pin unavailable producers. [§2–3][event] |
| MPK | Event-triggered SM tasks; shared pages acquired/released at task boundaries; descriptor prefetch | Reserved schedulers consume worker capacity; runtime admission benefits are outside this layer. [§5][mpk] |
| Hazy | Per-SM streams, per-chunk consumers, shared operand pages spanning instructions | Whole-model dispatch/communication benefits do not all exist here. [page allocator][hazy-pages] |
| FlashFormer | Unified shared-memory pipeline across operators; some global joins remain; current-token KV handled separately | Avoid prefetch races with this call's KV append. [§4][ff] |
| Cohere | Static waves plus dynamic attention/expert drains; common worker ABI, separate controller barrier; load weights before activation waits | Its parallel attention/MoE branches do not match Muse's post-attention router dependency. [implementation][cohere] |

HBM continuity uses otherwise idle bandwidth; shifting successor reads earlier cannot overlap two
saturated streams. A cached hint still needs the real operand load. Shared next-op pages save that
refetch only if storage survives the transition and readers release ownership correctly.
Register/smem residency changes can cost more than removing a boundary saves. A merged binary
also allocates its maximum live register footprint; aliasing shared slabs does not reduce that.

## Synchronization and replay correctness

Counter readiness and data readiness are separate. Finish relevant ordinary/TMA stores and
proxy ordering before device-scope release publication; consumers acquire before dependent
loads. Async-copy full/empty barriers have their own slot generations. A count over unordered
producers cannot certify a named K-slice. [PTX memory model][ptx]

Validate an acyclic task DAG, satisfiable waits, page release after every reader, exact producer
sets/counts, and producer progress under actual occupancy. Blocking consumers cannot fill every
resident slot while their producers remain unissued. Device route-dependent work needs bounded
scratch/ring capacity under skew, including no-local-route tokens that must publish zero output.

Invocation epochs must be exact, not merely nonzero initialized flags. Event-bank reuse waits
until old tasks and async transfers retire. Every invocation, including no-local-route work,
finishes terminal bookkeeping. Stream-ordered graph replays may share scratch; simultaneous
independent invocations require disjoint scratch. Epoch advancement by the final retired worker
avoids treating a graph node's persistent grid ID as invocation identity (DeepGEMM details:
[research_deepinfer](research_deepinfer.md#mega-gate-selection-and-glue-ownership)).

True full-row norms and per-token expert combinations remain joins. Fine-grained readiness
can localize them, not permit normalizing incomplete rows. Validate alternating changed inputs,
poisoned scratch, epoch wrap, none/some/none routes and new-KV append ordering separately from
performance instrumentation. Static SASS establishes dependencies/resources, not their runtime
wait shares. Whole-prefix/layer timing determines whether a shortened wait merely moved elsewhere.

## Flare and b12x: exact scale coding

Pins: Flare `bce032c014bae34502ba22d3304fa084d5b40afe`; its b12x dependency
`259648d983fdd1488bb6b8df8e5ea6134901a4d4`. Ignition prepares/compiles/captures graphs;
Helios replays them in Rust. Opaque native calls remain operations: graph-level forks are not
proof of a persistent whole-layer GPU kernel. [core design][flare-design]

CSF encodes native scale bytes without float conversion. Each row chooses a base covering the
most values in a 16-value window for NVFP4 (four-bit offsets), or two-value window for MXFP4
(one-bit offsets); outliers remain verbatim exceptions. Bases, offsets, group/page metadata and
exceptions can outweigh the saved codes. The one-bit MXFP4 ratio is not an NVFP4 ratio.
[encoder][csf-encode] · [storage layout][csf-blocks]

`CsfInlineReader` stages compressed records, reconstructs native scale words in shared memory,
and retains row bases/bounds across the K sweep. Inline decoding repeats per consuming tile;
timed expansion reconstructs each selected expert once per call but adds scratch traffic.
The complete-call comparison includes routing, resets, expansion and reduction. Preparing
lossless codes is distinct from retaining widened numeric scales. [reader][csf-inline] · [contract][b12x-readme]

The donor packed-scale contract discards nonzero normalized E4M3 values below 1/64. CSF preserves
those already packed bytes; it cannot restore discarded subnormals. A checkpoint retaining all
its scales proves nothing about held-out Muse weights. Audit raw-byte reconstruction and the
independent global-scale contract; import no such narrowing. [packed-scale contract][b12x-scale-contract]

b12x's packed E2M1→BF16 helper expresses nine source bit operations/four values before scale
multiply/moves. Source operations are not SASS totals; fragments and ptxas fusion differ.
Its W4A16 mainloop's cp.async, register dequant and BF16 MMA are software patterns that can port;
native FP4/activation-quantized paths cannot be presumed SM90-compatible. Its persistent
FC1→activation→FC2 kernel still has software grid barriers. [helper][b12x-packed] · [mainloop][b12x-gemm] · [fused kernel][b12x-fused]

[forge]: https://arxiv.org/html/2609.12379v1
[amk]: https://arxiv.org/html/2606.09682v1
[ada]: https://arxiv.org/html/2605.11581v1
[event]: https://arxiv.org/html/2604.13327v2
[mpk]: https://arxiv.org/html/2512.22219v2
[ff]: https://arxiv.org/html/2505.22758v2
[cohere]: https://cohere.com/blog/megakernels
[ptx]: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#memory-consistency-model
[flare-design]: https://github.com/NVIDIA/flare-local-inference/blob/bce032c014bae34502ba22d3304fa084d5b40afe/docs/core-design.md
[csf-encode]: https://github.com/local-inference-lab/flashinfer/blob/259648d983fdd1488bb6b8df8e5ea6134901a4d4/flashinfer/experimental/b12x/_lib/quant/csf_encode.py
[csf-blocks]: https://github.com/local-inference-lab/flashinfer/blob/259648d983fdd1488bb6b8df8e5ea6134901a4d4/flashinfer/experimental/b12x/_lib/quant/csf_blocks.py
[csf-inline]: https://github.com/local-inference-lab/flashinfer/blob/259648d983fdd1488bb6b8df8e5ea6134901a4d4/flashinfer/experimental/b12x/_lib/quant/csf_inline.py
[b12x-readme]: https://github.com/local-inference-lab/flashinfer/blob/259648d983fdd1488bb6b8df8e5ea6134901a4d4/flashinfer/experimental/b12x/moe/fused_moe/README.md
[b12x-scale-contract]: https://github.com/local-inference-lab/flashinfer/blob/259648d983fdd1488bb6b8df8e5ea6134901a4d4/docs/design_docs/b12x_nvfp4_packed_scale_contract.md
[b12x-gemm]: https://github.com/local-inference-lab/flashinfer/blob/259648d983fdd1488bb6b8df8e5ea6134901a4d4/flashinfer/experimental/b12x/moe/_shared/kernels/w4a16/kernel/gemm.py
[b12x-packed]: https://github.com/local-inference-lab/flashinfer/blob/259648d983fdd1488bb6b8df8e5ea6134901a4d4/flashinfer/experimental/b12x/_lib/intrinsics/packed.py#L598
[b12x-fused]: https://github.com/local-inference-lab/flashinfer/blob/259648d983fdd1488bb6b8df8e5ea6134901a4d4/flashinfer/experimental/b12x/moe/_shared/kernels/w4a16/kernel/fused.py
[hazy-pages]: https://github.com/HazyResearch/Megakernels/blob/7309cec801537b61fea3b50d7dfe454a6cde578e/include/controller/page_allocator.cuh
