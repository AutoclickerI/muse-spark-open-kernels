# Routed NVFP4 MoE serving port

Serves G1/G2. Read with [expert_layout.md](expert_layout.md) for bank geometry.
The current serving producer is **stock Marlin at every M**. V9 is retained
unbound; grouped and Humming have no serving dispatch call sites.

## Producer admission

Real-weight H200 lab `26d7e05ab97d4f96` gives V9/Marlin931/563us at M512 and
1332/716us at753. Humming/V11 fail agreement (relative0.11–0.32). These replace
the earlier synthetic routing decisions. The desired policy is V9 through256
and Marlin above;257–511 is explicitly assigned to Marlin as well.

The requested hybrid is blocked by layout ownership. The old V9-only loader
omits Marlin packed parameters, so merely changing the large-M branch would
read missing tensors. The current hook
`MuseSparkNvFp4FusedMoEMethod._process_weights_for_marlin` runs stock Marlin
preparation, releases serialized aliases and sets `_muse_v4_state=None`.

The incoming MoE-lane source now preserves raw E4M3 scale bytes and converts
them in-kernel, resolving the earlier encoding boundary. This lane did not
edit that source. A V9 reader over the Marlin bank, or an admitted bounded
additional layout, is still needed to bind the hybrid. Another full bank costs
54.358GB/GPU and cannot be retained. A single-layer layout is0.906GB and is
permitted if its measured speed and actual KV/graph headroom justify it; no
such copy is currently selected. There is no prepared dequantized weight bank.

## Output and collective ownership

Any producer may return dense `[T,topk,H]` expert rows. The finalize interface
uses an identity index for local slots and `-1` for remote experts. It preserves
separate gate/up global scales, every post-expert norm and BF16 activations.

At M<32, CuTe replaces stock's Torch finalize sequence. At M>=32, standalone
dense finalize retains stock's already-fused Triton implementation. Extending
CuTe across all prefill sizes was locally slower and added index-map launches.
A supplied fused-AR workspace always owns the reduction: the caller may skip
the outer all-reduce only after that ownership transfers.

The future V9 adapter calls `_v9_call` directly through M256, independent of the
standalone kernel's grouped dispatch. It bounds routing scratch from compiled
shared-memory limits and keeps chunk maps global. The cooperative V9 counters
are per-call; streams must not share generations. A legal load-time repack may
change layout only; scales remain stored E4M3 and dequantization stays in-kernel.
No model-loader source is changed.

## Checks and remaining evidence

The production Marlin storage test covers12 shapes M1..2048, including all
257–753 boundaries,72 fresh graph replays and all-remote routing. M128..753 is
bit-exact against stock; larger graph/eager variation is within stock controls.
`test_dense_finalize_dispatch.py` checks28 shapes/norm forms,84 fresh graph
replays and explicit fused-AR ownership. Standalone runtime-M CuTe matches the
pinned static-M kernel at30 shapes/90 fresh replays; standalone/AR sm90a compile
checks pass.

The full TP1/TP2 native/forced-ABI margin, random and teacher-forced gates are
recorded in [serving_levers.md](serving_levers.md). No local result establishes
new raw-scale V9 performance on H200, eight-rank multicast arithmetic, the full
released checkpoint's memory, or hidden-workload TTFT. The old parked prefill
diagnostic embeds the retired scale encoder and is not admitted under the new
stored-E4M3 boundary.
