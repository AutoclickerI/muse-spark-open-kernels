# Routed NVFP4 MoE serving port

Serves G1/G2. Read with [expert_layout.md](expert_layout.md) for bank geometry.
The current serving producer is **stock Marlin at every M**. V9 is retained
unbound; grouped and Humming have no serving dispatch call sites.

## Producer admission

Real-weight H200 lab `26d7e05ab97d4f96` gives V9/Marlin931/563us at M512 and
1332/716us at753. Humming/V11 fail agreement (relative0.11–0.32). These replace
the earlier synthetic routing decisions. The desired policy is V9 through256
and Marlin above;257–511 is explicitly assigned to Marlin as well.

The hybrid's storage blocker is resolved by a direct V9 reader over Marlin's
bank. `MuseSparkNvFp4FusedMoEMethod._process_weights_for_marlin` still runs
stock preparation. Once admitted, `muse_spark_expert_bank.py` joins the stock
codes/scales without changing bytes and binds V9 to the same allocations;
Marlin keeps its original contiguous views. `muse_marlin_v9_reader.py` derives
only the weight-reader region of the MoE lane's source, retaining its launch
ABI, routing and arithmetic. The owned `muse_spark_moe.py` was not edited.
`MARLIN_READER_ADMITTED=False` leaves Marlin at every M until H200 lab source
`6a3168ef5d749e10` establishes the new reader's performance.

The incoming MoE-lane source now preserves raw E4M3 scale bytes and converts
them in-kernel, resolving the earlier encoding boundary. This lane did not
edit that source. The direct reader converts stock's packed scale representation
in-kernel; no expanded weight bank is prepared. Resident expert storage remains
54.358GB/GPU, with zero additional layout or timed repack workspace. A serialized
one-layer bridge was tested and rejected: even its ideal4.8TB/s traffic cost
is377us/layer versus a114us c32 break-even (13 turns,220 tokens,tau4.05).

## Output and collective ownership

Any producer may return dense `[T,topk,H]` expert rows. The finalize interface
uses an identity index for local slots and `-1` for remote experts. It preserves
every post-expert norm and BF16 activation. The serving sidecars reproduce
stock Marlin's rounded, shared gate global; standalone V9's independent FP32
gate/up scale convention is not substituted for stock's.

At M<32, CuTe replaces stock's Torch finalize sequence. At M>=32, standalone
dense finalize retains stock's already-fused Triton implementation. Extending
CuTe across all prefill sizes was locally slower and added index-map launches.
A supplied fused-AR workspace always owns the reduction: the caller may skip
the outer all-reduce only after that ownership transfers.

The V9 adapter calls `_v9_call` directly through M256, independent of the
standalone kernel's grouped dispatch. It bounds routing scratch from compiled
shared-memory limits and keeps chunk maps global. The cooperative V9 counters
are per-call; streams must not share generations. A legal load-time repack may
change layout only; scales remain stored E4M3 and dequantization stays in-kernel.
No model-loader source is changed.

## Checks and remaining evidence

The new production shared-bank test covers13 shapes M1..2048, all257–753
boundaries, changed-input graph replays, shifted EP ownership and remote-zero
routing. The direct reader is bit-identical to native V9 with matched global
scales at M1/8/32/128/256; versus stock Marlin relative L2<=0.00538,
maxabs0.03125. Its own eager/graph difference is zero. The Marlin large-M path
is exact through753 and larger variation stays within stock controls. The new
reader compiles for sm90a; local timing does not admit it on H200.
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
