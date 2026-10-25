# P7 gate split: mechanism of the remaining gap

The admitted gate split is in S11 final (`patch_gate_split.py`; the original
paired results are in its docstring). `build_trace.py` builds
`lab_gate_trace.py` from the frozen original `prof/lab/gate_split.py` chain.

The diagnostic records native attention thread-0 entry/return per CTA and
gate CTA entry, GEMM-partial store, last-split wait entry/exit and gate output
store. The native globaltimer gives a shared clock. An isolated gate graph
separates the GEMV's own duration from concurrency slowdown. Attention's
thread-0 return is a proxy for CTA completion, not a hardware completion event.
Traced and untraced gate chains must produce bit-identical outputs at every
shape before any timing evidence is used.

Local admission passes all six decode shapes, with eight fresh cold trace
replays per shape. Local ratios are not H200 performance evidence. Each
simultaneously live graph owns a capture stream, and each chain object stays
alive while the graph holds pointers into its ordinary P/counter allocations.
The first trace failed replay agreement because a temporary control-chain
object was freed after capture; retaining that owner fixes the reproducer.
No numerical check was relaxed.

Payload SHA-256 `fe25ff0c`. The proj host's checks are exhausted; its unsent
waiter was stopped and the identical lab was rerouted to the prefill host.
H200 `0a48a058edf54880`: b1 sliding attention ends at 8.944 us,
gate partials at 7.280, gate wait ends 9.280 and gate output at 10.208
(isolated gate 8.272 us). At b1 full, gate partials stretch to 19.456 versus
6.624 us in isolation; attention ends 19.744, gate ends 21.120. B16 full
shows the same sharing: partials at 97.552 versus 6.608 isolated, attention
ends 106.224, gate ends 108.160. At B32 full, gate starts late at 168.944,
attention ends 191.376 and gate ends 193.296. These are relative device-clock
stamps; attention return is a completion proxy. The trace itself changes
scheduling (e.g. b1 sliding trace/plain chain ratios 0.96513/0.94619), so it
supports the contention/late-launch mechanism without an isolated causal
latency claim. The gate-split recipe is closed; structural glue continues in
`../producer_glue/`.
