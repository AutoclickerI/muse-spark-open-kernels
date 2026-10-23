# MoE focus

Serves: G3, G2, G1.

V11 is the H200 MoE board best: receipt `41bcf27cfc0e4eff` is 1.664x overall, with
2.185/1.684/1.398 decode and 1.800/1.592/1.439 verify at M=1/16/32 and 8/128/256.
V9 was 1.654x; Humming wins V32.

V18 copies the committed V17 portfolio and keeps its 1% correctness gate. It also accepts a
full decode-layer weight bank when present and invokes `problems/muse_decode_layer/moe_tune.py`
for router-derived shape selection; the standalone MoE harness has no router fields, so its
local screen uses the existing synthetic fallback. V18 passes 5/5 tests and 6/6 graphs at
1.264x sm120 GM, with 8.3 s cached prepare.

The decode-layer verify-32 mechanism is routing skew plus grouped-path fixed work. Direct sm120
M256 graph timings (ms) for real hot-tail / matched-local-count uniform / all-local uniform are:
grouped 0.915/0.866/1.818, V7 0.584/0.763/1.693, V9 0.668/0.841/4.738,
V11/Humming 0.536/0.760/1.652, and V17 adaptive 0.662/0.827/4.659. Real routing had
672 local pairs, 8 experts, and a 221-route hot expert for the default loaded router; the
judge's `sim/problems.py` target is 386.93 pairs and 21.35 experts. Matched uniform had 672
pairs across 32 experts. Grouped performs histogram+plan+scatter, two GEMMs and a combine (six launches),
so its fixed task/launch cost survives the hot tail. The new tuner routes loaded router weights
on random hidden states and applies the same 1% gate before choosing a path.

Serving remains V9 native through M=128 plus grouped wide from M=256, 864 MiB/GPU; no serving
port change is justified until the decode-layer H200 result validates router-aware selection.
