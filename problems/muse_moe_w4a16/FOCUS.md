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

The new V9 NT3 (24-row block) specialization is in `dev/v9_reuse/nt3_dropin.py` and
`cand_r3_v9_nt3.py`; its 12-seed sm120 means are 370.7/507.6 us at M=128/256, and
144 captured configurations plus 576 fresh route/input replays pass the FP32 gate (max L2
0.001333). sm90a ptxas reports NT1/2/3/4 at 125/219/255/255 registers; NT3 has only
8-byte spills versus NT4's 288-byte frame. H200 lab `711370966a58ac2b` is pending.

The corrected Humming probe `376ef26ba9bb46a0` is incomplete: it timed bs=1 then failed
on `shape_m is not within any range`, so it provides no decode-weight agreement evidence.

The raw17-derived WGMMA decode probe `dev/v9_wgmma/raw64_dropin.py` uses packed
NVFP4/E4M3 banks and `m64n64k16` token tiles, with W13 SwiGLU fused before W2.
It is correct but closed on H200: receipt `63c980e800194764` gives raw64
534/679 us at M=128/256 versus Humming masked 342/450.  Do not promote it.

Humming is the current MoE bar.  Profile source key `e858855cac1b5ad0` sweeps
masked configs, block-M, SM allocation and per-stage event spans; variant source
key `053c794fc0aba59a` compares masked, masked-TMA and indexed plans.  Both use
the packed bank and exact R0 plus 12-seed FP32 gates; receipts are pending.
