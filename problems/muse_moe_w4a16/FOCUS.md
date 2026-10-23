# MoE focus

Serves: G3, G2, G1.

V11 is now the MoE board best: H200 receipt `41bcf27cfc0e4eff` is 1.664x overall,
with decode 2.185/1.684/1.398 and verify 1.800/1.592/1.439 at M=1/16/32 and 8/128/256.
V9 receipt `f6028a478d594b31` was 1.654x; V11's Humming path wins V32.

`cand_v17.py` is the next portfolio. It keeps V9, V12, V16, Humming/grouped-wide, and adds
an adaptive expert-major kernel: each expert selects 8/16/32/64 token rows from `n_e`, reusing
each W13/W2 tile across that block. Prepare ranks every path with graph replay, 1 GiB L2 flush,
three fresh hot-tail routing sets, and keeps the fastest shape choice.

The sm120 screen passes 5/5 tests and 6/6 graph shapes at 1.269x local GM; prepare is 174.0 s.
Local choices were V9 at M=1/8/32, V12 at M=16/128, and adaptive at M=256 (`NT=1` on sm120);
H200 has more shared memory and may select wider row tiles. The adaptive M256 graph passed.

V9's direct-byte model still gives 1.50 TB/s at verify-32 versus 3.0+ elsewhere. M256 routing
averages 386.93 pairs, 24.39 experts, and 60.53 `ceil(n_e/8)` blocks versus 36.76 at M128;
that 1.65x block growth tracks V11 H200 time growth 1.57x better than bytes (1.08x).
Static CTA imbalance is only 1.01x, so row-block reuse is the active mechanism.

The serving port remains V9 native through M=128 plus grouped wide from M=256, 864 MiB/GPU;
`V9_ABI.md` is unchanged. Do not port the adaptive path until its H200 receipt proves a win.
