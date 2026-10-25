# Cooperative glue repair

The 80b70f7f result diagnoses an implementation bug: 576 repeated norms/token and
serial last-CTA tails. It does not close fusion or the +6.7% glue lever.

`kernel.cu` replaces that mapping with a cooperative resident grid (one 128-thread
CTA per SM). Eight CTAs partition each 8192-column boundary row. Each chunk writes
one squared-sum partial; a row leader computes its post RMS once. Chunks form the
residual and a second squared-sum partial; the leader computes pre RMS once, then
all chunks write the BF16 hidden row. This uses the actual two norms rather than
the old expanded second-moment expression. The once-normalized row is shared by
all pool/router GEMM tiles. Output reduction and boundary tails are distributed
across the grid. Pool pre-expert RMS/top8 runs once per token, with parallel output
stores. There are no last-CTA arrival counters or unbounded launch-order waits:
CUDA cooperative launch guarantees residency for every `grid.sync()` participant.

`route_glue/route_once.cu` similarly computes routing/pre-expert RMS once per row
before the expert body, publishes global IDs/weights/inverse norms, then copies
the shared summary into each expert worker. It emits shared `MoERouteMeta` v1 at
M<=32; all 736 words, including padding and remote slots, are checked on every
small-M replay. The packed expert MMA body and E4M3 scale storage are unchanged.
M256 retains the incumbent Humming path.

`build_lab.py` freezes all sources in one payload. Cells are pool, tail, all,
epi, and route metadata; all 12 decode/verify sliding/full cases use identical
inputs and weights, cold256MiB AB/BA6, A:A and three changed-input graph replays.
The proj host lacks experts, so seed5252 draws a 32-expert packed NVFP4 bank with
the organizer's distribution; this is not the prior MoE host's exact bank.
Dense/router/norm seed941732 is unchanged. No dequantized expert bank is prepared.

A flipped cell is recorded with IDs, 8th/9th biased-sigmoid gap, cutoff loss and
candidate score perturbation. It is disabled for that M; other cells continue.
Any flip disables admission even when a near-tie explains it. The old M256
pool epilogue is retained as an untimed diagnostic for failed receipt8a3d7181;
`9eeb39ba` shows that event is two experts swapping slots inside the same top-8
set, not a routing error (FINDINGS "Failures").

Local validation: 36 native graph cases, zero flips, maxL2 0.000088; 180 full-layer
comparisons, zero flips, maxL2 0.004623; 24 exact metadata descriptors. Injecting a
wrong expert disables one cell, skips its other window and finishes 174 timings.
Nine host checks and three static checks pass. SM90a builds pass: cooperative
glue56 registers (pool has4B stores/24B loads of spills), route128/no spills.
All switches in `BEST7_COOP_GLUE_CELLS` remain empty.

H200 job **9eeb39bab3fd4742** (proj; send 270212081442eff6, payload SHA
`da57e4ccf74a6d6559ae1e9de882de77201a72d68203740deca6c72c369a9f9f`) loses every cell at
every case, B/A 1.17-9.5x (case G0 pool 156.8 vs 103.5 us, route 327.7 vs 104.0; G8 all
1500.5 vs 218.6). This closes the exact recipe; the mechanism is not isolated (the
per-row grid-wide sync is the likely cost, unmeasured). The case-10 flip that disabled
three cells is a slot-order swap. `sent.json` and `admission.json` pin the sources.

Build: `build.py`, `build_cell.py`, `build_lab.py`. Screen: `validate.py`, then
`screen.py`; `screen.py --inject-flip` tests failure isolation. Run the GPU scripts
through `CUDA_VISIBLE_DEVICES=0 tools/img python` from the repository root.
