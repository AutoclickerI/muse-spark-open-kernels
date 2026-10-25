# vq8: paged FP8 attention for q_len 8 (and 1), native FP8 WGMMA

Replaces the native_verify8 design, whose H200 clocks (a1679f9c) put ~70% of each 64-key tile in the
consumer's V transpose + barrier (slot c[2] ~990 cycles) and softmax (c[4] ~770).

Design (`vq8.cu`):
- One persistent CTA per SM (384 threads, 1 CTA/SM by shared memory). Stream-K: the (request, head pair,
  64-key tile) space, each segment = its cached tiles + one virtual tile for the block's own L keys, is
  cut into G = min(#SM, T/min_tiles) equal ranges; a CTA writes one partial per segment it touches.
- Both KV heads of a pair per CTA: each cache row read is one 128-byte line (K, V), one page-table
  lookup per key for both heads.
- Producer warps 8-11: cp.async 16 B (8 lanes per 128-byte row) into SW64 K and raw-V tiles; page-table
  entries PD = LAG+2 (>= 4) tiles ahead through a shared ring filled by 4-byte cp.async in the same commit
  groups (each warp its own 16 keys; the publish's wait covers them). Iteration it first publishes tile it-LAG
  (cp.async wait, ldmatrix.trans + byte_perm transpose of V into the K-major PV operand with FA3's key order
  inside each 16-key group, operand column 4t+j = key (2t, 2t+1, 8+2t, 9+2t)[j]; arrive full), then waits for
  tile it's empty slot and issues it. LAG = ST-2: a consumer step holds tile rr while it waits for rr+1, so
  the publish of rr+1 must not depend on the release of rr-1, or consumer and producer serialize (H200
  c9c4ef99 / d5bcbc7d: 2,700-2,900 cycles per tile with each side ~50% waiting; LAG = ST-2 gave ~1,700,
  44aa67ba / 1b993e2f). LAG = ST-1 serializes, LAG = ST deadlocks (pipe_sim.py, pipe_time.py).
  A register page ring unrolled PD times cost ~30 us at B32 full (PD 6-8, d5bcbc7d), hence the shared ring.
- Consumers (warpgroup per head): QK = 2 x QGMMA m64n64k32 (Q, K in smem); online softmax in registers
  (FP32 MUFU; ex2.approx.f16x2 lowers to two MUFU.EX2.F16 on sm_90, no saving); P x256 to E4M3 packed
  in registers as PV's A operand (no smem, no barrier); QK(j+1) issued before PV(j); O rescale skipped
  when no row max moved in the warp.
- Portion prologue: one batch of global loads (16 query rows, the virtual tile's K/V and cache slots),
  RoPE cos/sin once per token (out-of-line sincosf), the new K/V written to the cache and KN/VN here.
- Combine (PDL, `griddepcontrol.wait`): up to 8 warps per output row, each merging a slice of the row's
  partials 8 at a time (loads in flight together, online max), then RMSNorm, sigmoid gate (`flags` bit 1
  off: RMSNorm only, for the glue lane's gate split), BF16 store. Main CTA 0 writes the partial-slot
  prefix after the partials (`SCHED` floats), so the combine needs no schedule pass; 64 registers, 4 CTAs
  per SM, one wave at every scored shape. Both kernels trigger `launch_dependents` at entry.
- L = 1 or 8 (runtime): rows 8L per head are real, the rest zero queries; fully padded warps skip the
  softmax.

ptxas pitfalls met (sm_90a, nvcc 13): wgmma inside a non-uniform branch -> "WG.AR in divergent path"
(fixed: uniform values via __shfl_sync, peeled loop); a register copy into a live wgmma operand (`pc =
pn`) -> C7513 serialization (fixed: ping-pong P buffers, loop unrolled by 2). Check with
`sm90.sh` (no C75xx lines, QK pair + PV pair issued back to back).

Code-shape traps (first H200 build, `4aa070c6`/`c816b9a7`: correct, slower than FA3/attn_main everywhere):
a load consumed inside a branch (`cond ? ld : 0`, a helper with a split/BF16 branch) is waited on at the
branch, so the page-table ring cost one memory latency per tile and the 16 query loads ran serially;
16 inlined sincosf copies made a 17K-instruction kernel (now 7.4K) whose cold instruction fetch cost
~5-8 us before the first tile (local phase clocks). `VQ8_CLK=1` builds the phase-clock variant
(`vq8.py` `clocks=`, `clock_summary`; `clk_local.py`, `clk_warm.py`); keep the executed path contiguous.

ABI: `vq8.py`: `vq8_attention(data, ...)` (attention problem) and `attention_vq8((P, S), data,
setting=(stages, min_tiles), pdl=False, gate=True)` (decode layer: FP32 projection partials [S, T, W], any split
stride, or BF16 with S == 1; returns [T, HQ*64]).

Verification: `check.py` (all TASK tests for q_len 1 and 8 plus the 12 scored shapes, three schedules,
graph capture + fresh inputs) and `check_partials.py` (decode ABI) pass on sm_120 through the mma.sync
shim (same layouts, USE_WGMMA=0). H200 correctness and timing come only from H200 runs: the attention
candidate (`build_cand.py`: d98ae45d with its `sorted(fa3)` tie crash fixed + vq8 where its import-time lab
beats the incumbent, with phase clocks; `lab_local.py` runs that lab here) and the decode-ABI lab `prof/lab/vq8_layer.py`
(`build_layer_lab.py`; `lab_layer_local.py` smoke-runs it here with FA3 stubbed).
