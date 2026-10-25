# vq8: paged FP8 attention for q_len 8 (and 1), native FP8 WGMMA

## P5 takeover: current result and integration gate

Target remains attention inside the decode layer: sliding <= 6 us at every load, full within ~1.3x of its KV
floor. Neither is achieved. Frozen integration base is `cand_r3_best7_s11_final.py` **acda6c62**.

The promising change is `partial_port.py`: S11's cluster attention kernel reads the QKV projection's FP32
partials directly, sums and rounds at the original BF16 boundary, gathers cached KV before the PDL wait, then
normalizes Q/K, applies RoPE, writes the new FP8 KV, attends every key, merges its splits and applies norm/gate.
One attention launch; no sumcast or metadata launch. Full attention retains the four-stage gather. The layer
adapter supplies the layer's fixed head counts and derives gating from the projection width (including padded
strides), so qkv3 from gate split remains ungated until `_gate_split_apply`.

`patch_partial_port.py` applies only onto that frozen base, with decode rows only and exact sincos. Its current
experimental output is `out/s11_partial_port.py` **e8a6828d**. Do not promote it from the stand-in chain results:
`layer_lab.py.inc` / `build_whole_layer.py` compare actual whole-layer S11 and the patch over 16 identical seeds,
8 cold replays per arm, and an A:A noise graph. H200 job **c114c35e787f46d1** landed: decode-1 sliding/full whole-layer ratios **0.96960 / 0.97415**, 16 seeds, all diagnostic gates pass. The default integration patch now selects only those two rows. Its measured all-decode host passed official correctness and timed at 1.72965x; that score is not the promotion criterion.

Local evidence: the candidate passes normal AND forced `local_eval` through `dev/s9/audit.py`: each has
13 tests, 12 fresh-input graph shapes and 24 static checks (all CUDA strings compile for sm90a). The lab's own
A/A2/PP captures and namespace bindings run locally on two inputs. Those local timings are not H200 evidence.
The port also passes attention TASK tests, page/scale/DP cases, and strided FP32 projection-partial graphs.
Logs and screen JSON are under ignored `runs/screen/p5_takeover/` and `out/port_{normal,forced}/`.

### H200 receipts

Ratios below are **variant / control**, lower wins. The component labs time a 37.7 MB projection stand-in plus
attention, cold L2 by read, alternating AB/BA x8 on three seeds. `us` subtracts the same stand-in median;
reported paired ratios include the entire chain. Controls are S11's attn_main for decode, public FA3 for sliding
verify/B1 full verify, and S11's frozen c262e4c5 vq8 for B16/32 full verify. Only whole-layer ratios promote.

| Job | Experiment | Landed result |
|---|---|---|
| `5747e98730e849cd` | inherited fused grid merge vs spill-free CH8 combine | Fused loses sliding: B1/16/32 decode chain ratios 1.069/1.208/1.166; verify .974/1.115/1.126. Decode full B16/32 stays 1.245/1.228 vs attn_main. Separate combine at verify-1 sliding wins .947 (14.21 vs 16.10 us), but this does not establish a whole-layer win. Full verify clocks 1,768/1,700 cycles per tile. |
| `8fd0d2a6dd9e4293` | cooperative four-lane Q/K prep, repeated RoPE | Loses: decode sliding c4 1.117/1.351/1.261; verify 1.068/1.194/1.130. Full verify c6 1.050/1.031. Q-ready moves from ~6 to ~10-13 us on H200 despite a local improvement. |
| `9aff77cc54bf471b` (`attn_partial_port`, merged with P10 staging) | direct FP32 projection partials into attn_main | Exact port wins every decode shape: sliding 8.61/6.90/10.96 vs 10.24/7.97/12.00 us; chain ratios .9630/.9786/.9677. Full 18.99/96.93/188.14 vs 20.05/99.78/190.46 us; .9788/.9788/.9878. All gates pass. Verify loses (roughly 1.22-1.57 chain ratios), excluded from patch. Fast sincos adds a small B1 sliding win but is not selected. |
| `8b8e4e2bf1d4409d` (`vq8_softmax`) | native RS-WGMMA balanced sum tree + skip exp2(0), stages 4/6/8 | c6 full verify B16/32 141.33/249.15 vs 140.61/247.87 us; 1.0050/1.0044, no win. Clock softmax ~890/899 cycles (was ~921), total 1,732/1,682. Decode full remains ~1.25x attn_main. |
| same job (`vq8_fastprep2`) | shared RoPE, reduced KV prep duplication, compact sliding grid | Still loses. Sliding decode c4 1.089/1.279/1.223, verify 1.002/1.130/1.081. Full verify c6 1.040/1.024. Fused merge also loses. All gates pass; 140 registers, no spills on sm90a. |
| `6d555796786a452e` (`attn_partial_stages`) | sliding-only S2/S3/S4 rings, identical full S4 controls | All gates pass. S3 sliding B1/16/32 9.33/6.61/10.18 us; S2 9.81/7.10/9.73; S4 9.66/6.80/10.70. These small changes do not meet the structural target; no promotion. |
| `c114c35e787f46d1` (`attention_p5_layer`) | exact direct-partials port vs final S11, whole layer | Decode-1 S/F .96960/.97415 (-2.25/-2.21 us), A:A .99941/1.00235, 16 seeds; both win. B16 S/F 1.00229/.99535; B32 S/F .99629/1.00267. Select only decode-1 for the immediate handoff. Official host passes, 1.72965x. |

The first cooperative preparation variant's exact source is preserved in its run receipt and generated payload
`prof/lab/vq8_fastprep.py`; `fastprep.cu.inc` now contains the revised shared-RoPE version. A partial-warp shuffle
hang found locally in its first version was fixed with the participating-lane mask before any H200 send. Both
versions passed fresh graph/cache checks. The H200 regression's exact cause is not isolated: the first changes
increase repeated KV loads and trigonometry, and the revision reduces both but still loses.

### Build / next decision

- Base repaired lab: `build_layer_lab.py`; `--softmax` selects native softmax, `--fastprep --compact` selects
  revised preparation and a compact sliding grid. `build_partial_lab.py` builds the direct-partials comparator;
  `build_stages_lab.py` builds its sliding ring study. `lab_layer_local.py <budget> <payload>` executes all glue
  locally, substituting only the unavailable FA3 binary; its FA3 output gates fail intentionally.
- `build_softmax.py` starts from the **frozen incumbent** c262e4c5, so the full-verify comparator is genuinely S11.
  Native vq8 already uses FP32 ex2, LDSM.T+PRMT V transpose and register P for RS WGMMA. The balanced sum experiment
  shaves ~20-30 softmax cycles without a paired time win; it does not close the 1,400-cycle model gap.
- `check.py` and `check_partials.py` now exit nonzero on a failed gate. `sm90.sh` fails on nvcc errors or C75xx;
  the previous pipeline could hide a failing nvcc exit status. CH16 combine spilled; reverting its chunk to CH8
  removes those spills. Fused merge is experimental and rejected for integration.
- Structural direction: one sliding launch with a CTA-local or cluster/DSM split reduction, targeting <=6 us at decode 1 and <=8-10 us elsewhere. Judge its critical-path increment with the S11 timeline lab. Full verify still needs a structural change; keep c262e4c5 there until a measured win. The incremental preparation/ring variants above are closed.


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

Verification: `check.py` covers the attention TASK cases and all 12 scored shapes; `check_partials.py` covers the
projection-partials ABI, and both return nonzero on a failed gate. `check_partial_port.py` exercises the new direct
partials port, including `--partials`, `--fast` and `--stages 2/3`. The local shim checks layouts and glue; H200
receipts above decide speed. The current lab builders and the exact final-S11 comparison are listed above.
