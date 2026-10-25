# qkv -> attention front (P7 glue)

S6/S7's attn_main rows (decode 1 sliding/full, decode 16 full) turn the qkv projection's FP32 split
partials into BF16 with `partial.sum(0).to(bfloat16)`: two torch kernels on the critical path (H200
`fecc29af` profile: reduce 3.7 + elementwise 1.0 us at decode 1, 4.2 + 1.3 at decode 16 full, no PDL).

- `patch_attn_front.py`: attn_main reads the projection's (P, S) output (`qkv_pair`) and launches with
  programmatic stream serialization: page table, positions, RoPE angles and the first STAGES-1 K/V tiles
  before `griddepcontrol.wait`, every projection read after it.
- H200 `aab93d7380654dda` (v1 lab): summing the S=8 partials inside attn_main LOSES (bit-exact, but its
  per-call split loop serializes L2 round trips in the prologue): base 32.30 / 41.06 / 130.14 us vs
  38.11 / 44.35 / 132.70 (b1 sliding / b1 full / b16 full); the PDL launch recovers 2-5 us (35.94 / 42.32 /
  127.78; + early release 35.78 / 41.54 / 127.04). Local sm_120 had ranked it a win.
- v2 lab (`lab_body.py` + `build.py` -> `prof/lab/attn_front.py`): keep a BF16 attention input and change
  the producer. H200 `87cd3a84a4844c3e` (lab_cold_l2; all arms bit-exact vs S6), paired us vs base at
  b1 sliding / b1 full / b16 full: `fuse` (one Triton sum+cast kernel, PDL behind the GEMV; attn_main launched
  normally) -4.21 / -4.46 / -5.25; `fusep` (+ PDL attn_main) -0.11 / -2.27 / -4.74; `fix` (GEMV with an in-kernel
  split-K fixup to BF16 + PDL attn_main) +0.34 / -2.69 / -5.86; `fixt` (+ early release) -0.22 / -2.56 / -5.55.
  Layer ext at b16 sliding: `laysum` -1.52, `layfix` -0.80. PDL-launching attn_main behind the producer costs
  ~2-4 us at b1 (inferred: its CTAs sit resident while the GEMV still streams). Local sm_120 had ranked
  fusep/fix best (`local_lab.log`).
- `patch_front_bf16.py <cand> <out>`: the winner, `fuse` + `laysum`: `_attn_port_raw` and the layer ext's FP32
  multi-split input go through `_attn_sumcast_kernel`. On S8: `cand_r3_best7_s8_front.py`, local 13/13 tests,
  12/12 timed shapes checked ok, 24 static checks (`s8_front_local.log`; its timings ran on a shared GPU).
