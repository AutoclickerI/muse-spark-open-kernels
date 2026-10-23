# One resident NVFP4 expert bank for serving
Serves: G1, G2. Read before porting an expert kernel into serving: the one-bank NVFP4 layout and per-GPU memory.

## Recommendation

Make SGLang's existing Marlin-repacked NVFP4 bank the single resident expert bank, and
teach the two custom readers to consume that bank directly. Keep the original FP32 global
scales as a tiny sidecar captured before Marlin processing: the checkpoint has separate
gate and up scales, while the kernel challenge inputs simplify W13 to one scale per expert.
This keeps the existing Marlin path as the no-conversion fallback and avoids a second
54.36 GB bank. It is a recommendation to prototype, not a measured performance result.

There is no zero-change layout shared by the current kernels. V6's 36 KiB tiles and v3's
WGMMA blobs are different lossless permutations, and Marlin has a third permutation. A
single bank can feed all three only after adding readers for the selected layout (or changing
Marlin). Keeping Marlin as the canonical bank minimizes fallback and dispatch changes, but
the custom readers lose their current contiguous prepared-tile loads. If H200 measurements
show that cost erases the gains, the alternative is a custom canonical bank plus a new
Marlin-layout reader; that also needs substantial kernel work and does not preserve the
stock Marlin fallback for free.

The reason to pursue both readers is the modeled serving impact: PROFILE projects the
isolated V6 MoE ratio to +9.8% serving score and a whole-layer prefill v3 port to +26.4%
(score multipliers 1.098 and 1.264); these are model sensitivities, not serving measurements.
The kernel-track H200 results are V6 1.4916x overall (verify 1.635x / 1.405x / 1.211x)
and prefill v3 1.3454x overall.
The combined non-overlapping model is 1.437x. Prefill v3's job notes say its 16K-token path
loses to dequantization plus DeepGEMM, so keep a large-token path that dequantizes from the
same resident bank into temporary BF16 workspace rather than adding another resident bank.
Sources: `docs/reference/profile.md:73-82,120-126,163-183`; `problems/muse_prefill_layer/dev/LIVE_STATUS.md:38`;
`problems/muse_prefill_layer/PORTING.md:11-12`; `problems/muse_prefill_layer/dev/cand_v12_alltest.py:489-520`.

## Existing physical layouts

All three start from the same logical tensors: packed E2M1 codes, one E4M3 scale for every
16 weights, and FP32 matrix global scales. The conversions below do not expand weights to a
resident BF16 matrix.

| Bank | Codes | Group scales | Global scales and logical tiling |
|---|---|---|---|
| MoE `cand_v6.py` | `prepare()` expands low/high nibbles, then bit-permutes code bits into `[expert, 16-row tile, 32768 B]`. Each tile is 16 rows by K=4096. W13 maps each tile to 8 gate rows followed by the matching 8 up rows; W2 uses natural groups of 16 rows. | The 4096 E4M3 bytes per tile are represented by one byte each, but recoded as an exact E5M3-with-offset magnitude. Negative scale signs move into the E2M1 sign bits, and zero-scale groups zero their codes. Physical scale order is `[tile, w, (rr,q), g, ri]`. The kernel rebuilds BF16 scale bits and multiplies them into the code-derived BF16 values. | `gs13` and `gs2` remain separate FP32 per-expert arrays. V6's `gs13` is one scalar for both gate and up, matching the challenge input contract; both globals are multiplied by 256 in the epilogue to compensate for the code/scale exponent encoding. Tile sizes are 36,864 B: 32,768 code bytes plus 4,096 scale bytes. Shapes are `[32,512,36864]` W13 and `[32,256,36864]` W2. `cand_v6.py:548-639`, especially `:574-613` and `:616-639`; loader and scale application at `:333-357`, `:371-400`. |
| Prefill `cand_v3.py` | Lossless quack/WGMMA fragment permutation into `wblob [E,NFB,KT,4,128,16]`, one thread's 16 code bytes for a 64-row fragment and K=64 chunk. With K=4096, `KT=64`; `NFB=32` for W13 and 16 for W2. Codes retain their E2M1 signs. W13's 256-row CTA order is 64 gate, 64 up, 64 gate, 64 up; W2 rows stay natural. | Original E4M3 bytes are reinterpreted as uint8 and permuted separately into `sfblob [E,NFB,KT,4,32,4,2]`; each pair is the scale for rows `r` and `r+8` in a K16 group. | `w13_gs` and `w2_gs` remain FP32 per-expert arrays, applied after grouped GEMM accumulation. `prepare()` builds `blob13/sfb13` and `blob2/sfb2`; `prepare_base()` also retains raw tensor references and tries to create a Marlin bank, so the standalone candidate's state is not a one-bank serving design. `cand_v3.py:378-416`, `:590-644`, `:754-760`; fragment loads and scale conversion in the embedded CUDA source around `:577` onward. |
| SGLang Marlin | `_repack_moe_fp4_weight_for_marlin` views each expert's packed bytes as int32, transposes K/N, then calls `gptq_marlin_repack` with 4-bit codes and Marlin's tile permutation. The output int32 shape is `[K/16, 2*N]` per expert, i.e. `[256,16384]` for W13 and `[256,8192]` for W2; each int32 carries eight E2M1 nibbles. | Group scales are transposed, then `marlin_permute_scales` reshapes into 64-scale groups and permutes each group by `scale_perm = [i + 8*j for i=0..7, j=0..7]`, before restoring `[K/16,N]` count. `nvfp4_marlin_process_scales` converts E4M3 to FP16, reorders each four-scale group `[0,2,1,3]`, multiplies by 128, bit-shifts/reinterprets as E4M3 and selects odd bytes. This preserves scale values while matching Marlin's expected encoded layout. | `nvfp4_marlin_process_global_scale` casts to activation dtype then multiplies by `2^(exponent_bias-7)`; with BF16 activations this is `2^119`. The model's Marlin path may collapse W13 to its gate global scale because its post-expert RMSNorm cancels the per-expert gate/up ratio. Keep the checkpoint's original `[expert, gate/up]` FP32 W13 scales separately for the custom readers. Relevant code: `marlin_utils_fp4.py:29-59`, `:300-343`, `:468-568`; scale order in `marlin_utils.py:301-321`; output shape in `gptq_marlin_repack.py:34-45`; repack tile and nibble order in `gptq_marlin_repack.cuh:144-225`; serving parameters and Marlin dispatch: `muse_spark_nvfp4_moe.py:133-159`, `muse_spark_fp4_moe.py:586-630`, and Marlin GEMM calls in `moe_runner/marlin.py:513-540`. |

For v6, each expert's W13 36 KiB tile contains gate/up rows already paired for its SwiGLU epilogue. For v3, WGMMA consumes 256-row/64-K code fragments and separate row-paired scales. Neither physical order can be passed to the other's current pointer arithmetic. Marlin's `qweight` and scales likewise cannot be passed to either current prepared-tile reader.

Marlin's code-bank order is explicit in the checked-in repacker: K is tiled by 16, N by 64,
tiles are stored K-tile-major then N-tile-major, and each 16x64 tile becomes 128 int32 words.
For 4-bit codes the pack order is `[0,2,4,6,1,3,5,7]` within each word; the thread/lane
mapping selects the four K positions and two N positions shown in the repack source. The
NVFP4 call passes an empty K permutation, so source K indices are sequential. Evidence:
`marlin.cuh:23-32`, `gptq_marlin_repack.cuh:52-63,102-139,144-225`, and
`marlin_utils_fp4.py:491-493`.

## One-bank compatibility and required kernel work

The canonical choice above is the actual Marlin repack. At the value level it is lossless for
the E2M1 codes and E4M3 group scales. Preserve raw FP32 W13 gate/up and W2 globals in a
sidecar before Marlin's conversion: Marlin's BF16 global-scale recoding is not an injective
encoding of arbitrary FP32 values, and the custom kernels need the distinct real-checkpoint
gate/up values. The sidecar is only 23,040 B per GPU across 60 quantized layers: 15,360 B
for W13 and 7,680 B for W2. These totals are 8 B/expert/layer for W13 and 4 B for W2; the
sidecar is negligible next to a bank.
The checkpoint's two W13 scales and the challenge simplification are documented in
`problems/muse_prefill_layer/PORTING.md:17` and `muse_spark_nvfp4_moe.py:133-141`.

| Reader | Change to read the one Marlin bank | Cost and risk |
|---|---|---|
| Decode/verify MoE (`cand_v6.py`) | Replace its `w13p/w2p` 36 KiB-tile pointers and one-bulk-copy producer with a loader that maps Marlin qweight indices into the v6 consumer's code fragments and reads/inverts Marlin's group-scale permutation. Apply separate FP32 gate and up global scales in the SwiGLU epilogue; W2 keeps its own scale. Keep routing, accumulation and norm/combine structure. | The present producer streams each 36 KiB tile contiguously with `cp.async.bulk`; Marlin order is not that tile. A direct reader needs address mapping, non-contiguous/gather loads, and scale decoding in the hot path, or a cooperative in-kernel gather into shared memory. It avoids a resident copy but may lose coalescing and producer overlap. This is the largest speed risk for V6's measured H200 verify gains (1.635x / 1.405x / 1.211x). Evidence: `cand_v6.py:4-23`, `:333-357`, `:616-639`; `docs/reference/profile.md:193-209`. |
| Prefill grouped WGMMA (`cand_v3.py`) | Replace `blob13/sfb13` and `blob2/sfb2` with Marlin qweight/scale pointers. At each `(expert, fb, kt, fragment, lane)`, calculate the Marlin packed-word and scale positions, invert their nibble/scale order into the existing `make_frag` WGMMA operand, and use the FP32 global-scale sidecar. Delete the resident `repack()` products. | Adds Marlin index arithmetic and potentially scattered global loads before each fragment. It removes the v3 repack bank. The row/K tile mismatch is structural: v3 fetches 256-row by 64-K fragments, while Marlin's tile mapping is designed for its own W4A16 kernel. Performance impact is unknown until H200 timing. For 16K tokens, implement the dequant+DeepGEMM path as another reader over the same Marlin bank; its BF16 weights are temporary workspace, not a persistent bank. Evidence: `cand_v3.py:617-644`, `:754-760`; `cand_v12_alltest.py:489-520`; `LIVE_STATUS.md:38`. |
| Marlin fallback | No custom change: SGLang's existing Marlin runner consumes this bank for every token count not dispatched to the custom kernels. Keep its own transformed global scales for its existing numerical behavior, alongside the tiny raw-FP32 sidecar for the custom readers. | This is why Marlin is the recommended canonical layout. The fallback remains available without materializing a second bank or repacking on a request. |

So: one stored layout **can** feed both custom kernels and Marlin without changing weight
precision, but only after the two custom readers are changed. One of the kernels cannot simply
consume the other's current prepared bank. If Marlin must remain unmodified, Marlin's bank is
the only one-bank option that keeps it as a direct fallback. A V6- or v3-canonical design
would need either a new Marlin reader or an additional Marlin bank.

## Per-GPU steady-state memory

Use the model's per-expert local shapes from `sim/muse.py:7-23`: W13 has `2*4096*4096`
weights and W2 has `4096*4096`, or `3*4096^2 = 50,331,648` weights per expert/layer. EP8
stores 32 local experts. Of 62 layers, 60 have NVFP4 routed experts; the two BF16 MoE layers
are excluded from these NVFP4-bank counts. Each NVFP4 weight costs `1/2` byte of E2M1 plus
`1/16` byte of E4M3 block scale = `9/16` byte. Global-scale sidecars add only tens of KB.

```text
NVFP4 parameters per GPU = 3 * 4096^2 * 32 * 60 = 96,636,764,160
one bank                 = 96,636,764,160 * 9/16 = 54,358,179,840 B
                         = 54.36 GB = 50.625 GiB
static budget            = 141 GB * 0.85 = 119.85 GB
KV per cached token      = 2 KV heads * 64 * (K + V) * 62 layers
                         = 15,872 B/token/GPU
```

| Resident NVFP4 banks | Expert bytes/GPU | GiB | Static-budget headroom before all other model memory | Added banks vs one bank, in KV-token equivalents |
|---:|---:|---:|---:|---:|
| 1 | 54.36 GB | 50.625 | 65.49 GB | 0 |
| 2 | 108.72 GB | 101.250 | 11.13 GB | 3.425 million |
| 3 | 163.07 GB | 151.875 | −43.22 GB (over budget) | 6.850 million |

Here, one is the recommended shared bank, two means separate v6 and v3 banks, and three
means v6 plus v3 plus Marlin's bank. The one-bank size is the same whichever exact
permutation is used; layout metadata and the FP32 scale sidecar are negligible.

Each added full bank costs `54,358,179,840 / 15,872 = 3,424,785` cached KV tokens per GPU.
Headroom is only the 0.85 static ceiling minus these NVFP4 expert banks; it still has to
hold dense model weights, the drafter, the two BF16 MoE layers, workspaces, graph pools and
allocator slack. Thus two banks are not shown to fit the serving model, and three do not fit
the static budget even before those allocations. This exact shape-based count is lower than
the rough `816B / 8 * 9/16 = 57.375 GB` estimate in the in-progress `docs/reference/moe_port.md:16-32`
because it counts only the 60 NVFP4 expert layers, not all parameters assigned to an EP8 rank.

The 15,872 B KV figure uses TP8's 2 KV heads/GPU, head dimension 64, FP8 K and V, and 62
layers (`sim/muse.py:11-14,23`). Treat the token equivalents as capacity arithmetic, not a
prediction of actual serving KV allocation; the remaining model allocations reduce it.

## Work order and blockers

1. Change the serving weight-processing path to retain exactly one Marlin qweight/scale bank
   and capture the original FP32 globals before Marlin conversion. Keep W13 gate and up globals
   distinct for custom paths. The current task kernels' one-W13-scale assumption must not leak
   into the real checkpoint port.
2. Implement a small direct Marlin-bank reader in the prefill path and v6 path, first checking
   exact code/scale reconstruction on hand-selected rows and tiles. Do not build full converted
   resident buffers. Keep Marlin fallback available throughout.
3. Route verify shapes to the v6 algorithm and prefill shapes to the v3 grouped algorithm;
   route the 16K prefill shape to dequant+DeepGEMM from the same resident bank. Preserve a
   shape-complete Marlin fallback while all supported shapes are validated.
4. Measure H200 end-to-end serving memory and speed. The deciding evidence is whether the
   Marlin readers preserve the v6 verify ratios and v3 prefill gains, plus whether the 16K
   dequant workspace coexists with the real KV pool and graph allocations.

Blockers to calling this complete: neither custom reader has been implemented or timed against
Marlin's bank; full-node H200 memory and serving score are unmeasured; the real checkpoint's
separate W13 global scales need an end-to-end correctness check; and the large-token temporary
dequant workspace needs peak-memory measurement. `docs/reference/profile.md` and the job notes provide modeled
or kernel-track evidence only. The repo's `sync_upstream.sh` was not run: it invokes `okc init`
and writes official files, conflicting with this task's no-okc / edit-only-this-document limits.
The local official-file diff was clean when inspected.
