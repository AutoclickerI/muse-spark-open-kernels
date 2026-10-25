# Retained state audit

`audit.py` runs `tools/local_eval.py` through S9's dispatch audit, recursively inventories the complete retained
state (including adapter objects and closures), and independently decodes every coded dense bank bit-for-bit.
`normal/precision_inventory.json` records all 78 tensors of candidate `7d4216dd`;
`forced/precision_inventory.json` records 118, including Hopper adapter layouts. Both full screens pass
13 tests, 12 scored-shape graph captures, and 23 static checks. Aliases are marked rather than counted as
different encodings. Humming's global weight_scale_2 tensors remain FP32, its weight_scale tensors remain
E4M3, and its int32 words hold packed E2M1 nibbles. CPU int64 kernel configurations are launch metadata.

| State tensors | Contents and precision |
|---|---|
| qkv_gate, o, unpooler, unpooler_slice, qkv3, gate_rows; projection-cell w | Original BF16 values or slices/copies in BF16 |
| pool_router | BF16 pooler followed by BF16 router hi and lo; router combines in FP32, the permitted router representation |
| coded_dense.*.coded | Lossless 12-bit BF16 sign/mantissa/exponent-window codes, packed uint8 |
| coded_dense.*.scales, base | One exact power-of-two FP32 window scale and int32 base per 16 rows, encoding the original BF16 exponent window; no expanded per-weight values |
| coded_dense.*.esc, eoff | Int32 coordinates plus original BF16 16-bit patterns for escapes; int32 per-tile offsets. Decode is bit-exact against original BF16 weights |
| moe_bank{,_raw}.w13p/w2p; s4_nt1_bank.banks.lop3.* | Packed E2M1 nibbles plus raw or losslessly recoded E4M3 scale bytes, 9 bytes per 16 weights |
| s5_hh_base.w13/w2 | Packed E2M1 nibbles in int32 words; gate/up row interleave and wide-bank permutation preserve original codes |
| s5_hh_base.w13_sf/w2_sf | uint8 E4M3 scale-code recode/permutation, 1 byte per 16 weights; no widened scales |
| moe6_adapters.raw17.blob13/blob2, sfb13/sfb2 | Packed E2M1 codes and byte E4M3 scales in the raw17 layout, respectively; no dequantized prepare bank |
| moe6_adapters.w13p/w2p (Hopper) | Same packed E2M1/E4M3 representation for timed expansion inside the DeepGEMM path |
| moe6_adapters.humming (Hopper) | Original NVFP4 packed codes and original byte scale permutation; forced screen verifies raw scale-byte histograms |
| gs13/gs2, w13_gs/w2_gs, gs13_gate/gs13_up | Original FP32 global expert scales; duplication does not change precision |
| pnw, post_expert{,_bf16}, post_expert_w | BF16 post-expert norm factors as consumed by the existing BF16 boundary |
| pre_expert{,_w}, pre_ffn_w, post_ffn_w, next_pre_attn_w | FP32 norm factors (1 + original weights), unchanged S11 arithmetic |
| router_bias, alpha1/beta1/alpha2/beta2, inv_freq | FP32 router bias, derived residual coefficients and RoPE frequencies; no expert weight expansion |
| ones, cnt, smallm_counters; projection-cell out | Constant FP32 ones, int32 synchronization counters, and BF16 output scratch overwritten during each timed call |

KV remains the input FP8 cache. Activations and intermediate buffers follow S11's BF16 boundaries; GEMV partials
and accumulations are FP32. F4 assigns disjoint output-channel slices to four CTAs with the same per-channel
arithmetic and reduction order. Co-residency is excluded after its H200 loss.
