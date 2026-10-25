| Load/layer | Calls | Kernel sum us | Stream floor us | Sum excess us | Adjusted layer us | Critical-path ceiling us |
|---|---:|---:|---:|---:|---:|---:|
| decode 1/sliding | 10 | 105.00 | 58.26 | 46.74 | 105.18 | 46.92 |
| decode 1/full | 10 | 120.00 | 63.89 | 56.11 | 115.22 | 51.33 |
| verify 1/sliding | 15 | 160.90 | 97.58 | 63.32 | 149.99 | 52.41 |
| verify 1/full | 15 | 181.80 | 103.21 | 78.59 | 172.58 | 69.37 |

**decode 1, sliding**

| Kernel (same-name calls grouped) | Calls | us | Useful bytes | TB/s | Excess us |
|---|---:|---:|---:|---:|---:|
| `void (anonymous namespace)::moe_fused<1>(__nv_bfloat16 co...` | 1 | 34.80 | ~70,787,092 | 2.034 | 15.14 |
| `_gemv_kernel` | 2 | 26.50 | 83,886,080 | 3.166 | 3.20 |
| `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT` | 1 | 12.70 | 37,748,736 | 2.972 | 2.21 |
| `void (anonymous namespace)::attn_main<1, 4>((anonymous na...` | 1 | 9.90 | 532,484 | 0.054 | 9.75 |
| `_boundary_fast` | 2 | 7.50 | 393,216 | 0.052 | 7.39 |
| `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_TNT` | 1 | 7.30 | 16,777,216 | 2.298 | 2.64 |
| `_route_fast` | 1 | 4.50 | 44,128 | 0.010 | 4.49 |
| `void cublasLt::splitKreduce_kernel<32, 16, int, float, __...` | 1 | 1.80 | ~23,040 | 0.013 | 1.79 |

**decode 1, full**

| Kernel (same-name calls grouped) | Calls | us | Useful bytes | TB/s | Excess us |
|---|---:|---:|---:|---:|---:|
| `void (anonymous namespace)::moe_fused<1>(__nv_bfloat16 co...` | 1 | 43.40 | ~70,787,092 | 1.631 | 23.74 |
| `_gemv_kernel` | 2 | 26.40 | 83,886,080 | 3.178 | 3.10 |
| `void (anonymous namespace)::attn_main<1, 4>((anonymous na...` | 1 | 18.30 | 20,800,264 | 1.137 | 12.52 |
| `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT` | 1 | 12.30 | 37,748,736 | 3.069 | 1.81 |
| `_boundary_fast` | 2 | 6.90 | 393,216 | 0.057 | 6.79 |
| `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_TNT` | 1 | 6.80 | 16,777,216 | 2.467 | 2.14 |
| `_route_fast` | 1 | 4.10 | 44,128 | 0.011 | 4.09 |
| `void cublasLt::splitKreduce_kernel<32, 16, int, float, __...` | 1 | 1.80 | ~23,040 | 0.013 | 1.79 |

**verify 1, sliding**

| Kernel (same-name calls grouped) | Calls | us | Useful bytes | TB/s | Excess us |
|---|---:|---:|---:|---:|---:|
| `void (anonymous namespace)::moe_fused<1>(__nv_bfloat16 co...` | 1 | 78.20 | ~212,344,892 | 2.715 | 19.22 |
| `_gemv_kernel` | 1 | 22.70 | 75,497,472 | 3.326 | 1.73 |
| `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT` | 2 | 18.00 | 46,137,344 | 2.563 | 5.18 |
| `void cutlass::device_kernel<flash::enable_sm90_or_later<f...` | 1 | 10.00 | 534,304 | 0.053 | 9.85 |
| `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_TNT` | 1 | 7.10 | 16,777,216 | 2.363 | 2.44 |
| `_boundary_fast` | 2 | 6.80 | 3,145,728 | 0.463 | 5.93 |
| `_route_fast` | 1 | 4.20 | 353,024 | 0.084 | 4.10 |
| `void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu...` | 1 | 4.20 | ~82,944 | 0.020 | 4.18 |
| `void cublasLt::splitKreduce_kernel<32, 16, int, float, __...` | 2 | 3.60 | ~266,240 | 0.074 | 3.53 |
| `_qk_prep_kernel` | 1 | 2.70 | 116,864 | 0.043 | 2.67 |
| `void flash::prepare_varlen_num_blocks_kernel<1, false>(in...` | 1 | 2.20 | unknown | unknown | unknown |
| `_norm_gate_kernel` | 1 | 1.20 | 65,536 | 0.055 | 1.18 |

**verify 1, full**

| Kernel (same-name calls grouped) | Calls | us | Useful bytes | TB/s | Excess us |
|---|---:|---:|---:|---:|---:|
| `void (anonymous namespace)::moe_fused<1>(__nv_bfloat16 co...` | 1 | 86.80 | ~212,344,892 | 2.446 | 27.82 |
| `_gemv_kernel` | 1 | 22.90 | 75,497,472 | 3.297 | 1.93 |
| `void cutlass::device_kernel<flash::enable_sm90_or_later<f...` | 1 | 18.60 | 20,802,084 | 1.118 | 12.82 |
| `nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT` | 2 | 18.40 | 46,137,344 | 2.507 | 5.58 |
| `_boundary_fast` | 2 | 7.30 | 3,145,728 | 0.431 | 6.43 |
| `nvjet_sm90_tst_64x8_64x16_2x1_v_bz_TNT` | 1 | 7.20 | 16,777,216 | 2.330 | 2.54 |
| `void cutlass::device_kernel<flash::FlashAttnFwdCombine<cu...` | 1 | 7.10 | ~82,944 | 0.012 | 7.08 |
| `_route_fast` | 1 | 4.40 | 353,024 | 0.080 | 4.30 |
| `void cublasLt::splitKreduce_kernel<32, 16, int, float, __...` | 2 | 3.50 | ~266,240 | 0.076 | 3.43 |
| `void flash::prepare_varlen_num_blocks_kernel<1, true>(int...` | 1 | 3.20 | unknown | unknown | unknown |
| `_qk_prep_kernel` | 1 | 1.30 | 116,864 | 0.090 | 1.27 |
| `_norm_gate_kernel` | 1 | 1.10 | 65,536 | 0.060 | 1.08 |

Estimated whole-board gain: 1.736%; uniform saving needed per small-M row: 10.796us.
