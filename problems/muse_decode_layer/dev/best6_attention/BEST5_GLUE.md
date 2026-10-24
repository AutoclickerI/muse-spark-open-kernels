# Best5 non-compute kernel inventory

From H200 receipt `691bff0bcff44e99`. Every profile row outside projections, attention and MoE is listed below. Times are aggregate GPU durations in µs per layer; PDL waits can overlap producer compute. Two boundary calls share one aggregate row. Names truncated by the organizer are kept verbatim.

## decode 1 sliding

| Kernel | Calls | Aggregate us | Category |
|---|---:|---:|---|
| `_boundary_kernel` | 2 | 11.7 | glue |
| `_route_norm_kernel` | 1 | 3.5 | glue |
| `void at::native::vectorized_elementwise_kernel<4, at::nat...` | 1 | 0.8 | misc |

## decode 1 full

| Kernel | Calls | Aggregate us | Category |
|---|---:|---:|---|
| `_boundary_kernel` | 2 | 12.1 | glue |
| `_route_norm_kernel` | 1 | 3.7 | glue |
| `void at::native::vectorized_elementwise_kernel<4, at::nat...` | 1 | 0.7 | misc |

## decode 16 sliding

| Kernel | Calls | Aggregate us | Category |
|---|---:|---:|---|
| `_boundary_kernel` | 2 | 8.7 | glue |
| `_route_norm_kernel` | 1 | 3.9 | glue |

## decode 16 full

| Kernel | Calls | Aggregate us | Category |
|---|---:|---:|---|
| `_boundary_kernel` | 2 | 8.8 | glue |
| `void at::native::reduce_kernel<128, 4, at::native::Reduce...` | 1 | 4.1 | misc |
| `_route_norm_kernel` | 1 | 3.8 | glue |
| `void at::native::vectorized_elementwise_kernel<8, at::nat...` | 1 | 1.1 | misc |

## decode 32 sliding

| Kernel | Calls | Aggregate us | Category |
|---|---:|---:|---|
| `_boundary_kernel` | 2 | 8.7 | glue |
| `_route_norm_kernel` | 1 | 6.0 | glue |

## decode 32 full

| Kernel | Calls | Aggregate us | Category |
|---|---:|---:|---|
| `_boundary_kernel` | 2 | 8.8 | glue |
| `_route_norm_kernel` | 1 | 6.1 | glue |

## verify 1 sliding

| Kernel | Calls | Aggregate us | Category |
|---|---:|---:|---|
| `_boundary_kernel` | 2 | 13.8 | glue |
| `_route_norm_kernel` | 1 | 3.9 | glue |
| `_qk_prep_kernel` | 1 | 2.8 | glue |
| `void flash::prepare_varlen_num_blocks_kernel<1, false>(in...` | 1 | 2.0 | scheduler |
| `_norm_gate_kernel` | 1 | 1.1 | glue |
| `void at::native::vectorized_elementwise_kernel<4, at::nat...` | 1 | 0.7 | misc |

## verify 1 full

| Kernel | Calls | Aggregate us | Category |
|---|---:|---:|---|
| `_boundary_kernel` | 2 | 13.0 | glue |
| `_route_norm_kernel` | 1 | 4.0 | glue |
| `void flash::prepare_varlen_num_blocks_kernel<1, true>(int...` | 1 | 3.0 | scheduler |
| `_qk_prep_kernel` | 1 | 1.6 | glue |
| `_norm_gate_kernel` | 1 | 1.1 | glue |
| `void at::native::vectorized_elementwise_kernel<4, at::nat...` | 1 | 0.7 | misc |

## verify 16 sliding

| Kernel | Calls | Aggregate us | Category |
|---|---:|---:|---|
| `_boundary_kernel` | 2 | 11.6 | glue |
| `_route_norm_kernel` | 1 | 4.3 | glue |
| `_qk_prep_kernel` | 1 | 3.3 | glue |
| `void flash::prepare_varlen_num_blocks_kernel<1, false>(in...` | 1 | 2.2 | scheduler |
| `_norm_gate_kernel` | 1 | 1.3 | glue |

## verify 16 full

| Kernel | Calls | Aggregate us | Category |
|---|---:|---:|---|
| `_boundary_kernel` | 2 | 11.8 | glue |
| `_route_norm_kernel` | 1 | 4.4 | glue |
| `void flash::prepare_varlen_num_blocks_kernel<1, true>(int...` | 1 | 3.1 | scheduler |
| `_qk_prep_kernel` | 1 | 2.0 | glue |
| `_norm_gate_kernel` | 1 | 1.7 | glue |

## verify 32 sliding

| Kernel | Calls | Aggregate us | Category |
|---|---:|---:|---|
| `_boundary_kernel` | 2 | 19.2 | glue |
| `_route_norm_kernel` | 1 | 6.0 | glue |
| `_qk_prep_kernel` | 1 | 3.4 | glue |
| `void flash::prepare_varlen_num_blocks_kernel<2, false>(in...` | 1 | 2.1 | scheduler |
| `_norm_gate_kernel` | 1 | 1.5 | glue |

## verify 32 full

| Kernel | Calls | Aggregate us | Category |
|---|---:|---:|---|
| `_boundary_kernel` | 2 | 18.7 | glue |
| `_route_norm_kernel` | 1 | 6.2 | glue |
| `void flash::prepare_varlen_num_blocks_kernel<2, true>(int...` | 1 | 3.5 | scheduler |
| `_qk_prep_kernel` | 1 | 2.1 | glue |
| `_norm_gate_kernel` | 1 | 1.9 | glue |
