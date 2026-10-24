# Prefill glue: direct H200 inventory

Named-kernel times from board-best a30fb411b0c74ffb and99820fa6e5c14fe1.
Each column is the official multiplicity-weighted mean us per layer call for that load.
Multiply by62/62/186 to get full-load us. Reports contain only12 kernels: absent is N/A.
Byte floors use4.8TB/s and compulsory activation read/write traffic; cached vectors/metadata
are omitted where small. They do not model SFU/reduction/atomic work or launch latency.
For expert-dependent rows, R local pairs is bounded by T..8T; touched experts E by1..32.
Those ranges are bounds, not an inferred routing count. Separate dequant is counted once.
Router uses sigmoid plus top8 weight renormalization, not softmax.

| Kernel | turn1 us | floor | turn4 us | floor | first-turn us | floor |
|---|---:|---:|---:|---:|---:|---:|
| _boundary_kernel | 41.53–42.75 | 30.84 | 163.45–164.72 | 123.37 | 851.67–855.25 | 671.09 |
| _canonical_expand_coalesced | N/A | 0.00 | N/A | 0.00 | 633.59–651.27 | 26.87–859.83 |
| _portfolio_swiglu | N/A | 0.00 | N/A | 0.00 | 245.49–245.90 | 83.89–671.09 |
| _swiglu_kernel | 14.75–14.77 | 3.86–30.84 | 44.03–44.91 | 15.42–123.37 | N/A | 0.00 |
| _combine_kernel | 10.27–10.30 | 2.57–11.57 | 28.50–29.76 | 10.28–46.26 | 133.74–133.95 | 55.92–251.66 |
| _gather_norm_kernel | 6.67–6.67 | 2.57–11.57 | N/A | 10.28–46.26 | 97.81–99.52 | 55.92–251.66 |
| _routing_kernel | N/A | 0.34–0.35 | N/A | 1.37–1.41 | 42.03–42.03 | 7.45–7.65 |
| _pre_expert_norm_kernel | N/A | 0.00 | N/A | 0.00 | N/A | 0.00 |
| _norm_gate_kernel | N/A | 0.96 | N/A | 3.86 | N/A | 20.97 |
| _qk_prep_kernel | N/A | 0.65 | N/A | 2.58 | N/A | 14.04 |
| _offsets_kernel | N/A | 0.00 | N/A | 0.00 | N/A | 0.00 |
| _gather_cache_kernel | N/A | 8.61 | N/A | 34.45 | N/A | 5.24 |

The two boundaries move24*T*8192 bytes combined: at16K,3.221GB and671.09us.
A single16Kx8192 BF16 read OR write is268.44MB /55.92us; a read+write is111.85us.
Boundary time is about159ms per first turn, with a125ms traffic floor (~34ms ideal headroom).
The earlier235ms remainder comes from a different run/stage accounting; this inventory uses
the named kernels in the current best and does not subtract separately-counted dequant twice.
Detailed per-receipt/per-profile values: runs/screen/glue_prefill/inventory.json.
