# Prefill glue: direct H200 inventory

Named-kernel times from board-best a30fb411b0c74ffb and99820fa6e5c14fe1.
Each column is the official multiplicity-weighted mean us per layer call for that load.
Multiply by62/62/186 to get full-load us. Reports contain only12 kernels: omitted is N/A.
Zero is used only where the receipt-selected path does not execute that kernel.
Byte floors use4.8TB/s and compulsory activation read/write traffic; cached vectors/metadata
are omitted where small. They do not model SFU/reduction/atomic work or launch latency.
For expert-dependent rows, R local pairs is bounded by T..8T; touched experts E by1..32.
Those ranges are bounds, not an inferred routing count. Separate dequant is counted once.
Router uses sigmoid plus top8 weight renormalization, not softmax.

| Kernel | turn1 us | floor | turn4 us | floor | first-turn us | floor |
|---|---:|---:|---:|---:|---:|---:|
| _boundary_kernel | 41.53–42.75 | 30.84 | 163.45–164.72 | 123.37 | 851.67–855.25 | 671.09 |
| _canonical_expand_coalesced | 0.00 | 0.00 | 0.00 | 0.00 | 633.59–651.27 | 26.87–859.83 |
| _portfolio_swiglu | 0.00 | 0.00 | 0.00 | 0.00 | 245.49–245.90 | 83.89–671.09 |
| _swiglu_kernel | 14.75–14.77 | 3.86–30.84 | 44.03–44.91 | 15.42–123.37 | 0.00 | 0.00 |
| _combine_kernel | 10.27–10.30 | 2.57–11.57 | 28.50–29.76 | 10.28–46.26 | 133.74–133.95 | 55.92–251.66 |
| _gather_norm_kernel | 6.67 | 2.57–11.57 | N/A | 10.28–46.26 | 97.81–99.52 | 55.92–251.66 |
| _routing_kernel | N/A | 0.34–0.35 | N/A | 1.37–1.41 | 42.03 | 7.45–7.65 |
| _pre_expert_norm_kernel | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| _norm_gate_kernel | N/A | 0.96 | N/A | 3.86 | N/A | 20.97 |
| _qk_prep_kernel | N/A | 0.65 | N/A | 2.58 | N/A | 14.04 |
| _offsets_kernel | N/A | 0.00 | N/A | 0.00 | N/A | 0.00 |
| _gather_cache_kernel | 2.86–2.94 | 2.22 | N/A | 9.78 | N/A | 2.81 |

Full-load totals, ms (H200 /byte floor):

| Kernel | turn1 | turn4 | first turn |
|---|---:|---:|---:|
| _boundary_kernel | 2.575–2.651 /1.912 | 10.134–10.213 /7.649 | 158.411–159.077 /124.822 |
| _canonical_expand_coalesced | 0.000 /0.000 | 0.000 /0.000 | 117.848–121.137 /4.998–159.929 |
| _portfolio_swiglu | 0.000 /0.000 | 0.000 /0.000 | 45.662–45.737 /15.603–124.822 |
| _swiglu_kernel | 0.915–0.916 /0.239–1.912 | 2.730–2.785 /0.956–7.649 | 0.000 /0.000 |
| _combine_kernel | 0.637–0.638 /0.159–0.717 | 1.767–1.845 /0.637–2.868 | 24.875–24.915 /10.402–46.808 |
| _gather_norm_kernel | 0.414 /0.159–0.717 | N/A /0.637–2.868 | 18.193–18.511 /10.402–46.808 |
| _routing_kernel | N/A /0.021–0.022 | N/A /0.085–0.087 | 7.818 /1.387–1.422 |
| _pre_expert_norm_kernel | 0.000 /0.000 | 0.000 /0.000 | 0.000 /0.000 |
| _norm_gate_kernel | N/A /0.060 | N/A /0.239 | N/A /3.901 |
| _qk_prep_kernel | N/A /0.040 | N/A /0.160 | N/A /2.611 |
| _offsets_kernel | N/A /0.000 | N/A /0.000 | N/A /0.000 |
| _gather_cache_kernel | 0.178–0.182 /0.138 | N/A /0.607 | N/A /0.523 |

The two boundaries move24*T*8192 bytes combined: at16K,3.221GB and671.09us.
A single16Kx8192 BF16 read OR write is268.44MB /55.92us; a read+write is111.85us.
Boundary time is about159ms per first turn, with a125ms traffic floor (~34ms ideal headroom).
The earlier235ms remainder comes from a different run/stage accounting; this inventory uses
the named kernels in the current best and does not subtract separately-counted dequant twice.
Detailed per-receipt/per-profile values: runs/screen/glue_prefill/inventory.json.

Byte formulas: boundaries24TH; separate dequant2.5625E(3D²); SwiGLU6RD;
combine/gather2D(R+T); router2176T+8R; attention-gate6T*1024; QK-prep4112T.
H=8192,D=4096. Cache gather uses the actual selected sliding/full capacity rule,
rounded to128 rows;753-token sliding uses FA3 directly and has no gather.
Candidate, interface and verification: [glue README](../../problems/muse_prefill_layer/dev/glue/README.md).
