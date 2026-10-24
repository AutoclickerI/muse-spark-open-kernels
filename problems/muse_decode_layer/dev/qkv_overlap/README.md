# QKV / attention KV-group overlap ceiling

Two KV heads give two independent groups of eight Q/gate heads. For equal groups,
an ideal two-stage pipeline saves `min(Q,A)/2`. This assumes full standalone
throughput while overlapping and zero extra launches, reductions or synchronization.
QKV main+reduce names are matched to isolated qkv H200 profiles; attention includes
its combine where present. Denominators are the FOCUS sensitivity benchmark medians.

| Decode load | QKV S/F us | Attention S/F us | Ideal saved S/F us | Overall gain from this load |
|---|---:|---:|---:|---:|
| 1 | 14.50/14.10 | 9.90/18.30 | 4.95/7.05 | 0.873% |
| 16 | 14.00/13.80 | 12.50/106.50 | 6.25/6.90 | 0.522% |
| 32 | 14.40/14.50 | 17.00/195.60 | 7.20/7.25 | 0.414% |

All three decode loads together: **1.820% overall**.
Using best7_reset's own benchmark denominators instead: 1.857%.
This clears the1% screening threshold only in aggregate. At 50% realization, the
overall gain is 0.897%.
The bound gives no GPU-resource availability or scheduling guarantee. M1/16/32
qkv and attention compete for the same SMs and memory; launch cost is not divisible
by head count. Splitting the current Lt split-K/reduction schedule can add cost.
No overlap kernel has been built or sent; a future design must stream each
group's complete Q/K/V/gates and preserve per-head QK RMS and cache publication.
Source hashes and exact inputs are in `bound.json`.
