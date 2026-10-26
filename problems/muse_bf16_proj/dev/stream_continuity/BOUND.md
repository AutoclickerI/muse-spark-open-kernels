# S15 cross-stage continuity bound

CUPTI activity has no HBM counter. These are exact activity intervals plus explicitly modeled ramp ceilings, not measured HBM idle. Times are weighted us/layer: (46 sliding +16 full)/62 at each load; multiply by62 for the full scored load.

| Load | Wall us | GPU idle us | No major kernel resident us | Early-ramp scenario us | Weight/KV ramp+idle ceiling us (% wall) | Contribution to snapshot GOAL gap |
|---|---:|---:|---:|---:|---:|---:|
|d1|92.31|0.00|2.73|2.67|13.45 (14.6%)|10.5%|
|d16|180.00|0.13|2.80|2.29|13.59 (7.6%)|5.2%|
|d32|226.86|0.48|3.75|4.66|12.89 (5.7%)|3.9%|
|v1|127.64|0.35|3.20|4.08|12.15 (9.5%)|6.7%|
|v16|316.42|5.45|20.00|11.34|17.25 (5.5%)|3.7%|
|v32|429.58|5.86|26.34|11.51|17.66 (4.1%)|2.8%|

Joint model ceiling R=1.08557, 32.8% of ln(2.3601/1.83789). This is not a demonstrated speedup or a physical HBM utilization bound.

| Boundary | d1|d16|d32|v1|v16|v32|
|---|---:|---:|---:|---:|---:|---:|
|qkv -> attention|2.36|2.36|2.36|2.36|2.36|2.36|
|attention -> gate|1.65|1.66|0.61|0.00|0.00|0.00|
|attention/gate -> O|2.36|2.36|2.36|2.36|2.36|2.36|
|O -> b1|0.00|0.00|0.00|0.00|0.00|0.00|
|b1 -> pool/router|2.36|2.36|2.36|2.36|2.36|2.36|
|pool/router -> route|0.00|0.00|0.00|0.00|0.00|0.00|
|route -> MoE|2.36|2.36|2.36|2.36|2.36|2.36|
|MoE -> unpooler|2.36|2.36|2.36|2.36|2.36|2.36|
|unpooler -> b2|0.00|0.00|0.00|0.00|0.00|0.00|

Logical read MB / completion-critical us, with busy same-owner DAG gaps assigned to that boundary. These are unique-byte work models, not DRAM byte counters or additive isolated-stage timings.

| Boundary | d1|d16|d32|v1|v16|v32|
|---|---:|---:|---:|---:|---:|---:|
|front|15.92 / 6.96|25.56 / 8.26|25.87 / 12.04|37.88 / 10.71|39.85 / 13.41|41.94 / 19.51|
|qkv -> attention|5.77 / 10.67|91.45 / 37.69|182.74 / 63.07|6.28 / 20.26|95.03 / 52.25|187.29 / 80.33|
|attention -> gate|16.80 / 1.80|12.67 / 1.66|4.48 / 0.84|0.00 / 0.00|0.00 / 0.00|0.00 / 0.00|
|attention/gate -> O|13.14 / 4.89|12.89 / 6.22|16.84 / 6.05|13.16 / 4.58|17.04 / 7.40|17.30 / 9.10|
|O -> b1|0.16 / 4.14|1.15 / 2.65|1.67 / 3.01|0.62 / 2.33|6.39 / 5.50|12.68 / 8.09|
|b1 -> pool/router|56.82 / 15.40|75.76 / 21.14|76.02 / 21.32|75.63 / 21.27|77.59 / 25.93|79.69 / 32.45|
|pool/router -> route|0.22 / 3.64|1.79 / 3.34|3.56 / 3.59|0.90 / 3.07|2.38 / 2.94|4.74 / 3.61|
|route -> MoE|85.01 / 38.22|213.42 / 81.69|212.92 / 97.38|170.16 / 58.34|552.09 / 177.53|594.03 / 236.12|
|MoE -> unpooler|8.40 / 3.88|50.74 / 14.55|51.13 / 15.81|8.45 / 3.87|68.16 / 19.90|69.21 / 25.73|
|unpooler -> b2|0.20 / 2.73|1.18 / 2.67|2.23 / 3.26|0.66 / 2.85|6.42 / 6.09|12.71 / 8.78|

Measured no-GPU-activity gaps by the next boundary (us); these exclude all gaps covered by another stream.

| Boundary | d1|d16|d32|v1|v16|v32|
|---|---:|---:|---:|---:|---:|---:|
|qkv -> attention|0.000|0.081|0.136|0.112|1.586|1.452|
|attention/gate -> O|0.000|0.000|0.119|0.000|0.446|0.858|
|O -> b1|0.000|0.000|0.000|0.000|0.711|0.442|
|b1 -> pool/router|0.000|0.000|0.000|0.000|0.360|0.582|
|pool/router -> route|0.000|0.000|0.000|0.000|0.616|0.592|
|route -> MoE|0.000|0.000|0.000|0.000|1.279|1.476|
|MoE -> unpooler|0.000|0.047|0.229|0.239|0.425|0.457|
|unpooler -> b2|0.000|0.000|0.000|0.000|0.032|0.000|

| Load | Interstage materialization MB | HBM service equivalent us | % wall |
|---|---:|---:|---:|
|d1|0.686|0.150|0.16%|
|d16|9.598|2.096|1.16%|
|d32|15.110|3.299|1.45%|
|v1|4.933|1.077|0.84%|
|v16|37.507|8.189|2.59%|
|v32|63.218|13.803|3.21%|

P11 retention scenario counts one store/read per interstage output at 4.58TB/s. Already-L2 hits reduce HBM savings; on-chip retention might additionally save L2 issue/latency or change inner-loop multiplicity. Those are not bounded by this trace and need the P11 oracle. The known MoE 171/403MB repeated activation-read estimates at v16/v32 are internal-stage multiplicity, not an additive cross-stage cold-HBM stream saving.

Pure materialization HBM-equivalent joint R=1.01602; it cannot be added to the ramp ceiling because removed latency/traffic can overlap.

The early scenario assumes the ramp starts at kernel entry; the deferred ceiling permits independent loads to have been sunk until their dependency frontier. The spread is precisely what SASS/load-order audits and the paired arms test. Internal MoE deficits cannot be assigned to cross-stage continuity from unique bytes alone.

The no-major-kernel and GPU-idle rows are different quantities. Resident kernels can wait or compute; other streams can stream while the logical critical chain waits. No busy MoE gap is relabeled as an idle launch gap.
