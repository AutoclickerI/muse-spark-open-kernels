M>=64 uses the full unpooler. BF16 bank197,132,288B, not138,412,032B (the latter uses the <=8-token unpooler slice). Router uses4096 pool+256 hi+256 lo rows, all FP32 outputs. Bytes = W+X+Y once; floors use3.6TB/s and800TFLOP/s, without launch/glue overhead.

| M | Role | W MB | All MB | GFLOP | Byte/math floor us | Floor us | cuBLAS/Lt us | Gap us/layer; x62 us |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 64 | qkv_gate | 37.75 | 39.09 | 2.416 | 10.86/3.02 | 10.86 | unmeasured | pending lab |
| 64 | o | 16.78 | 17.96 | 1.074 | 4.99/1.34 | 4.99 | unmeasured | pending lab |
| 64 | pool_router | 75.50 | 77.73 | 4.832 | 21.59/6.04 | 21.59 | unmeasured | pending lab |
| 64 | unpooler | 67.11 | 68.68 | 4.295 | 19.08/5.37 | 19.08 | unmeasured | pending lab |
| 64 | **sum** | 197.13 | 203.46 | 12.616 | — | **56.52** | unmeasured | pending lab |
| 128 | qkv_gate | 37.75 | 40.44 | 4.832 | 11.23/6.04 | 11.23 | 21.728 | 10.50; 650.7 |
| 128 | o | 16.78 | 19.14 | 2.147 | 5.32/2.68 | 5.32 | 11.808 | 6.49; 402.5 |
| 128 | pool_router | 75.50 | 79.95 | 9.664 | 22.21/12.08 | 22.21 | 28.064 | 5.85; 363.0 |
| 128 | unpooler | 67.11 | 70.25 | 8.590 | 19.52/10.74 | 19.52 | 23.728 | 4.21; 261.2 |
| 128 | **sum** | 197.13 | 209.78 | 25.233 | — | **58.27** | **85.328** | **27.06; 1677.4** |
| 256 | qkv_gate | 37.75 | 43.12 | 9.664 | 11.98/12.08 | 12.08 | 24.128 | 12.05; 747.0 |
| 256 | o | 16.78 | 21.50 | 4.295 | 5.97/5.37 | 5.97 | 13.776 | 7.80; 483.9 |
| 256 | pool_router | 75.50 | 84.41 | 19.327 | 23.45/24.16 | 24.16 | 33.856 | 9.70; 601.2 |
| 256 | unpooler | 67.11 | 73.40 | 17.180 | 20.39/21.47 | 21.47 | 28.640 | 7.17; 444.2 |
| 256 | **sum** | 197.13 | 222.43 | 50.466 | — | **63.68** | **100.400** | **36.72; 2276.3** |

At the FOCUS best6 denominators, reaching these floors at verify16/32 gives1.99% overall. This is optimistic: the floor excludes launch overhead and assumes stage peak rates;1.3x these floors gives 0.81%.
M64 has no incumbent measurement in this receipt; the batched lab supplies it. No number is interpolated.
