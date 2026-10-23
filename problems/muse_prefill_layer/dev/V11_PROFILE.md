# V11 H200 profile ranking versus tensor floors

Serves: G2, G1. Profile calls are single traces, not scored medians. All times below are judge evidence; floors are `sim.bound` model calculations (989 TFLOP/s BF16, 1979 TFLOP/s FP8 QK). Top-k profiles omit six small kernels per shape. The floor currently charges PV at BF16 peak; legal FP8 PV can lower its attention component.

| Run | Load/layer/prefix | Expert GEMM+dequant ms | Dense/router ms | Attention ms | Boundary ms | Total ms | Tensor floor ms | Expert / dense / attention floors ms |
|---|---|---:|---:|---:|---:|---:|---:|---|
| check | turn 1/sliding/80000 | 0.526 | 0.208 | 0.033 | 0.042 | 0.852 | 0.293 | 0.138 / 0.150 / 0.005 |
| check | turn 1/full/80000 | 0.573 | 0.205 | 0.423 | 0.042 | 1.287 | 0.472 | 0.134 / 0.150 / 0.188 |
| check | turn 4/sliding/80000 | 1.340 | 0.775 | 0.062 | 0.164 | 2.431 | 1.184 | 0.564 / 0.600 / 0.019 |
| check | turn 4/full/80000 | 1.434 | 0.770 | 1.625 | 0.163 | 4.088 | 1.895 | 0.542 / 0.600 / 0.752 |
| check | first turn/full/0 | 4.444 | 4.141 | 0.897 | 0.842 | 11.016 | 6.779 | 3.096 / 3.266 / 0.417 |
| check | first turn/sliding/32768 | 4.952 | 4.138 | 0.316 | 0.851 | 10.979 | 6.474 | 3.104 / 3.266 / 0.104 |
| check | first turn/full/32768 | 5.027 | 4.141 | 4.494 | 0.846 | 15.208 | 8.404 | 3.054 / 3.266 / 2.084 |
| check | first turn/full/65536 | 4.785 | 4.149 | 8.072 | 0.852 | 18.540 | 9.957 | 2.939 / 3.266 / 3.752 |
| scored | turn 1/sliding/80000 | 0.528 | 0.209 | 0.033 | 0.042 | 0.854 | 0.293 | 0.138 / 0.150 / 0.005 |
| scored | turn 1/full/80000 | 0.587 | 0.203 | 0.422 | 0.041 | 1.295 | 0.472 | 0.134 / 0.150 / 0.188 |
| scored | turn 4/sliding/80000 | 1.297 | 0.778 | 0.062 | 0.165 | 2.394 | 1.184 | 0.564 / 0.600 / 0.019 |
| scored | turn 4/full/80000 | 1.123 | 0.773 | 1.618 | 0.165 | 3.770 | 1.895 | 0.542 / 0.600 / 0.752 |
| scored | first turn/full/0 | 4.630 | 4.192 | 0.900 | 0.843 | 11.253 | 6.779 | 3.096 / 3.266 / 0.417 |
| scored | first turn/sliding/32768 | 5.026 | 4.201 | 0.309 | 0.848 | 11.090 | 6.474 | 3.104 / 3.266 / 0.104 |
| scored | first turn/full/32768 | 4.516 | 4.194 | 4.531 | 0.843 | 14.756 | 8.404 | 3.054 / 3.266 / 2.084 |
| scored | first turn/full/65536 | 4.823 | 4.215 | 8.431 | 0.856 | 18.995 | 9.957 | 2.939 / 3.266 / 3.752 |

See `v11_profile_ranking.json` for each visible kernel, official shape weights and inferred routing. Check `6a9cd943fe404d74`; scored `965a9a816d4744ea`. Expert FLOP floors use the bound model's inferred mean routes, not observed counts from a profiled invocation.

## Weighted per-load ranking (ms)

Profile totals exclude graph replay overhead; tensor floors here are replay-adjusted. The raw bound table adds 62/62/186 replays (0.262/0.262/0.786 ms).

| Run | Load | Ranked visible stages (ms) | Total / tensor floor (ms) |
|---|---|---|---|
| check | turn 1 | expert gate/up 22.218 > dense/router GEMM 12.845 > expert down 11.161 > attention 8.271 > residual boundary 2.575 | 59.764 / 21.035 |
| check | turn 4 | expert gate/up 58.174 > dense/router GEMM 47.956 > attention 28.846 > expert down 26.404 > residual boundary 10.135 | 177.229 / 84.767 |
| check | first turn | dense/router GEMM 769.919 > expert BF16 grouped GEMM 746.304 > attention 259.062 > expert dequant 165.157 > residual boundary 158.135 > expert SwiGLU 69.515 | 2231.306 / 1295.661 |
| scored | turn 1 | expert gate/up 22.505 > dense/router GEMM 12.858 > expert down 11.158 > attention 8.273 > residual boundary 2.582 | 59.992 / 21.035 |
| scored | turn 4 | expert gate/up 53.205 > dense/router GEMM 48.161 > attention 28.726 > expert down 24.408 > residual boundary 10.208 | 170.452 / 84.767 |
| scored | first turn | dense/router GEMM 781.297 > expert BF16 grouped GEMM 744.857 > attention 264.493 > expert dequant 172.280 > residual boundary 157.742 > expert SwiGLU 67.549 | 2250.498 / 1295.661 |
