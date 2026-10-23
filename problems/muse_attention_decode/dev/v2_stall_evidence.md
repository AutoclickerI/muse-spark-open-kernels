# v2 verify math stall evidence

Device: NVIDIA RTX PRO 6000 Blackwell Max-Q (sm_120), B32/full, page-1,
q_len=8. `v2_clock64_profile_sm120.log` records one clock64 record per warp
and 64-key tile in the instrumented four-warp consumer kernel:

| warp | wait median | next-stage issue median | MMA median | total median |
|---:|---:|---:|---:|---:|
| 0 | 50 | 238 | 3827 | 4208 |
| 1 | 47 | 254 | 3824 | 4205 |
| 2 | 47 | 254 | 3824 | 4205 |
| 3 | 45 | 242 | 3827 | 4204 |

MMA is about 91.1% of the phase for every warp; there is no isolated warp
stall. Nsight Systems (`v2_nsys_sm120_run.nsys-rep`) reports eight
`attn_main<8,4>` instances averaging 590.2 us and 62.4% of traced GPU time.
The trace warns that CPU context-switch profiling is unavailable, but CUDA
kernel tracing is complete.

The v2 source keeps P in the existing register softmax-to-PV path, launches
the next cp.async ring stage before the current MMA, and adds a two-group
consumer kernel whose groups own disjoint K/V issue and m16 row tiles. Split
factors 1/2/4/8 are portfolio settings. On sm_120, B16/full is 1.103/1.096
TB/s (one-group/two-group) versus 1.429 TB/s q_len=1 control; B32/full is
1.306/1.280 versus 1.495 TB/s. All 16 correctness and 12 graph cases pass;
SM90 compilation of both vmath and two-group launchers passes. H200 execution
remains open.
