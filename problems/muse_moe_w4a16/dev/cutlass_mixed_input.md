# Hopper mixed-input MoE

The shipped CUTLASS SM90 RS mixed-input builder **compiles** an E2M1/E4M3 tuple with BF16
activations and FP32 accumulation. It **cannot implement NVFP4 group16** unchanged:
`sm90_mma_tma_gmma_rs_warpspecialized_mixed_input.hpp:484` requires
`group_size == K || group_size % TileShapeK == 0`. The compiled probe uses tile K64,
so group16 is invalid. The header also requires at least four MMA K blocks (:855).
This confirms the host `can_implement` failure in H200 job `3a98cfc8ef9545c7`.
Skipping that check does not repair the collective's one-scale-per-K-tile machinery.
The old grouped scaffold is archived at `muse_decode_layer/dev/parked/cand_r3_mixed_grouped.py`.

The replacement is [mixed_wgmma/engine.py](mixed_wgmma/engine.py): TileLang 0.1.12 RS WGMMA,
weights on M, routed tokens on N, GPU grouping/hot-first scheduling and N tiling.
E2M1 and raw E4M3 are converted in the timed mainloop; separate FP32 global scales are
applied in each GEMM epilogue. No prepared BF16/FP8 expert bank exists.
The scalar PRMT and bit-plane/BF16x2 layouts are both lossless packed formats.

Experiment and validation details are in [mixed_wgmma/README.md](mixed_wgmma/README.md).
The scalar lab is H200 job `296fdf45382244dc`; H200 results are pending.
