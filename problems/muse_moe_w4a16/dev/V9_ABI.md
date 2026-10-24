# V9 ABI for decode-layer integration

- **Prepare input:** `weights` holds local EP8 tensors: `w13 [32,8192,2048] u8`, `w13_sf [32,8192,256] E4M3`, `w13_gs [32,2] FP32`, `w2 [32,4096,2048] u8`, `w2_sf [32,4096,256] E4M3`, `w2_gs [32] FP32`, and `post_norm_w [4096] BF16`.
- **Prepare output:** state owns one lossless canonical bank: `w13p [32,512,36864] u8`, `w2p [32,256,36864] u8`, FP32 gate/up/down scales, norm, counters, and the compiled V9 module. Bank size is 905,969,664 bytes (864 MiB) per GPU.
- **Kernel entry:** `kernel_fn(data, state)`; `data` has `hidden_states [M,4096] BF16`, `topk_ids [M,8] int32` global IDs, `topk_weights [M,8] FP32`, and `config` with `local_start`, `local_count=32`, `d_hidden=d_expert=4096`.
- **Native call:** V9 resident cooperative `moe_fused<1>` receives `x, ids, weights, post_norm_w, w13p, w2p, gs13_gate, gs13_up, gs2, inter, y, ss, pos_map, cnt, out, dense_mode, M, local_start, nstage`; it returns BF16 `[M,4096]` for normal mode.
- The kernel computes every route local to this EP rank and masks remote IDs.
