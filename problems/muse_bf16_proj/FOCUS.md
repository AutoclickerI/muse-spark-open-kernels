# Projection lane focus (P7 / G1)
Serves: G1, G2.

G0 is the overall board score, i.e. the decode layer (R-20261025-1).
The proj lane owns dense M1-32 GEMV efficiency; the glue lane owns boundary/route fusion and verify dense M128/256.

Current: `dev/pdl_prefetch/README.md`, payload `prof/lab/dense_pdl_prefetch.py`.
Send **4abb8bc59fee7ae0**, MoE host ac9dd461, check **7426e17cd3bc4659**.
Detached log confirms acceptance; prefill waiter3107409 terminated and removed.
`dev/pdl_prefetch/sent.json` binds queued source/payload/drop-in hashes.
Eight cold whole-layer best7 pairs: decode1/16/32 + verify1, sliding/full.
Bulk L2 hints; early/late original-row and lossless tile-packed smem rings;
512-thread GEMV and packed rings at all M1/8/16/32 roles; matched528-CTA read.
24-pair layer finalists,3 refreshed streams, A/A and single-stage ablations.
Local forced Hopper375 attn_main/150 FA3 calls,8 layers/16 stages,5 MoE tests;
2,060 native numerical checks/708 graphs;16 sm90a native kernels +12 hints
and4 BF16 boundary kernels compile. No native/hint spills; SASS ordering checked.
Keep incumbent BF16 producer boundaries; M32 rings retain KS512 to keep IDs exact.
Seed72140 fills missing tensors; real first32 host experts/norm retained.
Identical baseline route IDs/weights/localIDs per stream; route kernel timed.
Case6 QKV A0/A4/R0/P0/P2 failed .003 layer gate: retain baseline and label it;
standalone QKV coverage unchanged. Live-route promotion still needs agreement.
Drop-in `dev/pdl_prefetch/projection.py`, SHA5769cab895f35afd…; best7 unchanged
until H200 whole-layer evidence lands. Local timing is not H200 evidence.

The2.0-2.5TB/s GEMV band (feb7b287) is an implementation result, not a roof:
matched528-CTA stream3.80TB/s; historical52-61MB stream4.04-4.25TB/s.
Earlier side-stream lab9367398c624b7796 remains receipt-gated and defaults off.
L2 oracle09d19813 landed: PLAN/FINDINGS own its next-kernel implications.
No path promoted. Native transport uses explicit
CUDA global addresses. No L2 persistence or prepared NVFP4 expansion.
