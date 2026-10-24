# Projection lane focus (P10 / G1)

Solo, GPU0 only. Do not poll the judge; send one H200 lab, then build while it runs.
The scored projection host remains `f3786341611f4f1f`.

## Closed evidence

The stock CUTLASS split-K/Stream-K recipe set lost at M=128/256 (`45f58e7009c440eb`).
The GEMV glue fusion also lost on H200 (`479aab3038c6407c`): b1 1.06–7.8x,
pool 2.04–11.36x, b2 1.09–9.02x, PREFIX 1.88–9.74x, chains 1.04–9.60x;
`picks=0`. Fusing norms/routing costs skinny-GEMV parallelism. The implementation
is closed; removing glue remains a serving design question.

## Active experiment

P10 is the serving DFlash drafter: TP8 dense shapes at M=8/32/128/256 for
qkv/fc, attention o, gate_up, down, convolution coefficients, context fc/kv,
and the borrowed LM head (7/28/112/224 draft slots). Compare stock PyTorch/cuBLAS,
cuBLASLt heuristics, skinny GEMV, and a concat-aware fc path in one H200 lab.
The serving module is `serving/sglang/python/sglang/srt/layers/muse_dflash.py`;
the lane wires it after the receipt. Agreement and CUDA-graph tests stay exact.

Lab payload: `problems/muse_bf16_proj/dev/dflash_tp8_lab.py`, SHA256
`deeb1d446fdc7262a74511f19c4cc9a70da1349586ea5dbcb5217688b3792699`, sent at priority 0.8.
The old judge waiter for `cand_dflash_diag.py` was killed before this lab was sent.
Keep the lab request ID and read its receipt once; do not submit another diagnostic.

## Constraints

No prepared BF16/FP8 copy of NVFP4 weights. Hopper-only Python branches execute
through a portable stand-in locally. Tuner failures reject and log; incumbent holds.
