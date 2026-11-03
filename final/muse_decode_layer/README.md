# S22: the scored decode layer

This is the `muse_decode_layer` submission behind our best score: judge job `210d40e0c9cb4b71` on 2026-10-27 at
10:37 KST, **1.8892x** SGLang (95% CI 1.861-1.918). All checks passed and `prepare` took 179.6 s. Receipt:
[`runs/muse_decode_layer/20261027T013731Z_submit_210d40e0c9cb4b71.json`](../../runs/muse_decode_layer/20261027T013731Z_submit_210d40e0c9cb4b71.json).

The file we sent, [`problems/muse_decode_layer/cand_r3_best7_s22p1_final.py`](../../problems/muse_decode_layer/cand_r3_best7_s22p1_final.py),
is a 34-line wrapper. It decompresses an LZMA + base85 blob into a 1.7 MB Python source and executes it. Most of
that source's CUDA kernels, and the Python modules it `exec()`s, sit in string literals on single lines. This folder
holds the same source split into files you can read:

- [`layer.py`](layer.py): the Python driver, 8,113 lines. Each large single-line literal there reads
  `_embedded("<file>")` instead, and `_embedded()` (defined near the top) returns the original string. A few CUDA
  sources were already multi-line `r"""…"""` strings and stay inline.
- [`src/`](src/): 61 files. There are 51 CUDA sources (`.cu`), 7 Python modules executed from strings (`.py`; four
  of them have `_embedded` references of their own) and 2 license-notice texts (`.txt`). The vendored projection
  module is packed again as base64 of zlib: [`src/NF_VENDOR/NF86aff75-selected.py`](src/NF_VENDOR/NF86aff75-selected.py)
  is its readable text and the `.b64` file beside it is the exact literal. Files are named after the variable, dict
  key or `compile()` name that held them, or after their line number when there was none. To see where a file is
  compiled, search `layer.py`, or its parent module under `src/`, for the file name.
- [`build.py`](build.py): `check` joins the files back and verifies three things. The result equals, byte for byte,
  the source unpacked from the scored file (SHA-256 `fd5e0ca2…`). `layer.py`'s own `_embedded()` rebuilds the same
  strings. The `.b64` literal decodes to its readable module. Nothing is recompressed, so the check doesn't depend
  on your zlib or liblzma build. `split` regenerates this folder from the scored file.

```sh
python3 final/muse_decode_layer/build.py check
```

Submit the scored file, not `layer.py`: a judged run sees only the one file.

## How the file is built

S22 is the end of a chain of integrations: S10, S11, …, S19, S21, S22. Each step appended its kernels and wrapped the
previous step's `prepare` and dispatch. For example, `_s19_prior_warm = _s16_prepare_delta` comes just before a new
`def _s16_prepare_delta(...)` that calls it. So later definitions override earlier ones. Read from the top for the
base pipeline, and from the bottom for the newest overrides. The module docstring states the pipeline:

```
qkv_gate GEMV -> qk_prep (QK-norm, RoPE, KV-cache store, query cast) -> attention (+ norm-gate)
-> o GEMV -> boundary 1 (post-attn norm, residual gate, pre-FFN norm)
-> pooler + router GEMV (router as BF16 hi+lo rows, FP32 out)
-> routing + pre-expert norm -> experts (V9 NVFP4) -> unpooler GEMV
-> boundary 2 (post-FFN norm, residual gate, next layer's pre-attn norm)
```

`prepare(weights)` runs once, untimed, with a 300 s limit. It repacks and codes the weights losslessly, compiles the
extensions, and fixes a plan for each shape from the dispatch tables. `kernel_fn(data, state)` looks up the plan for
the shape it is given and launches its kernels. The judge captures and replays that call as a CUDA graph.

## Where things are

Line numbers are in `layer.py`.

| Line | Section | Kernels |
|---:|---|---|
| [42](layer.py#L42) | Verify-dense split-K projections (TMA descriptors) | Triton |
| [112](layer.py#L112) | Dispatch table: one row per scored case (glue, attention, MoE, projections) | |
| [144](layer.py#L144) | Cluster glue: one thread-block cluster per token | [`src/GLUE_CL_CUDA.cu`](src/GLUE_CL_CUDA.cu) |
| [271](layer.py#L271) | Split-K swap-AB GEMV; consumers sum the partials | Triton |
| [484](layer.py#L484) | QK prep: norms, RoPE, FP8 KV store, query cast | Triton |
| [552](layer.py#L552) | Portable split-KV attention | Triton |
| [730](layer.py#L730) | Residual boundaries | Triton |
| [884](layer.py#L884) | MoE V9: persistent resident NVFP4 expert kernel | inline CUDA string below it |
| [1507](layer.py#L1507) | Lossless expert repacking (prepare only) | |
| [1773](layer.py#L1773) | Coded dense GEMVs: exact 12-bit BF16 weight codes | inline CUDA string below it |
| [2425](layer.py#L2425) | Projection portfolio: custom split-K GEMV, cuBLAS, cuBLASLt | |
| [2603](layer.py#L2603) | `attn_main`, our fused FP8 paged-KV decode attention; then its verify sibling and XQA-style verify families | [`src/ATTN_PORT_CUDA_SRC.cu`](src/ATTN_PORT_CUDA_SRC.cu), inline strings |
| [4137](layer.py#L4137) | MoE V9 NT2 body | [`src/V9_NT2_CUDA_SRC.cu`](src/V9_NT2_CUDA_SRC.cu) |
| [4192](layer.py#L4192) | Prefill-layer adapter with isolated names: grouped experts from V6 tiles on BF16 register-A WGMMA | [`src/_pf_GROUPED_SRC.cu`](src/_pf_GROUPED_SRC.cu) |
| [5404](layer.py#L5404) | Side-stream L2 transport within the call | [`src/P7_PREFETCH_CUDA.cu`](src/P7_PREFETCH_CUDA.cu) |
| [5570](layer.py#L5570) | Small-M MoE flags (NT1 variants) | [`src/SMALLM_VAR_CUDA.cu`](src/SMALLM_VAR_CUDA.cu) |
| [5633](layer.py#L5633) | Large-M MoE: heavy experts on a side stream | [`src/NF_CUDA.cu`](src/NF_CUDA.cu) |
| [5838](layer.py#L5838) | vq8: verify-block attention over the paged FP8 cache | [`src/VQ8_CUDA.cu`](src/VQ8_CUDA.cu), [`src/cs_CUDA.cu`](src/cs_CUDA.cu) |
| [6041](layer.py#L6041) | Nathen Fernandes's projection kernels (MIT) | [`src/NF_VENDOR/`](src/NF_VENDOR/) |
| [7480](layer.py#L7480) | Exact coded banks, geometry chosen by shape only | |
| [7522](layer.py#L7522) | QKV exact-12 dispatch | |
| [7852](layer.py#L7852) | Shape-only FA3 schedules | |
| [8040](layer.py#L8040) | Licensed public attention behind our producer ABI | [`src/_PUBLIC_SOURCES/`](src/_PUBLIC_SOURCES/) |
| [8075](layer.py#L8075) | Hybrid direct-partial attention | [`src/_HY_SOURCES/`](src/_HY_SOURCES/) |

At the S22 step, the attention behind each scored cell (S = sliding layer, F = full layer) comes from S19 (ours),
C3 (ours: a three-stage decode kernel that reuses a dead stage for the new token), Lee
([lee101/openkernels](https://github.com/lee101/openkernels)), OP
([operatorstack](https://github.com/operatorstack/open-kernel-challenge-solutions)) or Rw4 (ours: a full-verify
kernel with a parallel row reduction and fused norm/gate):

| Load | Sliding | Full |
|---|---|---|
| decode 1 | S19 | S19 |
| decode 16 | S19 | C3 |
| decode 32 | Lee (direct input) | C3 |
| verify 1 | OP (direct input) | Lee |
| verify 16 | Lee | S19 |
| verify 32 | Lee | Rw4 |

S21 picked the ports per cell from attention-call timings. S22 then measured each port and hybrid with a
leave-one-out arm inside the whole layer, dropped the one that lost (v16F), and shipped the composition after it beat
S21 in a paired whole-layer run (R 1.0100, 95% CI 1.0097-1.0103). One kept hybrid, the decode-32 sliding direct
input, came out slightly negative in that confirmation (R 0.9987). The numbers are in
[`problems/muse_decode_layer/dev/s22/README.md`](../../problems/muse_decode_layer/dev/s22/README.md); the other
S steps' READMEs sit beside it.

## Licenses

Our code is Apache-2.0. The vendored code keeps its notices in place: Nathen Fernandes's MIT notice at the top of
the scored file and in [`src/NF_VENDOR.py`](src/NF_VENDOR.py); operatorstack's MIT and lee101's Apache-2.0 texts in
[`src/_P5B_PUBLIC_NOTICES.txt`](src/_P5B_PUBLIC_NOTICES.txt) and
[`src/_HY_PUBLIC_NOTICES.txt`](src/_HY_PUBLIC_NOTICES.txt); FlashAttention's BSD-3-Clause header in
[`src/_P5B_BRIDGE_SRC.cu`](src/_P5B_BRIDGE_SRC.cu). See [`THIRD_PARTY.md`](../../THIRD_PARTY.md).
