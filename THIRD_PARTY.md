# Third-party code and files

Our own code is under the Apache License 2.0 ([`LICENSE`](LICENSE)). The parts below belong to others and keep their
own terms. Every copy keeps its original notices.

## Code from other teams

Each team's code was ported, with its notice, into our candidates (`problems/*/dev/public_*` and
`moe_public_ports`, some `dev/sNN/inputs/`) and into the scored decode layer.

| Source | Pin | License | In this repo |
|---|---|---|---|
| [NathenFernandes/muse-bf16-proj](https://github.com/NathenFernandes/muse-bf16-proj) | `86aff75` | MIT, © 2026 Nathen Fernandes | Projection kernels: [`final/muse_decode_layer/src/NF_VENDOR/`](final/muse_decode_layer/src/NF_VENDOR/), `problems/muse_bf16_proj/dev/public_nf/` and the decode-layer candidates from S15 on |
| [operatorstack/open-kernel-challenge-solutions](https://github.com/operatorstack/open-kernel-challenge-solutions) | `6d9b776` | MIT, © 2026 bigboateng; it reproduces SGLang code under Apache-2.0 | Attention: [`final/muse_decode_layer/src/_PUBLIC_SOURCES/operator.py`](final/muse_decode_layer/src/_PUBLIC_SOURCES/operator.py), [`src/_HY_SOURCES/operator_direct.py`](final/muse_decode_layer/src/_HY_SOURCES/operator_direct.py); the `public_ports` lanes |
| [lee101/openkernels](https://github.com/lee101/openkernels) | `601c654` | Apache-2.0 | Attention: [`final/muse_decode_layer/src/_PUBLIC_SOURCES/lee/`](final/muse_decode_layer/src/_PUBLIC_SOURCES/lee/), [`src/_HY_SOURCES/lee_hybrid/`](final/muse_decode_layer/src/_HY_SOURCES/lee_hybrid/); the `public_ports` lanes |

The full operatorstack and lee101 license texts, including operatorstack's NOTICE for its SGLang-derived parts,
ship with the decode layer in
[`src/_P5B_PUBLIC_NOTICES.txt`](final/muse_decode_layer/src/_P5B_PUBLIC_NOTICES.txt) and
[`src/_HY_PUBLIC_NOTICES.txt`](final/muse_decode_layer/src/_HY_PUBLIC_NOTICES.txt). Every lane holding ported code
carries the texts too, either as a `NOTICES.txt` in the lane folder or inside its patches (`public_hybrid`,
`dev/sNN/inputs/`). Nathen Fernandes's MIT text is in
`problems/muse_bf16_proj/dev/public_nf/LICENSE.NathenFernandes`, at the top of the scored decode layer and in
[`src/NF_VENDOR.py`](final/muse_decode_layer/src/NF_VENDOR.py).

## Libraries

- **SGLang** ([sgl-project/sglang](https://github.com/sgl-project/sglang), Apache-2.0). `problems/*/baseline.py` are
  the organizers' copies of SGLang's kernels, and `serving/patch.diff` is a patch to SGLang.
- **FlashAttention** ([Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention), BSD-3-Clause,
  © 2022 the respective contributors and © 2023 Tri Dao). Its parameter header, with its license, is embedded so our
  code can call the FA3 build inside the judge's image: in
  [`final/muse_decode_layer/src/_P5B_BRIDGE_SRC.cu`](final/muse_decode_layer/src/_P5B_BRIDGE_SRC.cu),
  `problems/muse_attention_decode/dev/full_attention/` and `problems/muse_decode_layer/dev/s21/inputs/`.

## The challenge's files

These are the organizers' files, included unmodified so the tools run:

- `problems/*/TASK.md`, `AGENTS.md`, `reference.py`, `checks.py`, `nvfp4.py` and `baseline.py`, written by `okc init`
- `upstream/`: `llms.txt`, the Official Rules (`terms.pdf`) and the public API's description
- `serving/RULES.md`

The challenge's judge adapts GPU MODE's [reference-kernels](https://github.com/gpu-mode/reference-kernels) and
[kernelbot](https://github.com/gpu-mode/kernelbot), which are under the
[June 9 Researcher Reciprocity License](https://github.com/gpu-mode/reference-kernels/blob/main/LICENSE). Where these
files derive from GPU MODE's materials, that license applies to them.
