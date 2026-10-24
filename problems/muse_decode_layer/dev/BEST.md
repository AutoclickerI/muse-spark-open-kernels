# Decode best candidate handoff

Source: [cand_r3_best.py](../cand_r3_best.py). SHA-256: `812f5f369d042c9bb7f80c4e7198bc4883922539c619c10f1e49622f1fe7aad2`.
This build was **not sent**. Jihoon owns its screening/submission. Earlier files with this name
have different hashes; the previous committed versions remain in git.

## Verification

- Normal image screen: **13/13 official tests, 12/12 scored-shape graphs**, 20 replays with 3 fresh inputs;
  **11/11 static checks** (undefined names, call signatures, all CUDA sources compiled for sm_90a).
  Prepare: **24.646 s** using the shared local caches. Evidence: `best_local.json` / `best_local.log`.
- Cold forced-Hopper screen: **13/13 tests, 12/12 fresh-input graph cases, 11/11 static checks**.
  Candidate prepare: **175.121 s**; wrapper including additional forced-family checks: **177.152 s**.
  Evidence: `best_forced_local.json` / `best_forced_cold_r3.log`, both pinned to this SHA.
- All four optional native attention modules were compiled for sm_90a at O3 during cold prepare.
  Their device launches used a portable C++ layer-attention stand-in; FA3 used a Triton attention stand-in.
  The real Lt library, all Python tuners, V9 MoE, routing, norms and graph capture ran locally.
  Additional family checks exercised 44 attn_main calls and 528 XQA calls before the full layer screen.
  After tuning, the full forced layer uses attn_main for q1, XQA warp for q8, and fused PREFIX with PDL enabled.
- Actual sm_90 Triton metadata: PREFIX O/stats **20,480 B** at T1/8/16, **24,576 B** at T32/128/256;
  pool/router **30,208 / 42,496 B** respectively. All fit **232,448 B**.
  Evidence: `check_best_resources.py`, `best_resources_local.json`.
- All **254 finite E4M3 encodings** match the device scale conversion bit for bit, including signs, zeros
  and subnormals; the packed bank retains the exact stored E4M3 bytes. `check_best_scales.py` / `best_scales.log`.

These checks establish local agreement, capture, glue and compilation. They do **not** establish
H200 latency, H200 cold prepare time, or correctness of native Hopper attention instructions.
At this turn's one read, attention sweep **59e53e746df341a9** had no receipt. Its final winners
have not been incorporated as fixed dispatch choices; the candidate retimes its embedded families in prepare.

## What changed

The implementation starts with the V9 layer and its receipt-backed GEMV tiles (0e94268cf5654012).
It retains cost_v9's projection and PDL tuning, but times projections with their actual reduction/norm
consumer. Pool/router at T≤32 stays on V9 GEMV. Seven additional H200-positive Lt recipes from projection
lab 596b2a5d436e4fee are checked against three fresh FP32 GEMMs and retimed in swapped order; only winners remain.
The original cost Lt source and merge2 now include `ATen/cuda/CUDAContext.h`.

Attention embeds the omitted XQA matrix/group/FP8-PV sources and restores the q1 attn_main launch planner.
The undefined `_cos_sin` path is replaced by the layer's own QK preparation, including theta and scale handling.
Dead vmath, V7, grouped-MoE, persistent-island and duplicate dispatcher code is removed.
The [cost_v9 receipt](../profiles/cand_r3_v9_cost_v9.906ca5580aef4f86.log) selected regular islands
at every T1/8/16/32; deleting their losing implementation removes compile/tuner work without discarding a measured winner.
The independent FP32 admission gate uses per-head normalization, post-store cache contents, a query-rounding
bound with 20% margin, and a separate 8% check of new K/V stores. It no longer uses a 3% FA3 proxy.

PREFIX uses TB≤32, a separate last-CTA counter/statistics region per token tile, and two pipeline stages.
The tuner checks two fresh inputs before timing. MoE calls V9 directly at every T; no Humming, H7, V11,
Marlin or dequantized expert-weight bank is present. NVFP4 nibbles and E4M3 scales are repacked without conversion;
V9 converts scales in its timed device loop. The four attention modules compile concurrently.

The screen's CUDA compiler now also covers host-only CUDA strings passed to load_inline, including the new Lt library.

## Cold prepare breakdown

| Stage | Seconds |
|---|---:|
| weights | 39.650 |
| projections | 57.063 |
| attention | 62.899 |
| pdl_prefix | 15.510 |

Fresh cache command (choose a new directory for each cold measurement):

```bash
CUDA_VISIBLE_DEVICES=0 tools/img env \
  TORCH_EXTENSIONS_DIR=/tmp/muse-best-cold-r3/extensions \
  TRITON_CACHE_DIR=/tmp/muse-best-cold-r3/triton \
  CUDA_CACHE_PATH=/tmp/muse-best-cold-r3/cuda \
  TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=2 \
  python problems/muse_decode_layer/dev/verify_best.py --cold-native
```

## Per-shape normal local selection

This is the sm_120 screen's actual plan. Both sliding and full have the same stage choices within each load
in this screen. `V9` means the named custom GEMV tile; above T64 the custom projection entry itself calls cuBLAS.
All rows use **V9 MoE** and the **regular PREFIX**. PDL is shown as boundary1/boundary2 (1=enabled).
Attention is V9 fused split attention for q1 and the Triton stand-in for q8; the H200 portfolio is selected only on Hopper.

| Load (sliding/full) | T | QKV | O | Pool/router | Unpooler | PDL |
|---|---:|---|---|---|---|---|
| decode 1 | 1 | V9 | V9 | V9 | cuBLAS | 1/0 |
| verify 1 | 8 | V9 | V9 | V9 | V9 | 0/1 |
| decode 16 | 16 | V9 | V9 | V9 | cuBLAS | 0/0 |
| verify 16 | 128 | cuBLAS | cuBLAS | cuBLAS | cuBLAS | 0/0 |
| decode 32 | 32 | cuBLAS | cuBLAS | V9 | cuBLAS | 0/0 |
| verify 32 | 256 | cuBLAS | cuBLAS | cuBLAS | cuBLAS | 1/0 |

H200 selection is shape-only and prints twelve `BEST_SHAPE` lines at the end of prepare:

- Attention q1: V9 layer, attn_main, all 27 FA3 split/pack settings.
- Attention q8: V9/FA3, port_verify, XQA group/warp/FP8-PV (eight settings per XQA family).
- Projections: V9 custom/cuBLAS/Lt; at T128/256 also the seven measured TN/NN Lt recipes.
  Small pool/router remains V9 regardless of the proxy timer.
- PREFIX: regular versus token-tiled fused, gated on two fresh inputs, then graph timed.
- Boundaries: PDL off/on per T and phase. MoE: V9 for every row.

The cold forced run's tuner selections and the seven paired Lt decisions are preserved in
`best_forced_cold_r3.log`; they are stand-in measurements, not an H200 ranking.

## Remaining evidence

Read attention job **59e53e746df341a9** when the owner resumes this lane with its receipt.
A full H200 run of this SHA must establish native attention agreement, the selected per-shape plans,
actual prepare time and speedup versus that run's SGLang baseline. No score is claimed for this build.
