# Decode glue-only best5

Candidate: [cand_r3_best5_glue.py](../../cand_r3_best5_glue.py), SHA-256
`9912056970eef92aea847fd1cb68e7426ce73df738eef739e2f63ca1837fb714`.
Base: exact best5 `a651af25cb44550932f497b57eb82fb32c0fbe07c1f21a8592bd0982f937c150`.
The owner sends this candidate. This lane sent only the independent H200 glue lab, queue key
`bd66426465d8b072`, priority 1 (`runs/detached/bd66426465d8b072.log`). No H200 gain or <=8 us result is established.

## Change and decisions

One PDL launch each performs boundary 1 (post-attention norm, gated FP32 residual, pre-FFN norm),
routing plus pre-expert norm, and boundary 2 (post-FFN norm, gated FP32 residual, next pre-attention norm).
Every residual CTA owns both norm reductions. Small loads may replicate statistics in independent CTAs,
with disjoint output chunks. There are no cross-CTA atomics, spin loops, scratch buffers or counter clears.
Routing combines FP32 hi/lo router logits, bias, exact top-8 and expert norm in one CTA per token.
The top-8 key preserves the full ordered FP32 bits and breaks ties toward the lower expert id; it does not
quantize the scores. The selected un-biased scores are renormalized as before.

Each launch signals dependents, loads static norm/gate coefficients, then waits before reading its producer.
QK prep now reads positions/cache locations/scales and evaluates RoPE before waiting for the QKV output.
The mean cold projection+glue time across three fresh seeded inputs selects warp count and replication geometry.
All inputs to those untimed geometry trials are synthetic. The normal layer path has four unchanged
projections, unchanged attention and unchanged V9 NT1/NT2 MoE. The old optional PREFIX is deleted because
it fuses norms into O/pool GEMMs, which this task excludes. Every native literal and 29 compute/weight
functions are pinned by [audit.py](audit.py).

No expert bank is dequantized in prepare. `_repack`, `v9_prepare`, `_prepare_weights` and best5's
FP32-oracle helper are unchanged. The oracle’s temporary dense expert matrices never enter returned state.
The inspected serving V9 bank and timed Marlin layout bridge also retain packed codes and E4M3 bytes;
no serving source was changed by this lane. [Weight audit](BOUNDARY_AUDIT.md) also records an unused legacy Humming scale-decoder bug.

## Evidence

[SHA-pinned validation](validation.json) records the complete final source and counts.

- Normal screen: **13/13 official tests, 12/12 scored-shape graphs, 13/13 static checks**;
  prepare **7.869 s**, warm sm120 caches. Three fresh inputs and 20 graph replays per scored shape.
- Forced-Hopper screen: **13/13 tests, 12/12 fresh graph shapes, 13/13 static checks**;
  candidate prepare **224.516 s**, wrapper including additional forced checks **226.144 s**.
  Extension/Triton/CUDA caches were empty and ccache disabled. Four native Hopper attention modules
  compiled, with 44 attn_main and 528 XQA stand-in calls. Full layer graphs force NT2 at M32/128/256.
  All 18 geometry rows completed; every trial used three fresh seeded FP32 gates and mean cold timings.
- All **68** new Hopper glue specializations compile, maximum shared memory **1,024 B**. Emitted PTX confirms coefficient loads between dependent-launch signaling and GDC wait.
- Lab smoke: three changed captured inputs per geometry, independent FP32 boundary/router/norm gates,
  exact selected ids and local ids; six token counts, both boundaries, all proposed launch geometries.
  Local timing is coarse and not an H200 latency claim.
- [PROFILE.md](PROFILE.md) lists every glue row in board receipt `0e94268cf5654012`, with scheduler rows
  separate. The organizer aggregates both residual calls, so individual boundary times are unavailable.
  PDL durations include waits and overlap compute: their sum is not exposed critical-path latency.
- Best5 had no receipt at the turn’s one read. Its stage times and the glue lab’s H200 comparison remain pending.

## Reproduce

```bash
python problems/muse_decode_layer/dev/glue/build.py
python problems/muse_decode_layer/dev/glue/audit.py
CUDA_VISIBLE_DEVICES=0 tools/img env MAX_JOBS=2 python tools/local_eval.py \
  problems/muse_decode_layer/cand_r3_best5_glue.py --no-baseline \
  --json problems/muse_decode_layer/dev/glue_normal_local.json
CUDA_VISIBLE_DEVICES=0 tools/img env CCACHE_DISABLE=1 MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=12.0 \
  TORCH_EXTENSIONS_DIR=/tmp/muse-glue-9912-cold/extensions \
  TRITON_CACHE_DIR=/tmp/muse-glue-9912-cold/triton \
  CUDA_CACHE_PATH=/tmp/muse-glue-9912-cold/cuda \
  python problems/muse_decode_layer/dev/glue/force.py --cold-native
```

Choose new cache directories for another cold test. [dropin.py](dropin.py) exports the exact new kernels
with their matching PDL launch helpers for a serving port. It is experimental until the H200 receipt.
The lab uses synthetic split counts (printed in each row), so the full candidate check is still required at the actual producer layouts. The H200 lab payload is [prof/lab/decode_glue.py](../../../../prof/lab/decode_glue.py), generated by
`build_lab.py`; `lab_smoke.py` executes its independent gates locally. After that lab lands, use
`python tools/h200_lab.py read <job>` once on resume. The candidate itself has not been sent.
