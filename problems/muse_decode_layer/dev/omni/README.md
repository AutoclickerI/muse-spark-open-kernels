# Decode omni handoff

Candidate: [cand_r3_omni.py](../../cand_r3_omni.py). Not sent by this lane; Jihoon sends.
SHA-256: `6774eceabbbb2c81b2baeb281866d7c533ad93e7711462d85eb1a96db9d6cbf1`.

Cold candidate prepare is **147.341 s**, including **84.377 s** for 11 concurrent extension builds.
Including all extra forced-family FP32 checks: **149.695 s**.
Extension, Triton and driver caches were empty; ccache was disabled. No recipe was skipped for budget.
Normal prepare with the existing local cache is 6.124 s.
Normal and forced-Hopper screens each pass **13 tests, 12 fresh-input graph shapes and 15 static checks**.
The static checks compile every CUDA literal for sm_90a.
[Machine-readable validation and every shape’s alternative times](validation.json).

The base is best `812f5f36`, preserving raw packed NVFP4/E4M3 V9 and its small pool/router GEMV.
MoE M128/256 compares V9 with NT2, NT4 and one CUTLASS mixed-input grouped recipe on three loaded-weight,
real-router fixtures, each checked against an independent FP32 oracle before and after paired timing.
Other M retain V9, matching the reuse drop-in’s scope. No prepared dequantized weight bank is retained.

The supplied mixed-input source was a compile scaffold. Omni adds a per-state grouped dispatch, GPU route
packing/shapes, correct device stride arrays and workspace reset on every timed replay. Its private
CUTLASS collective specializes group16 scaling with four scale columns per K64 tile, and widens E4M3
scales before multiplication. This avoids the stock collective’s K64 scale broadcast and FP8 product
rounding. [Readable native source](mixed.cu) exactly matches the embedded literal. NVCC `-O1` controls host
optimization; device optimization remains enabled. Tensor-only headers avoid pybind work in the CUDA TU.

Projection choices keep the measured cost-v9 Lt recipes and add two CUTLASS recipes per role at M128/256.
Only tile IDs 0, 2 and 6 are compiled. Recipe layout is `(tile, decomposition, splits, SMs, raster, swizzle, atomic)`:

| Role | C0 | C1 |
|---|---|---|
| qkv / unpooler | `(0,2,4,132,0,1,0)` | `(6,3,1,132,0,1,0)` |
| O | `(2,1,1,132,0,1,0)` | `(6,3,1,132,0,1,0)` |
| pool / router | `(0,2,8,132,0,1,0)` | `(2,3,1,132,0,1,0)` |

Attention includes the four-stage q8 `port_verify` from the current attention drop-in, all 24 XQA settings,
q1 attn_main and shortlists of FA3 recipes from cost-v9 and sweep `59e53e746df341a9`. The sweep receipt
again rejected every variant, including FA3: its timings inform the shortlist, while Omni’s own post-store
FP32 gate determines admission. Attention’s dynamic shared memory is 73,984 B for q1 and 104,448 B for q8;
PREFIX keeps best’s TB<=32 configuration (maximum 42,496 B).

Every new choice uses alternating paired cold graph timing, refreshed through three inputs. The shared
230-second deadline guards each tuner/recipe. The final 12 summary lines total 8,647 bytes and name every
alternative time or rejection. A labels: `base`, `main`, `verify`; `F<split><n/t/f>` is FA3 pack=None/True/False;
`W<stage>/<warps>`, `G<stage>/<warps>` and `P<stage>/<warps>` are XQA warp/group/FP8-PV.
P labels C0/C1 follow the table, L0/L1 are discovered Lt algorithms, NN/TN are the measured cost recipes.

Hopper-only attention and CUTLASS launches use portable stand-ins locally; all native extensions actually
compile during the cold prepare. The forced test executes GPU route planning, global scales, SwiGLU,
RMSNorm and combine with packed weights dequantized inside the stand-in GEMM. NT2/NT4 execute natively.
Extra preflight calls: mixed 48, CUTLASS 148, NT2 24, NT4 24, attention main 52 and XQA 624. Full layer graphs
force NT4 at M16/32, NT2 at M128, mixed at M256, CUTLASS projections, port_verify/XQA and fused PREFIX.
Native Hopper numerical agreement, selections, timing and cold prepare remain for the owner’s receipt.

## Reproduce

```sh
CUDA_VISIBLE_DEVICES=0 tools/img env CCACHE_DISABLE=1 \
  TORCH_EXTENSIONS_DIR=/tmp/muse-omni-cold-final2/extensions \
  TRITON_CACHE_DIR=/tmp/muse-omni-cold-final2/triton \
  CUDA_CACHE_PATH=/tmp/muse-omni-cold-final2/cuda \
  TORCH_CUDA_ARCH_LIST=12.0 MAX_JOBS=2 \
  python problems/muse_decode_layer/dev/omni/force.py --cold-native

CUDA_VISIBLE_DEVICES=0 tools/img env MAX_JOBS=2 python tools/local_eval.py \
  problems/muse_decode_layer/cand_r3_omni.py --no-baseline \
  --json problems/muse_decode_layer/dev/omni_local.json

python problems/muse_decode_layer/dev/omni/lever_table.py
python problems/muse_decode_layer/dev/omni/write_handoff.py
```

## Forced tuner selections before the full-layer overrides

These are sm_120 diagnostics with attention/CUTLASS launch stand-ins; they do not rank H200 kernels.
Projection order is QKV, O, pool/router, unpooler. M<=8 uses the sliced unpooler.

| B | Q | Context | Attention; projections; MoE; PREFIX |
|---:|---:|---|---|
| 1 | 1 | sliding | `('layer', None) ('custom', 'custom', 'custom', 'custom') v9 ('regular', True)` |
| 1 | 1 | full | `('layer', None) ('custom', 'custom', 'custom', 'custom') v9 ('regular', True)` |
| 1 | 8 | sliding | `('xqa_group', ('group', 5, 1)) ('custom', 'custom', 'custom', 'custom') v9 ('regular', True)` |
| 1 | 8 | full | `('xqa_warp', ('warp', 4, 1)) ('custom', 'custom', 'custom', 'custom') v9 ('regular', True)` |
| 16 | 1 | sliding | `('attn_main', None) ('custom', 'custom', 'custom', 'custom') v9 ('regular', True)` |
| 16 | 1 | full | `('attn_main', None) ('custom', 'custom', 'custom', 'custom') v9 ('regular', True)` |
| 16 | 8 | sliding | `('port_verify', None) ('custom', ('lt', 1), 'cublas', ('short_lt', 1, 0)) nt2 ('regular', True)` |
| 16 | 8 | full | `('port_verify', None) ('custom', ('lt', 1), 'cublas', ('short_lt', 1, 0)) nt2 ('regular', True)` |
| 32 | 1 | sliding | `('attn_main', None) ('cublas', ('lt', 0), 'custom', 'cublas') v9 ('regular', True)` |
| 32 | 1 | full | `('attn_main', None) ('cublas', ('lt', 0), 'custom', 'cublas') v9 ('regular', True)` |
| 32 | 8 | sliding | `('port_verify', None) (('lt', 1), ('cutlass', 0), 'custom', ('lt', 1)) nt2 ('regular', False)` |
| 32 | 8 | full | `('xqa_fp8_pv', (5, 1)) (('lt', 1), ('cutlass', 0), 'custom', ('lt', 1)) nt2 ('regular', False)` |
