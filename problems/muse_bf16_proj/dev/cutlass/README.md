# CUTLASS verify projections

Read when handing the M128/256 H200 lab to the decode lane. No judge polling or wait:
inspect a receipt once per turn, build independently, end if only the receipt remains.
No candidate or serving lane file is modified. #45015 already belongs to the serving
lane (`docs/serving/upstream_sglang.md`); its port is not duplicated here.

`projection_cutlass.py` is a self-contained source block. Its only dependencies are
standard Python, torch, CUDA and image-shipped CUTLASS headers. The image's FlashInfer
CUTLASS version is4.5.0 with the Hopper3.x collective API. The relevant implementation
is `sm90_gemm_tma_warpspecialized_cooperative.hpp` and
`sm90_tile_scheduler_stream_k.hpp`; ping-pong explicitly disallows Stream-K.

Weights remain BF16. The allowed router hi/lo representation is concatenated after
pooler as one BF16[4608,8192] bank. One FP32 GEMM returns all4608 rows to the layer's
existing consumer; no separate router-product merge or GEMM reduction launch is added.
The consumer preserves BF16 pooler rounding, router hi/lo sum and every norm.

Eight tile/ring/layout specializations x BF16/FP32 outputs:

|Tile id|M,N,K|Cluster M,N|Stages|Operands/output|
|---|---|---|---|---|
|0|128,128,64|1,1|auto|X@W.T /row-major|
|1|128,128,64|1,2|auto|X@W.T, activation multicast across N|
|2|128,64,64|1,1|auto|X@W.T, more output tiles|
|3|128,256,64|1,1|auto|X@W.T, wider output tiles|
|4|128,128,128|1,1|auto|X@W.T, larger K granule|
|5|128,128,64|1,1|3|X@W.T, shorter ring|
|6|128,128,64|1,1|auto|W@X.T /column-major; same physical output|
|7|128,64,64|2,1|auto|W@X.T, activation multicast across weight rows|

Each tile tests explicit Stream-K, split-K2/4/8/16, data-parallel and heuristic modes;
additional grid sizes132/98/66 SMs, M/N raster, swizzle4, deterministic and atomic
reduction orders. 144 recipes/shape. CUTLASS performs partial reduction and epilogue
within the persistent kernel. Its required workspace-counter zeroing is included in
**every timed invocation**, including capture/replay; it is never hoisted into prepare.
Workspace bytes, shared storage, stage count and threads are exposed in the native ABI.
The release compile uses `-O3 -DNDEBUG`, not device debug assertions.

## Lab and admission

Payload `prof/lab/verify_proj_cutlass.py` is appended to proj's f3786341611f4f1f source.
The scored kernel remains byte-identical to the host. Optional compile, plan, graph,
oracle, timing and finalist exceptions reject/log; they cannot displace the host.
Lab deadline225s after entry, bounded by host deadline285s from prepare; base prepare
was about8s in prior H200 runs. The actual cold H200 compile/total must be checked.

Exact W[N,K]: qkv2304x8192, pool_router4608x8192 (FP32 result), unpool8192x4096,
o8192x1024. M128/256 only. Three fresh BF16 banks include a one-column layout probe;
checks against independent FP32 and H200-proven cuBLAS/Lt use relative L2<=1%.
Cold1GiB float32 flush, CUDA graphs, swapped order,20 paired finalists. Coarse4-pair
screen is diagnostic only. The small-M GEMV math is also timed with64-row token tiles,
including its final split sum; the incumbent's actual M>64 path is cuBLAS.
Compact controls, every tile/scheduler family's best loss, and20-pair winners print last.

Local checks use GPU0/sm120. The CUDA code is cross-compiled for sm90a; **Hopper SASS
cannot execute here**. `validate.py` forces all144 recipes for all8 shapes through a
strict portable stand-in at the real C ABI, reaching1152 distinct cases. It captures
real BF16/FP32 math and checks fresh replay, tuple ABI, plan lifetime, changed weight/
stride/dtype guards and rejection of attribute/runtime/accuracy/budget/unknown tile.
Mock ordering values in that test are selection fixtures, never timing evidence.

## Drop-in interface

Append `projection_cutlass.py` to the lane's single file. Keep all non-target shapes:

```python
_vgc_original_prepare = prepare
_vgc_original_project = project
project = vgc_bind(_vgc_original_project)

def prepare(weights):
    started = time.monotonic()
    state = _vgc_original_prepare(weights)
    rows = vgc_prepare(state, _vgc_original_project, started + 240, recipes=VGC_RECIPES)
    for row in rows:
        print(row, flush=True)
    return state
```

`project(x,w,name,state,out_dtype=torch.bfloat16)->(partials,splits)` is unchanged.
`vgc_prepare` retimes only the **H200-positive** shortlist, preserving the supplied
incumbent if a finalist loses or fails. Unknown shapes/strides/weights keep the caller.
It reserves100s before a cold module compile, then divides remaining time by shape.
The currently empty default shortlist means **no unmeasured CUTLASS variant is installed**.
After receipt, `read_receipt.py receipt.json --recipes <file>` produces the mapping,
verifying LAB_BEGIN's payload SHA before attributing any performance. Refresh a new
handoff, don't mutate a frozen sent payload. A new full-layer screen is required for
any H200-selected integration; isolated projection correctness isn't that evidence.

Reproduce: `python build.py`; GPU0 `tools/img python validate.py`; build the host with
`tools/h200_lab.py build prof/lab/verify_proj_cutlass.py -o <fixture>`; run
`tools/img python tools/local_eval.py <fixture> --json <result>`.
Evidence: `runs/screen/verify_proj/cutlass_*`. Final host9/9+30/30,1.026s prepare,
3 static checks pass. Full-layer portable fixture13/13+12/12,3.113s prepare,
9 static checks pass. `check_identity.py` confirms all15 production definitions and
constants in the20KB pruned block match the admitted fixture. Full factory/ctypes
construction is independently forced. The final stand-in test covers1152 recipes,
2367 ABI calls plus48 fresh/guard checks; mock times are not measurements.

Frozen block SHA256a5823cb913cb24fbbd316a004c450c7e7825aa4e409ad8bd96e6dcc02f10ed60.
Sent payload SHA256e43ed9b225d98cf1feaf1b720f295ca21a69bf74a389cc560ed4ffc588246bab;
scored hostSHAb81c9564402fa62c4e4a87cca541c853506aa6884c8cd4b41b61f583b7fb34ab.
H200 timing and actual cold prepare total remain pending.
