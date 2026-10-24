# Best6 expert portfolio — frozen construction

Current candidate and attention sampling fix: **[best6_attention/README.md](../best6_attention/README.md)**.
This directory reproduces the original expert-portfolio construction, SHA-256
`c582f3427684b69b67b13be3f1bcb9f43c68fc8b32e9556bffe4377687d85491`, preserved in git commit
`313a282b8d1393bfdad928716becb53986f4dd1a`. Its validation applies to that frozen base.
Rebuilding this directory overwrites the current attention fix; use the attention builder for the current candidate.
Base is exact glue `9912056970eef92aea847fd1cb68e7426ce73df738eef739e2f63ca1837fb714`.
Prefill adapter source is `cand_mid_sweep.py` at SHA
`1de6c32c7a5337dc539a5b6feb24a479bed217cb794a92a32760f9fcda82e075`.
This candidate has **not been sent**. The owner sends it only after the separate H200
M128/256 R0 + twelve-seed adapter lab confirms native safety; a C++ abort cannot be caught in Python.

## What changes

At M128/256 the unified portfolio contains V9 NT1, V9 NT2, raw17, humming_masked and dg_w8.
Every other token count uses best5's V9 path. All 75 base helper/kernel implementations and all native
source literals are unchanged; only prepare and the layer's expert dispatch call change.
Twenty-seven prefill definitions are embedded with isolated global names. [audit.py](audit.py)
reverses the namespacing and requires exact AST equality; its GROUPED_SRC literal is byte-identical.
Humming uses the exact schema transformation, heuristics, compile/plan methods and masked adapter,
with setup restricted to M128/256 rather than precompiling all prefill sizes.

Prefill adapters take pooled BF16 activations before expert normalization. Two small adapter-ABI
kernels materialize those pooler partials and derive per-expert counts/slot positions from the selected
local ids. The adapters then apply their own pre-expert norm exactly once. The existing V9 path retains
its existing input. The pool materialization and slot-layout cost is included in the alternatives' timing.

Three fresh synthetic inputs come from the loaded router/pooler and the public correlated verify-stream
distribution. Each alternative must pass an independent FP32 oracle before timing and again after graph
replays with changed inputs and routes: relative L2 <=1%, worst row <=5%, finite outputs. Timing uses
four cold replays per seed, reverses backend order on alternate repetitions, and selects by the mean of
the three seed means. No submission input is inspected by prepare. Missing budget or rejection retains
V9's existing measured NT choice. Compact `BEST6_BUILD` and `BEST6_MOE` lines print last.

Optional RuntimeErrors are isolated at construction, graph admission, timing and final replay gates.
Compiler failures also drop raw17 rather than the run. V9 failures propagate. CPU native compilation
runs in parallel; **all GPU adapter setup completes before graph capture** because CUDA global capture
mode rejects concurrent GPU setup from another thread.

The exact prefill packed layouts losslessly re-encode scale bytes and preserve FP32 global-scale
sidecars. This follows the current owner request to reuse the verified adapters as they are, rather than
the prior layout-only method for raw E4M3. Prepared expert weights remain packed NVFP4; no BF16/FP8
expanded weight matrix is retained. dg_w8 expands the active experts inside every timed call and uses
a single aliased dense scratch workspace for its two GEMMs. The oracle's dense matrices are temporary
synthetic validation scratch and never enter returned state.

## Verification

Normal final source: **13 official tests, 12 fresh-input graph shapes and 14 static checks** pass; warm prepare **9.272 s**.
All three routings admit raw17 at M128/256 (relative L2 ~0.45%); local V9 NT2 is faster and is selected.
DeepGEMM/Humming native execution is Hopper-only.

The warm forced screen exercises actual prefill gathers, expansion, normalization, epilogues, slot-layout
and raw17 kernels. DeepGEMM/Humming native GEMMs use portable grouped BF16 GEMMs; the Humming schema,
transformations and plan Python code run, while only native compilation/launch use stand-ins.
All five choices pass FP32 gates at M128/256. The full layer cycles the three alternatives during eager
calls/capture so every alternative is also exercised in complete target-shape graphs.
The final empty-cache forced screen passes **13 official tests, all 12 fresh graph shapes and 14 static checks**.
Candidate prepare is **174.799 s**, **176.472 s** including extra forced attention checks, below the ~240 s target.
Native compilation (including the exact grouped adapter source) overlaps the initial weight setup.
Final SHA-pinned evidence: [validation.json](validation.json). Maximum relative L2 **0.5432%**, worst row **0.5946%**.
Full-layer target graphs/capture calls force raw17/humming_masked/dg_w8 **3/3/2 times at M128** and
**3/2/3 at M256**, in addition to the three-seed standalone graph gates. Their stand-in library launch counts
are 22 DeepGEMM and 22 Humming calls; these are coverage counts, not H200 timings.

[error_audit.py](error_audit.py) passes **18 injected RuntimeError cases**: three alternatives x three
failure stages x two M. The representative error is Humming's `shape_m is not within any range`.
Each run continues with a gated V9 choice. All three setup failures and the expired-deadline default
are also checked. Failure tests use real V9 calculations only as computational stand-ins for the
backend being deliberately failed; they are not timing evidence.

Best5 `691bff0bcff44e99` is resolved at **1.353449x**, H200 prepare **158.992 s**; its stage times are
in [BEST5_STAGES.md](../best6_attention/BEST5_STAGES.md). Glue H200 timing remains unverified here;
its lab queue key is `bd66426465d8b072`. The owner's native-adapter submission hold remains in effect.

## Reproduce

```bash
python problems/muse_decode_layer/dev/best6/build.py
python problems/muse_decode_layer/dev/best6/audit.py
CUDA_VISIBLE_DEVICES=0 tools/img env MAX_JOBS=2 python tools/local_eval.py \
  problems/muse_decode_layer/cand_r3_best6.py --no-baseline \
  --json problems/muse_decode_layer/dev/best6_normal_local.json
CUDA_VISIBLE_DEVICES=0 tools/img env MAX_JOBS=2 python problems/muse_decode_layer/dev/best6/error_audit.py
CUDA_VISIBLE_DEVICES=0 tools/img env CCACHE_DISABLE=1 MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=12.0 \
  TORCH_EXTENSIONS_DIR=/tmp/muse-best6-c582-cold/extensions \
  TRITON_CACHE_DIR=/tmp/muse-best6-c582-cold/triton \
  CUDA_CACHE_PATH=/tmp/muse-best6-c582-cold/cuda \
  python problems/muse_decode_layer/dev/best6/force.py --cold-native
```

Choose new cache directories for another cold measurement. The native H200 adapter lab and a complete
H200 best6 check remain necessary; the portable forced screen does not establish native safety or speed.
