# Best6 — public-draw attention tuning

Candidate: [cand_r3_best6.py](../../cand_r3_best6.py), SHA-256
`4516cb979dac623a6b6a2347be54cb5ecb416be23d7631aeae092a403beb9528`.
This is the original best6 `c582f342` with only the attention tuner, q8 `layer` metadata dispatch,
synthetic-fixture metadata field and final diagnostic summary changed. All **114** other helper
implementations and every native source literal are unchanged, including the entire expert portfolio,
projections, glue and `kernel_fn`. [audit.py](audit.py) verifies the delta.

The owner holds submission until `moe_decode_adapters` on the MoE host confirms native execution at
M128/256. This lane did not send the candidate. Normal/forced local results do not establish native
H200 adapter safety or an H200 speedup.

## Selection

Every q1/q8 attention setting is compared on three complete inputs from the public `generate_input`
for the scored load: ctx80000, page1, soft KV caches, fixed lengths at B1, agentic lengths at B16/32.
Seeds are fresh synthetic constants, unrelated to the judge's submission inputs. The public generator
and its seven helper functions are embedded exactly with namespaced globals; no reference import,
reference computation or check runs in the submission. Projection uses the run's own QKV weights.

Every setting first passes the existing independent FP32/cache-store gates. Cold graph timing uses
the median of three replays per draw, then the mean of those three medians. `layer` is measured on
all three draws and is the default. Incomplete trials cannot displace it. q8 is tuned first, starting
with B32 full. The shared prepare deadline and all expert selection/error handling remain unchanged.

For q8, `layer` now uses the board's exact FA3 host function and forwards the public input's scheduler
metadata. The public generator returns **None at q8**, allowing FA3 to schedule inside the call.
The previous alias constructed explicit `(0, None)` metadata and therefore did not reproduce that
control. Explicit FA3 split/pack settings remain alternatives. q1 keeps its existing kernels and choice
set but gets the same three public draws and mean selection, eliminating the one-draw bias there too.
Public full-q1 fixtures retain the harness-generated metadata field.

The old sampler already used the public +/-7.5% length envelope. Its single fixed draw, different
cache/input distribution, and different scheduler control were the mismatch. This revision changes
all three. `ATTN_PORT_PLAN` and final `ATTN_PUBLIC` rows record seeds, length vectors, winning and
control means. Full trial details stay in prepare state for inspection.

## Verification

Both normal and cold forced-Hopper screens pass **13 official tests, 12 fresh-input graph shapes,
and 14 static checks**, including sm90a native-source compilation. Normal warm prepare is **10.469 s**.
Empty-cache candidate prepare is **196.634 s**, **198.312 s** including extra forced-family checks.
All 12 attention shapes complete all three public draws with `layer` among their timed choices.

[public_equivalence.py](public_equivalence.py) compares all **36 complete fixtures bitwise** with the
public generator, including caches, stream, query input and metadata tensors. q1 and q8 use identical
length draws for equal batch/seed. Native Hopper scheduler construction is excluded on sm120;
the exact helper AST and scheduler forwarding are checked separately.

[regression.py](regression.py) gives a split a large win on one draw and losses on the other two;
the real tuner retains the better three-draw control on all 12 q1/q8 shapes. A budget-expired partial
trial also retains `layer`. Three FA3 calls verify scheduler-metadata identity, including q8 None.

The forced screen still covers all five expert choices at both M128/256 and every optional adapter
in full-layer graphs. DG/Humming native GEMMs have portable stand-ins; native compilation, Python
schemas, packed transforms, gathers, normalization and epilogues are exercised. Maximum FP32 gate
error is **0.5432%** overall and **0.5946%** worst row. The unchanged expert helpers retain the prior
[18 RuntimeError rejection cases](../best6/validation.json).

All evidence is pinned in [validation.json](validation.json). Local timings are sm120 measurements.

## Best5 receipt

Best5 check `691bff0bcff44e99` passes at **1.353449x**, H200 prepare **158.992 s**.
The [stage comparison](BEST5_STAGES.md), [complete non-compute inventory](BEST5_GLUE.md), and
[NEXT_CHECKS](../NEXT_CHECKS.md) contain all 12 loads. Verify-32 full FA3 main regressed
**377.9 -> 501.5 us** despite SGLang improving 1.6%; best5's one-sample tuner had chosen `(32, True)`.
Recovering that gap would imply about +0.7% score by accounting, not a measured revision speedup.
MoE differences at verify16/32 follow routing-seed variance. Glue H200 timing remains unverified here;
its lab queue key is `bd66426465d8b072`.

## Reproduce

```bash
python problems/muse_decode_layer/dev/best6_attention/build.py
python problems/muse_decode_layer/dev/best6_attention/audit.py
tools/img python problems/muse_decode_layer/dev/best6_attention/regression.py
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_decode_layer/dev/best6_attention/public_equivalence.py
CUDA_VISIBLE_DEVICES=0 tools/img env MAX_JOBS=2 python tools/local_eval.py \
  problems/muse_decode_layer/cand_r3_best6.py --no-baseline \
  --json problems/muse_decode_layer/dev/best6_attention_normal_local.json
CUDA_VISIBLE_DEVICES=0 tools/img env CCACHE_DISABLE=1 MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=12.0 \
  TORCH_EXTENSIONS_DIR=/tmp/muse-best6-attn-4516-cold/extensions \
  TRITON_CACHE_DIR=/tmp/muse-best6-attn-4516-cold/triton \
  CUDA_CACHE_PATH=/tmp/muse-best6-attn-4516-cold/cuda \
  python problems/muse_decode_layer/dev/best6_attention/force.py --cold-native \
  > problems/muse_decode_layer/dev/best6_attention_forced_cold_local.log 2>&1
python problems/muse_decode_layer/dev/best6_attention/write_validation.py
```

Use new cache directories for another cold measurement. The builder retrieves the frozen c582 source
from git commit `313a282b8d1393bfdad928716becb53986f4dd1a` when its ignored local copy is absent.
Use this builder for the current source; `best6/build.py` reproduces the earlier portfolio construction.
