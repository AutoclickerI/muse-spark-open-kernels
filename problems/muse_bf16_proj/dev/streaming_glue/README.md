# Streaming projections and glue (P7)

`build.py` emits `../streaming_glue_lab.py`, a self-contained projection
implementation and one batched lab. It extends `../streaming/kernel.cu` with
the boundary continuation in `glue.cuh`. Both shared-memory consumers (scalar
fragment loads and 16-byte loads for two MMA steps) live in one CUDA library.
The original weights remain BF16; the FP32 router uses BF16 hi/lo rows with
FP32 accumulators and addition. No expert weight bank is read or prepared.

`FusedStreamingProjection(w, M, config, splits, phase, params)` is constructed
in prepare. Its call returns `(hidden, stream)` for `phase='o'/'unpool'`, or
`(normalized_pool, ids, weights, local_ids)` for `phase='pool'`. Norm vectors
in params are FP32 `1+weight`; gates are FP32 alpha and beta. Call-site scratch
and outputs have stable addresses and must have one ordered stream owner.

The projection and split-K reduction share one launch. Every row tile writes
FP32 output and publishes completion; the last row tile continues into the
normalization/routing epilogue. Atomic acq_rel publication and CTA fences cover
all writer threads. Counters return to zero. There is no grid spin or residency
assumption. Final CTAs execute independently for each token tile.

Config `c+200` distributes branch and residual moments across all projection
CTAs. The last CTA reduces their FP32 float4s, reconstructs the residual RMS
from the quadratic and emits the boundary in one pass. Config `c` instead
computes the two RMS reductions directly in the last CTA. Both variants remain
in the lab until paired H200 evidence selects one; neither is a decode default.

The o epilogue includes post-attention RMS, residual gate and pre-FFN RMS.
The alternate `normalize=False` o leaves its FP32 stream for a pool op with
`prologue=True`, which normalizes and rounds it to BF16 inside each projection
tile. Pool epilogues include pre-expert RMS, FP32 hi/lo logits, deterministic
top eight, route weights and local IDs. The unpool epilogue includes post-FFN
norm weights, post RMS, residual gate and the next pre-attention RMS; M<=8
uses exactly 1024 output rows and zero branch elsewhere.

The lab compares qkv and the three fused stages at M=1/8/16/32/128/256,
including both pre-FFN placements, then dependent o->pool/router plus unpool
chains. Controls are cuBLAS plus the existing separate boundary operations.
Inputs change between paired AB/BA graph replays, each after a 256 MiB flush.
Finalists use 20 repetitions; times are raw, without replay-cost subtraction.
The same streaming geometry plus separate boundary operations is a second
20-pair control, isolating the gain/cost of fusion from projection arithmetic.
It reports all stage errors/rejections and missing/budgeted keys explicitly.

Before timing, FP32 projection gates include outliers and exact column probes,
and independently check router hi/lo against the original FP32 router. Fused
outputs are checked against a separate FP32 oracle, including exact route and
local IDs. The chain uses its own FP32 prefix for routing: the control's BF16
o result can choose different experts near a boundary, so unnamed route slots
from the two prefixes are not comparable. Scalar outputs/streams still compare
to the control, and both standalone paths have their own oracle gate.
Cancellation probes additionally bound FP32 residual forward error relative
to both addends (1e-5); the raw projection gate and 1% normalized-output gate
still apply. A residual-relative gate alone becomes ill-conditioned there.

Reproduce local admission with `CUDA_VISIBLE_DEVICES=0 tools/img python
problems/muse_bf16_proj/dev/streaming_glue/validate.py`; `validation.json`
is SHA-pinned to the payload/source. It covers every compiled consumer and
both moment modes, split factors 1/2/4, row/token/K tails, full projection
shapes, fresh graph replays and counter reset. `audit.py` compiles the actual
sm_90a source and rejects register spills or missing vector/async/MMA families.
The local sm_120 reader retains the existing 96 KiB shared-memory stage cap.
Current admission: 6,105 eager configurations/edge checks, 510 captured graphs
with fresh replays, all 30 local host stage winners without rejection, and
5/5 official prefill host tests. Native sm_90a: 33 kernels, no entry or device
continuation spills, up to 240 registers. H200 speed and routing safety on its
fresh router remain unverified; no decode dispatch is changed.

The unsafe overlap payload stays deleted. No overlap code rides this lab.
