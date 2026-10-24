# Full q8 FlashDecoding structure

`build.py` generates `../../cand_verify8_flashdecode.py` and the decode-layer
adapter `../../../muse_decode_layer/dev/verify8_flashdecode.py` from the same
CUDA source and driver. The attention board is used only to measure this
stage for the decode layer's overall score.

Each request/KV-head/split has one 128-thread CTA. Its four warps each own
16 of the 64 query-token/GQA rows. QK uses raw FP8 E4M3 keys and normalized
queries rounded to FP8. Online max/sum, probabilities and PV accumulators stay
in warp registers. Probability exponentials use packed FP16x2 approximate
exp2 before the allowed FP8 P conversion; online max/sum and output
accumulators remain FP32. Four-lane groups own each output row. Packing adjacent QK
panels produces FP8 P fragments without a probability tile in shared memory.
PV uses raw FP8 V. The CTA transposes each staged V tile once using packed
byte permutations; all four warps then load contiguous B words. The transpose
is shared by all 64 query rows, with no FP8-to-BF16 KV expansion.

The producer stages scattered page-1 K/V through cp.async into a 3/4-stage
ring. The stage handoff synchronizes raw KV readiness and the shared V
transpose; softmax/PV have no block barriers. Tile sizes 64 and 128 are
available for diagnostic comparison. Every warp reuses the staged KV tile;
B16/B32 use 128-key stage tiles with two warp-local 64-key math subtiles,
with no synchronization between subtiles. B1 uses 64-key tiles. Only the last split stores
new quantized KV and attends to its local stored bytes, with exact causal
masking. Historical loads stop below seq_lens, avoiding store/read races.

Producers write `[B, HK, splits, 64, 68]` FP32 O/max/sum partials. A separate
combine launch has 16 CTAs per request/KV head, four query rows per CTA.
It reduces split max/sum/output and applies output RMSNorm and sigmoid gate.
No counter, last-CTA serialization, or cross-CTA synchronization is involved.
The layer adapter also accepts FP32 projection partials and performs their
sum plus BF16 boundary rounding in the producer and epilogue.

`check_configs.py` gates every stage/tile combination on four fresh-input
fixtures, plus the layer projection ABI. `tune.py` gates every timed setting
against the task's post-store FP32 reference. Official local_eval tests and
all 12 scored graph shapes are required before sending; sm90a static builds
cover the exact CUDA source. Local timing is not H200 performance evidence.

The candidate changes full q8 only. Other shapes retain measured incumbents:
attn_main q1, attn_main B1 q8 sliding, FA3 B16/B32 q8 sliding on Hopper.
The previous single-launch sliding experiment remains a separate queued
measurement until its H200 receipt clears a shape.

Frozen H200 launch policy: B1 `(splits=128, stages=3, tile=64)`,
B16 `(17,3,128)`, B32 `(17,3,128)`. Splits scale from four waves of the
architecture's SM count below B32 and eight waves at/above B32, capped by
the allocation bound and 128 splits. The resulting nominal H200 producer
counts are 256/544/1088 and combine counts 32/512/1024. This is a measurement
policy, not an H200 performance claim or a decode settings promotion.

Accepted attention-board job `be19da94dc534323`, send key `2182717863236b53`.
Candidate SHA-256 `2182717863236b53599288f2c383e6191ebb2db529bdd7269c67e2cb0ba1ff9d`.
Layer adapter SHA-256 `613e2e564d076fc83215202f41b2e8b3ef6d7a4e430e90844288632d2f31b161`.
Final local gates: 16 task tests, 12 scored graphs, 4 static checks, 16
stage/tile fixtures with fresh captured input values, 9 adapter ABI graphs,
and 2 forced FA3 sliding glue graphs (normal and 1/64 KV scales). Profiling
confirms the adapter performs exactly two kernels even with split projection
inputs. H200 timing and whole-layer promotion remain pending.
