# P5 resident-history sliding verify

Picked load: **verify32 sliding**, **29.824 us critical-path increment** in
H200 receipt [c6bd22e2084c4809](../../../../runs/muse_prefill_layer/20261025T071342Z_check_c6bd22e2084c4809.json).
It reads **16,834,560 logical KV bytes**, a **3.5072 us KV floor** at4.8TB/s.
Verify16 is24.225us /8,417,280bytes /1.7536us. `receipt_floor.py` extracts
these numbers directly into `floor.json`; it distinguishes completion
increment from residence. This is S11's unchanged sliding path also retained
by S14; S14/S15-specific native confirmation still needs the paired batch.
Round2's three shapes do not change verify16/32, so this gap survives even if
all its arms win. Every us removed here saves46us of this load's layer time.

## Structural rebuild

`resgen.py` creates a single attention launch per layer. A CTA owns all64
query rows (eight tokens x eight GQA heads) of one KV head and one disjoint
history slice. There is no query partition/KV reread, metadata pass, global
partial pass, combine launch or reset launch on the native path.

Before `griddepcontrol.wait`, every owned history page and K/V row is fetched
with volatile PTX. V transposes through4KiB scratch and is retained in place;
all cached K/VT tiles remain resident for the entire compute phase. New K/V
storage becomes available after this transpose. Only QKV/new cache append
and dependent norm/attention/gate work wait for the producer.

The actual S14 verify producers do not expose a PDL release: vdsk has no
trigger and Lt is a vendor kernel. Source-level preloads alone would serialize.
`early_edge.cu.inc` creates our consumer with a
[documented launch-completion programmatic edge](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html)
from the immediately preceding QKV kernel. It advances the stream's capture
dependency to that consumer using CUDA's documented graph/stream interop.
It does not modify the producer, its parameters, existing graph nodes/edges,
input data or timing. The consumer's real `griddepcontrol.wait` still waits
for full QKV completion and memory visibility before dependent work. The
ordinary eager path remains synchronously correct; this overlap is a CUDA
graph scheduling feature. RAW4/RAW8 use the identical resident shaders with
the original producer-completion behavior as scheduling controls.

The post-wait compute loop never refills or transposes history. It uses the
proven vq8 schedule: issue QK(j+1), issue PV(j), wait for QK, compute the next
softmax, wait for PV, rescale O. Two probability buffers preserve the live RS
WGMMA operands. One score buffer avoids the two-score prototype's ptxas
serialization warning. The final shader has no C75 warning, stack or spills.
Preparation keeps four row vectors live at a time instead of twenty; both
BF16 and strided FP32 partials retain the required BF16 activation rounding.

Cluster members publish shared partials then distribute all query-row
reductions and norm/gate stores among the members. The constant cluster
cardinality bounds epilogue register arrays and removes dynamic split
arithmetic. The16-way case reuses the dead1024-byte KN buffer after virtual
attention, making exactly enough space for64x68 FP32 partials; a static
assert checks that lifetime-based allocation.

## Arms and byte/residency model

| Arm | CTAs/cluster | Keys/CTA | Grid verify32 | Dynamic shared/CTA | Registers |
|---|---:|---:|---:|---:|---:|
| C2P | 2 | 1024 | 128 | 142336 B | 168 |
| C4P | 4 | 512 | 256 | 76800 B | 168 |
| C8P | 8 | 256 | 512 | 44032 B | 168 |
| C16P | 16 | 128 | 1024 | 27648 B | 168 |
| C4S | 4 | 512 | 256 | 76800 B | 151 |
| C8S | 8 | 256 | 512 | 44032 B | 151 |

P is the QK/PV pipeline; S is the serial compute control with identical early
resident history and bounded preparation. Every variant reads the same
unique KV bytes; the split count changes residency and math/dependency depth,
not attention work. Native CUPTI reports0B static shared per attention CTA;
use the dynamic column above as its total native shared allocation.
H200's228KiB/SM and64K
register budget permit up to three C4 resident attention CTAs without a
producer; the runtime occupancy in the H200 payload remains authoritative.
C4 is the verify32 target: all64 KV-head groups can occupy the GPU with two
CTAs/SM and process four cached tiles. C2 tests one128-CTA wave with eight
tiles; C8/C16 trade more blocks/shorter math tails against duplicated QKV
prep and cluster work. This is an achievable-resource model, not an H200
latency claim.

Verify16's unchanged vdsk producer compiles to **196672B shared on sm90**
(`audit_producer.py`, grid36x3,4 warps), and has no programmatic release. The
sm120 compile had reported172088B; that local resource figure did not predict
native residency. Including1024B static shared for attention, C4/C8 cannot
co-reside by memory budget; **one C16 attention CTA plus the producer uses
225344B**, within233472B/SM. Two C16 CTAs beside it do not fit. Its native
register ceiling of256/thread still leaves room for the168-register attention
CTA, but actual placement/overlap is measured by the payload, not assumed.
This correction follows the native producer compile, not local timings.
C4 remains the verify32 structural target; Lt's selected native resource
usage must be observed on the H200. All four derived split shapes stay in
the one paired payload.

## One paired payload

`build_payload.py` embeds exact S14 f1c77ce6 and unchanged-JIT A/A2 controls.
The six resident arms plus RAW4/RAW8 run at verify16 and verify32: **24 fresh seeds x12
rotated cold READ rounds**, every arm/seed checked against the complete
public whole-layer reference with20% margin. CUPTI records QKV, attention,
gate and O completion increments; each changed attention captures exactly
one launch. The compact receipt retains seed-level samples at1ns resolution,
critical increments, stage bounds, resource reports and output drift.

It is independent of every pending reset outcome. `tools/h200_lab.py` hosts
it at attention import time with a650s budget. It is accepted as check
**857f78fa889f4646**, at current priority2 (initially5),
sender **1465956**, board SHA **a6c3423d**, not before **1792972860**
(2026-10-26 09:01 KST), behind all four earlier P5 senders;
no waiting payload is absorbed or replaced. After the first three P5 reset
receipts completed, its priority became2 alongside round2, preserving the
sender and source bytes while moving structural work ahead of component-board
diagnostics. No conditional selection or
fallback dispatch depends on a reset result.

## Admission and rebuild

```
python3 problems/muse_attention_decode/dev/verify_resident/receipt_floor.py
python3 problems/muse_attention_decode/dev/verify_resident/build_payload.py
python3 tools/h200_lab.py build problems/muse_attention_decode/dev/verify_resident/out/resident_verify_lab.py --host muse_attention_decode --base-file problems/muse_attention_decode/dev/verify_resident/out/component.py --phase import --import-budget 650 -o problems/muse_attention_decode/dev/verify_resident/out/board.py
python3 problems/muse_attention_decode/dev/verify_resident/local_wrapper.py
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_attention_decode/dev/verify_resident/check_kernels.py
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_attention_decode/dev/verify_resident/check_edge.py
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_attention_decode/dev/verify_resident/sass_check.py
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_attention_decode/dev/verify_resident/screen.py
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_attention_decode/dev/verify_resident/screen.py --forced
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_attention_decode/dev/verify_resident/check_lab.py
```

The local standalone math checks exercise every arm with fresh graph data,
eight repeated replays, BF16/strided FP32 producer inputs, pages1/64, edge
keys, nonunit KV scales and nearby300K context. Native WGMMA uses its exact
MMA shim onsm120. Native DSM is explicitly replaced by a genuine nonspinning
global-last-CTA reduction; C2/CAP1024 uses C4/CAP512 because the local shared
memory limit cannot hold139KiB. Those substitutions are limited to the screen
tools and recorded. The remote shaders retain the actual DSM geometry.
Local timing establishes no H200 speedup or cluster placement.

`check_edge.py` independently tests the actual graph port with a producer
that has no release instruction. Onsm120 the ordinary edge began the consumer
after the producer; the launch-completion edge began it while the producer
was running, and all32 fresh replays per arm read the completed producer's
value correctly. This establishes API/visibility behavior locally, not the
H200 co-residency or speedup of the attention payload.

The first local compile exposed the C16 partial-space static assertion;
accounting for the dead KN lifetime fixed it. That failed build was never
sent. The earlier two-score pipeline had a C7515 warning and was replaced
by the proven schedule, not accepted as an overlapping implementation.
Admission passed: normal and forced local_eval each passed16 tests,12 shapes
and7 static checks;72 component cells passed eight captured replays each;
the exact compressed whole-layer payload passed2 shapes x10 arms x2 fresh
seeds with the complete public checker at20% margin and immediate cache
validation. All6 sm90a shaders compiled with history loads and V transpose
before the dependency wait, no C75 warnings, stack or spills. The32-fresh-replay
per-arm graph-port probe passed. The native receipt below measures scheduling,
producer resources and whole-layer performance.

Lab source SHA **74864386**, component SHA **5e7cb4d1**. Full source hashes,
screen counts, prewait SASS evidence and initial sender identity are in
`admission.json`. Native receipt: **857f78fa889f4646**.

## Native decision

All10 arms x24 fresh-seed TASK/cache gates pass at both loads. No arm
has a point estimate below1, so the new handoff rule leaves this verdict unchanged. Best verify16 **C8S/A1.012806** (SE0.003340), verify32
**C4S/A1.020146** (SE0.002518); A:A0.999296/0.998184,
SE0.000342/0.000252, ranges0.99586-1.00187/0.99557-0.99991.
The pipeline variants lose1.026-1.089 atv16 and1.043-1.098 atv32.
Drop these tested resident/serial/edge recipes; **no confirmation sent and
no P4 promotion**. Full arm ratios and gates are in `reset_result.json`.

The launch-completion edge works natively: verify32 C4P begins2.847us,
before QKV ends20.063us; RAW4 begins20.128us. Its attention completion
increment nevertheless rises28.224→31.584us. Native Lt uses**221508B**
shared per QKV CTA, leaving11964B of233472B: even the smallest tested
attention CTA needs27648B, so no tested arm can co-reside with that producer.
Early consumer blocks can use spare SMs; kernel start does not prove every
CTA overlaps its gather. Verify16 C16 can fit beside its196672B producer
and starts4.513us before QKV ends13.792us, but loses1.05542 overall.
The native resource/start observations supersede the pre-run residency model.

This closes the tested schedules on S14, not resident attention generally.
Lower-footprint QKV producers, different gather/transpose or output schedules
remain open; none is measured by this receipt. The critical-path increments
are diagnostic traces, not per-CTA preload-completion clocks.

Current handoff rule: every measured per-shape point ratio below1 goes to P4,
with SE, seed IDs and A:A, for composition and held-out six-load confirmation.
This packet supplies none. Complete rejudgment, including all resident arm
estimates, is in [point_handoff.json](../sliding_round2/point_handoff.json).
The newer shared-full barrier review is audited in
[operand_ready/README.md](../operand_ready/README.md): S15 verify sliding is
FA3, not VQ8. It does not change these measured resident-kernel losses.
