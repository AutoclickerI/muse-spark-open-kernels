# S19 sliding setup and parallel DSM merge

Base: exact expanded S19 `62ea5bf4836277ceffb344e4792c92dd6d1e5980ea315fcbf96ae037b5ebc01a`.
Decode retains D8/D4, verify32 retains RAW2, verify16 retains FA3. The RAW
device bodies match S19 byte for byte; only decode's unused host dispatch
specializations are pruned. `audit.py` proves this identity.

The previous oracle's `Z-E` was a residual, not a measurement of Q alone.
Z still executes every history iteration with zero-byte `cp.async`, including
64-bit precomputed-offset reads, integer/swizzle arithmetic, commits, waits,
WG barriers, query/append preparation and scheduling. It has no KV payload,
QK/PV/transpose, softmax or merge. The original has 32-bit page-entry loads;
the coordinate oracle trades those for cold 64-bit offsets. Its residual is
not a hardware attribution or a direct native-Q-preparation time.

In the actual kernels, query work consists of BF16 loads or FP32 projection
partial loads/sum rounded once to BF16; FP32 sum-of-squares plus warp shuffle
and RMS normalization; position/frequency loads and RoPE sin/cos; FP32
rotation, E4M3 pair rounding, and XOR-swizzled shared-memory stores. Decode
also prepares new K/V and saves gate values in this block. Verify prepares
new K/V only after history, in the last split. Metadata RoPE and the first
two history prefetches already precede the PDL wait. Q cannot load before
that wait without moving its dependency to the producer.

History transport is `cp.async` (`LDGSTS.E.BYPASS.128`), not TMA. Page
lookups and physical offsets are ordinary LDG/integer operations. There are
no tensor-map descriptors or TMA instructions. The `desc()` helpers form
WGMMA shared-memory operand descriptors with integer shifts/ORs; they are
not TMA descriptor encoding. Line-attributed SASS excerpts/opcodes are in
`dA_sass_audit.json`, `vA_sass_audit.json`; resources and full-source hashes
are in `native.json`. `F2FP.SATFINITE.E4M3`, `MUFU.RSQ`, `SHFL.BFLY`, LDG
and sin/cos paths substantiate the preparation operations. Static site
counts alone do not identify the latency bottleneck.

| Valid arm | One structural change |
|---|---|
| Q | One split owns query loads, normalization, rotation and E4M3 packing. Cluster barrier publishes its 4 KiB Q tile; other splits copy the tile through vector DSM reads. Metadata RoPE, new-KV ownership, gates, history math and final merge remain. Decode load predicates actually suppress nonowner Q reads. |
| T | Verify-only parallel tree: all eight warps/CTA combine their two local WG partials on disjoint rows, then every WG owns final output rows and merges two CTA results through DSM. Every key and FP32 max/denominator/partial precision are retained; there is no last-WG serialization. |
| RAW | Native D8/D4 or RAW2 with the exact same runtime as Q/T. This isolates Q/T from FA3 at v16 and from module/host-dispatch differences at decode. |
| A/A2 | Identical unmodified S19 controls. |

Q uses the DSM-sharing alternative, so no P10 producer code is changed.
This leaves producer-epilogue RoPE/Q packing available if the measured
query term warrants it; moving normalization across incomplete projection
partials requires a real reduction, not treating each partial as a query.

The finer lab-only oracles use one loaded binary per family:

| Mode | Altered work |
|---|---|
| M/M2 | Full-work duplicate controls |
| Z | Precomputed offsets, zero-byte copies, no math/merge; full preparation |
| ZV | Z, projection-value loads replaced by zero; preparation arithmetic remains |
| ZR | ZV, RoPE sin/cos replaced by identity |
| ZQ | Z, entire query/new-KV/gate preparation skipped |
| ZI | ZQ, copy instructions skipped; volatile address operand retains offset reads/arithmetic |
| ZL | ZQ, history iterations skipped; initial empty commits and final wait/barrier remain |
| E/G | Immediate-return launch / empty graph |

Z-ZV, ZV-ZR, ZR-ZQ, ZQ-ZI, ZI-ZL and ZL-E telescope to Z-E. These are
ordered removal estimates. In this ready-BF16-input diagnostic, Z-ZV cannot
measure S19's actual FP32 partial-reduction cost. Whole-layer paired arms
consume the real S19 selected producers. Oracle inputs are separate fresh
seeds and altered-work outputs never nominate.

One attention-host payload packs all four loads, twelve disjoint seeds
`42000037+104729*i`, eight cold rounds and every arm, with public TASK on
every layer arm/seed and M/M2 oracle. The oracle seeds add7000000. Normal
and forced local screens use explicit C1 DSM stand-ins (verify retains two
real WGs); all-window math, actual broadcast and tree arithmetic execute.
Native H200 geometry is D8/D4 and two verify CTAs/four WGs. TASK is the only
correctness gate. Native timings and the A:A controls decide measured
recipes; every changed per-shape point below1 goes to P4 with SE/CI/seeds.
P4 owns held-out six-load G0 confirmation. A loss closes only its recipe.

Commands:

```sh
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_attention_decode/dev/setup_tree/native.py
python3 problems/muse_attention_decode/dev/setup_tree/audit.py
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_attention_decode/dev/setup_tree/screen.py
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_attention_decode/dev/setup_tree/screen_lab.py
python3 problems/muse_attention_decode/dev/setup_tree/s19_patch.py Q
python3 problems/muse_attention_decode/dev/setup_tree/s19_patch.py T
```


## First receipt: Q sharing and parallel tree

Attention check **b1a50bae4feb4ad1**, sender1227411, wrappere50f1c06,
payloade2e98dd9:12 seeds x8 cold, **216 layer TASK+96 full-work oracle TASK**
pass in178.26s private time. All layer A:A95% intervals include1. Ratios
are whole sliding-layer time/S19, with paired t11 intervals:

| Load | Q/S19 [95% CI] | T/S19 [95% CI] | A:A |
|---|---:|---:|---:|
| d16 |1.024033 [1.020005,1.028061]|—|.996862|
| d32 |1.008590 [1.005388,1.011792]|—|.998634|
| v16 |1.018728 [1.006200,1.031256]|1.018511 [1.006539,1.030483]|.999980|
| v32 |1.003218 [.999975,1.006461]|1.001648 [.999438,1.003858]|.997772|

Q/RAW is1.021559/1.010555/1.003671/1.004207; T/RAW verify is
1.003469/1.002648. No point nominee. Close these Q-broadcast and tree
recipes, not producer-fused Q prep or DSM generally. S19 v32 already uses
one-launch parallel DSM gather; the tree adds a local reduction and
barrier, not a removed standalone combine. Its registers rise172→197,
with unchanged CTA occupancy and no spills. Q remains138/172 registers.
No instrumented Q/T phase measurement establishes a sole regression cause.

The finer same-binary residual split (us) is:

| Exposed removal | d16 | d32 | v16 | v32 |
|---|---:|---:|---:|---:|
| Projection-value loads, Z-ZV |.290|.289|2.221|2.116|
| RoPE sin/cos, ZV-ZR |.383|.500|.233|.285|
| Norm/rotation/FP8 pack and decode new-KV/gate, ZR-ZQ |1.441|1.394|1.296|1.204|
| Zero-byte copy issue, ZQ-ZI |2.738|3.664|1.057|1.169|
| Cold offsets/address arithmetic/history loop, ZI-ZL |1.059|4.017|4.967|4.941|
| Remaining protocol/scheduling, ZL-E |1.075|1.090|.893|.938|

M2/M is.992023/.992579/.996870/.991583: directional0.3–0.8% control
drift. Keep it as uncertainty; weak submicrosecond partitions are unresolved.
Z-E differs from the prior oracle (6.987/10.954/10.668/10.654us here):
this new binary adds branch paths and a volatile address-use guard and
changes verify register allocation. Do not overlay these partitions onto
the old residual or call them native hardware utilization. ZV removes
projection values in a ready-BF16-input experiment; it does not measure
the actual producer's FP32 partial reduction or isolated HBM bandwidth.
The whole-layer Q/T losses use independent unmodified S19 controls.

Prior real-producer clocks (7bd3eced, S15 other stages) exposed post-wait
preparation medians2.768/1.600/3.808/2.144us, with median waits
.064/.064/.240/.256us. They support rejecting the interpretation of the
old7.3us residual as query preparation alone; they are not new S19 clocks.

## Response to address/loop cost: cooperative register page map

H preloads each split's physical-page/row map before the PDL wait. Four
adjacent lanes own four different tiles, keeping four physical rows per
thread; each16-byte copy sector obtains its row through a quad shuffle.
This replaces four redundant page-table reads/key with one, removes page
LDG from the history issue loop, and adds no shared memory, kernel launch
or producer dependency. KV copies, all math, new-KV ownership and the
parallel DSM gather remain. Native selected splits need at most four128-key
tiles. The C1 local stand-in expands to16 tiles to retain every key.

SASS: ordinary LDG.E sites41→33 decode and35→27 verify; four page LDGs
are in the prelude, none in the source-attributed copy loop. Quad SHFL.IDX
sites0→12/24. Native regs138→140 and172→168, same shared bytes/occupancy,
no new stack/local allocation or C75. QGMMA remains9 sites; LDGSTS sites
remain24 decode and rise24→36 verify through code duplication, not extra
logical KV copies. `protocol.json` proves every sector's old/new physical
coordinate across page1/64, windows1..2048 and seq0..80000.

H passes16 fresh/strided component TASK gates,32 packed S19 layer gates,
normal/forced13TASK+12graphs+6static each. The packed wrapper's dynamic
function lookup now uses globals()["lab"], resolving a static-lint false
positive; GPU bodies are unchanged. Normal/forced evidence and source
hashes are pinned in `layer_admission.json` and `native_hoist.json`.

Attention quota was spent after the first run. Its unsent reset waiter
1447395 was cancelled, and H was rerouted to prefill import check
**6b15d130e20f439f**, sender1466444, wrapper69b5e0f6, payloadcac767ec.
All four loads and16 disjoint seeds x10cold are packed against S19/A:A/RAW.
No P10 files or producer formats changed. Outcome and P4 nominations follow
the same per-shape point rule; no standalone small-gain confirmation.


## Page-map receipt and final handoff

Prefill import check **6b15d130e20f439f** is complete in161.33s private time,
all**256 layer TASK** gates pass. All A:A95% intervals include1. No oracle
runs inside this follow-up; it times the real selected S19 producers.

| Load | H/S19 [95% t15 CI] | H/RAW [95% CI] | A:A |
|---|---:|---:|---:|
| d16 |1.005026 [1.002175,1.007876]|1.000529 [.997386,1.003673]|.997644|
| d32 |.999373 [.997703,1.001043]|1.000746 [.998531,1.002961]|.998277|
| v16 |1.001271 [.994404,1.008138]|.997187 [.995800,.998574]|1.000299|
| v32 |1.000132 [.998087,1.002178]|1.000536 [.998543,1.002528]|.999751|

The page map is moved out of the loop, and native v16 improves RAW2 by
0.28%, but does not beat S19's FA3 point. This does not recover a native5us
page-coordinate term: the oracle retained cold64-bit offsets and a different
loop/binary. Do not transfer that attribution to native paged4-byte loads.
Other copy issue, query/value preparation, transpose, max/sum/packing and
merge work remain. No causal phase clock was added to H, so single-capture
CUPTI increments are calibration rather than per-seed component timings.

**Hand d32 H to P4** by the per-shape point rule. H/S19.999373 hasSE.000784,
CI crossing1 and an A:A.998277 point; H/RAW1.000746 does not establish a
shader win. The specialization-only RAW control also has d32 points
.998061/.998637 in the two cohorts: these are preserved explicitly as
control/module alternatives, not presented as page-map mechanism evidence.
All seed IDs and SEs are in `P4_HANDOFF.json`, with every full result in
`result.json` and `hoist_result.json`.

The selected d32-only shipment**557deae0** passes exact normal/forced
13TASK+12graphs+6static each. `s19_patch.py selected` rebuilds it; its CUDA
bytes match measured dHA. `P4_d32_page_map.patch` is the full expanded-S19
integration delta; `P4_d32_page_map.cuda.patch` is the readable shader delta.
Keep every other S19 selection, full attention, PDL and frozen artifact.
Fold the weak point into P4's next held-out six-load batch; G0 remains
unconfirmed. No standalone small-gain repeat and no P5 job remain pending.

All**568 native TASK** gates across the two runs passed. The old first-run
wrapper used a dynamic direct lab lookup that static lint cannot resolve;
`wrapper_rebuild.json` proves its post-run globals-lookup repair changes no
GPU body. The immutable first receipt/source still pin the measured bytes.
All rebuilt-wrapper static checks pass; no test was weakened or suppressed.
Next work must target a larger surviving term, preserving merge parallelism
and separating actual copy/transpose/max/sum/packing from residual oracles.
