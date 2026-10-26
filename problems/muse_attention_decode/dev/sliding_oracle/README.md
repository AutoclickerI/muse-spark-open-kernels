# P5 sliding mechanism oracle

This ranks the next structural change at d16/d32 and v16/v32. Baselines are
the native one-launch DSM D8/D4 decode kernels and RAW2 verify, plus exact
S15 FA3 at the verify loads. All keys, FP8 Q/PV and the existing split geometry
remain in the full-work controls. The scalar reference sees each kernel's
own appended cache entries and uses the public TASK checker.

The altered-work arms are measurements inside an import-time lab. They do
not feed the host's timed outputs, never count as candidates, and never
produce a speedup nomination.

| Arm | Change from the one loaded M kernel |
|---|---|
| O/O2 | Unmodified original kernel and identical captured control |
| M/M2 | Oracle-capable binary in full-work mode and identical control |
| C | Precomputed per-request/head 64-bit byte offsets replace page lookups/address generation; cold offset reads remain |
| S | C, with constant FP8 probabilities instead of historical and appended-tile max/exp/denominator/packing; QK and PV still execute |
| N | Full M math, skip partial publication/DSM merge/norm/gate |
| X | S with the same merge skipped |
| L | Precomputed offsets, all volatile KV copies and waits, query prep; bypass transpose/QK/PV/softmax/merge |
| Z | L with zero-byte cp.async, retaining addresses and copy issue |
| E | Same kernel launch/cluster geometry, immediate uniform return |
| W | Full M math with the ready-input dependency wait bypassed; diagnostic only |
| G | Empty CUDA graph, to measure event/replay overhead |
| F | Exact S15 five-kernel FA3 verify path |

M-C, C-S, S-X, X-L, L-Z, Z-E and E-G telescope to M-G. This is an
**order-dependent exposed removal cost**, not a sum of independent hardware
utilization counters. M-N separately measures merge removal with real
softmax; M-W measures only the ready-input PDL protocol. An active QKV
producer's wait and contention cannot be inferred from W. The structural
follow-up must measure the real projection-to-attention context.

The precomputed-address oracle reads 8 bytes/key/head instead of the
original 4-byte page entry. Its instruction savings come with that cold
traffic; a loss closes this oracle recipe rather than all coordinate
hoisting. A native clocked copy records page-load issue, address-dependent
copy issue, copy waits, V transpose, QK, softmax/packing and PV cycles,
plus global-timer prologue/publication/merge phases. These diagnostic bodies
have different code allocation; uninstrumented paired seeds rank arms.

Native admission: dA/dM have138 registers and32B stack, no local allocation;
dCLOCK140. vA172 registers, vM155, vCLOCK161; verify has no stack/local
allocation. All six contain24 LDGSTS and9 QGMMA instruction sites, no C75
warnings. Retained [native.json](native.json) records the exact source
hashes/resources. M/O is explicitly measured to expose the compiler change.

Local GPU0 screens pass36 public TASK checks and execute30 altered-work
branches, including fresh captured inputs, scale1/64, page64 and edge
fixtures. Decode uses an explicit C1 DSM stand-in with every key; verify
uses two real WGs and a nonspinning global last-WG combine. The exact packed
payload passes56 TASK gates with these stand-ins and portable FA3, including
all oracle branches and clock probes. No local time is H200 evidence.

Sent attention check **f365764142034130**, wrapper **18781d29**, payload
**768703f7**: all four loads,12 fresh seeds x8 cold READ rounds, fresh graph
pools and rotated captures. Sender3354514. Receipt is complete: **312
full-work arm/seed TASK gates plus4 clock-output TASK gates** pass.
[oracle_result.json](oracle_result.json) retains paired95% t11 intervals,
seed IDs, full native symbols/resources and phase clocks. Immutable source
and receipt are in runs; [queue.json](queue.json) pins both.

The exposed ordered split (us) is:

| Component | d16 | d32 | v16 | v32 |
|---|---:|---:|---:|---:|
| Launch/cluster scheduling, E-G |2.168|2.195|2.063|2.212|
| Remaining query setup/address-copy issue, Z-E |4.556|6.087|7.269|7.331|
| Precomputed-address removal, M-C |-0.120|-0.304|0.226|-0.107|
| KV transport without math, L-Z |1.895|4.489|2.432|4.595|
| Softmax/packing, C-S |1.167|2.454|5.368|5.320|
| QK/PV/transpose/new-KV, X-L |1.908|2.404|3.968|3.569|
| Merge after constant P, S-X |2.691|2.219|4.910|4.818|
| Total oracle-capable M-G |14.264|19.544|26.236|27.737|
| Unmodified O-G, for calibration |13.994|18.612|25.663|27.030|

With real softmax, M-N removes2.682[2.604,2.760]/2.632[2.469,2.795]/
4.950[4.824,5.076]/4.980[4.853,5.108]us. Constant-P replacement ratios
S/C are0.932858/0.892509/0.814772/0.827377. Precomputed offsets yield no
large exposed gain: C/M1.007013/1.013572/0.992287/1.003528. Do not clip
the negative removal estimates or call this a general coordinate rejection.

O2/O is0.992690/0.996913/0.994085/0.994747; M2/M is
0.996239/0.994544/1.001899/0.997449. The roughly0.5–0.7% directional
drift remains uncertainty evidence; subpercent coordinate/wait effects are
not resolved. M/O is1.0160/1.0433/1.0201/1.0236; the large softmax and
merge removals are measured within M, not attributed directly to O.

The native verify clock records lack first-copy and publication stamps
(phases1/7), because those source anchors differed from decode. Those two
phase costs are unmeasured in this receipt; wall-time removals, other clocks
and TASK results are intact. New real-producer clock bodies repair the
anchors; the sent oracle generator/source hashes remain unchanged.

## Structural response: probability exponent block

The verify softmax/packing ceiling is about5.3us, compared with less than
0.3us from this coordinate oracle. [structural.py](structural.py) changes
only the per-key exponential in both historical and appended-key softmax.
It constructs a FP32 exponent/mantissa through an FMA and integer
conversion; online maxima, exact row-rescaling exponentials, FP32 denominator
sums, FP8 P packing/PV, every key and original DSM merge are retained.
It performs no input-derived precomputation. This is approximate math at
the existing formats, assessed by TASK rather than a new error threshold.

Across500001 sampled x in[-120,8], the scalar exponent's relative error
is[-2.9383%,+3.0289%]. The production function handles masked/FTZ-range
inputs separately. This characterization is not a correctness gate.
The24 component TASK checks pass on fresh data and strided FP32 partials,
including short/page64/scale1/64/edge fixtures. Candidatebabd11fe passes
normal and forced13TASK/12graphs/25static each; evidence is
[layer_admission.json](layer_admission.json) and the linked full records.

Native static SFU exp sites decrease91→23 at decode and78→10 at verify;
FMA/integer conversion issue grows. QGMMA9 and LDGSTS24 sites are unchanged.
Regs remain138 at decode and172→167 at verify; shared memory/cluster
geometry is unchanged. Static counts do not establish a speedup.
[structural_native.json](structural_native.json) retains all native bodies,
including corrected baseline/candidate producer-context clocks.

Whole-layer follow-up **7bd3eced41e24402**, sender3610930, wrapper0f1a49c5,
payloadf23abf6c:16 disjoint seeds x10 cold rounds at all four loads.
INC isD8d16/D4d32/FA3v16/RAW2v32; INC2 is identical, RAW adds native RAW2
at v16 to isolate the probability block, EXP changes only that block, and
A is exact S15. Expected320 full-layer TASK gates plus8 real-QKV producer
attention TASK gates. The exact payload passes40 layer+8 producer attention
TASK gates locally with forced glue and explicit DSM substitutes.
These probes preserve actual partial layouts (d16 qkv3/gate split included),
record real PDL wait and clocks, and do not time a synthetic delay.
Native result is pending; no new EXP nomination yet.

Decision on the follow-up: establish every TASK count, source identity and
INC2/INC controls. Rank EXP/INC, and EXP/RAW at v16 separately. Every
affected measured point below1 goes to P4 with CI/SE/seeds and exact source;
P4 confirms G0 on held-out six-load pairs. A loss closes only this exponent
construction, not the softmax lever. No standalone small-increment run is
prepared. Failed implementation/control is unmeasured.

```sh
python3 problems/muse_attention_decode/dev/sliding_oracle/build_oracle.py
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_attention_decode/dev/sliding_oracle/screen.py
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_attention_decode/dev/sliding_oracle/native.py
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_attention_decode/dev/sliding_oracle/screen_lab.py
python3 tools/h200_lab.py read f365764142034130
python3 problems/muse_attention_decode/dev/sliding_oracle/analyze.py runs/muse_attention_decode/20261026T062008Z_check_f365764142034130.json
python3 problems/muse_attention_decode/dev/sliding_oracle/layer.py
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_attention_decode/dev/sliding_oracle/screen_structural.py
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_attention_decode/dev/sliding_oracle/screen_followup.py
python3 tools/h200_lab.py read 7bd3eced41e24402
```
