# Full decode16/32 softmax batch

Base: S14 final **f1c77ce6**, full-decode `attn_main` CUDA **498e93cc**.
This is P5b: full attention at B16/32 only. Sliding and decode1 remain with P5.
No variant is promoted.

Queued on proj: wrapper **06e39759a67ca297**, sender PID**949154**, not-before
epoch**1792972800** (2026-10-26 09:00 KST), priority**2.5**. No judge job ID
exists before acceptance. The unchanged verify batch is wrapper03086413,
PID3171016, priority2.0 at the same time gate. Both live in runs/waiting until
their sends. The new sender log is runs/detached/06e39759a67ca297.log.

**Decision rule:** the decode batch is independent of the verify reset outcome
and queues now behind it; after both receipts land, generate a patch only for a
native decode arm whose candidate/incumbent ratio and seed spread beat A:A,
then require exact normal/forced whole-layer screens and a fresh paired layer run.
If verify's preferred technique loses in decode, retain the decode incumbent.

## One batch, ten arms, eight seeds per shape

`build.py` freezes the original S14 code and the exact source strings of the
queued verify batch **03086413**. It verifies that queued file's full-content
identity before using it and never changes it. `build_lab.py` packs every arm
into `decode_softmax_lab.py`, with eight fresh seeds ×eight rotating cold
replays per full shape and a duplicate incumbent A:A control. All keys are read.

| Arm | Actual kernel and change |
|---|---|
|ref|S14 four-warp FP16-MMA decode incumbent, with compiled-out clock hooks|
|mix25 /mix50|25/50% of probability exps use the verify study's FP32 degree3 polynomial; remaining exps and running-max corrections use MUFU|
|tc|Row sums from the actual FP16 P operands, using two m16n8k16 MMAs per32-key warp slice|
|tc25|Tensor-core sums plus25% mixed exponentials|
|wgref /wg25 /wg50|Frozen native two-warpgroup kernel at q_len1, native/25/50% exponent split|
|ping /ping25|Explicit two-warpgroup ping-pong at q_len1, native/25% split; inactive query warps use the correct live-row predicate|

Decode `attn_main` differs from verify: four32-thread warps use FP16 MMA and
FP16 P fragments; the cached64-key CTA tile splits into two32-key warp slices
per KV head. Its TC denominator therefore uses FP16 P and exact half ones,
not the FP8 WGMMA denominator from the verify batch. The final four-lane row-sum
reduction is preserved. An independent m16n8k16 probe checks all16 rows and8
columns against the FP64 sum of the exact half P values (maximum error0).
It would catch a denominator/layout bug that output RMSNorm could conceal.
The mixed-exp probe's worst relative error is0.000187975 over[-125,0].

Native resources: all five four-warp decode main kernels remain120 registers
in plain builds; the unused q_len8 sibling has231–239 and is not a claim about
decode occupancy. Two-WG forms use159–165. `native.json` checks all20 complete
plain/clocked sm90a units: no spills and noC75 serialization warnings.

The two-WG family can lose despite winning at verify: it still carries padded
query rows at q_len1. It is measured against `attn_main`, never merely against
its own slower family. This makes the transfer test decisive rather than a
promise that a verify win generalizes.

Clocks report per-tile gather wait/sync/page issue, QK, softmax, PV, max,
alpha/rescale, exp/sum and packing/TC sums for the four-warp path; and consumer,
softmax, QK/PV issue/wait and producer phases for the two-WG path. The plain
paired time, not the clock sum, selects a winner. There is no local timing
promotion claim. Both full shapes use exactly the incumbent split/cluster plan
for every four-warp arm, separating math changes from scheduling choices.

## Screens

GPU0 in the judge image:

- `check.py`: ten arms ×four fixtures ×two fresh seeds =80 attention/cache/
  capture checks. Fixtures include full B16/B32 context80K, page64 with KV
  scale1/64 and context4096. Every case also checks a captured strided
  three-FP32-partial ABI, three ordinary graph replays and plain/clock bitwise
  parity. Worst error is43.4% of its unchanged TASK bound.
- `screen.py`: tools/local_eval normal wrapper plus every arm with Hopper
  dispatch forced. Each run passes16 TASK tests, four captured decode16/32
  shapes (full and sliding) and7 static checks. Each forced arm is counted
  eagerly and in a captured call at both full shapes. sm120 executes the real
  ring/cache/reduction code and explicit arithmetic stand-ins for native
  WGMMA. Native WGMMA remains compiled and gated on H200 before timing.
- `native.py`: the20 strict native proofs described above.
- The exact weighted proj wrapper passes9 TASK tests and3 static checks; its
  real prepare executes all ten arms at both full80K shapes, eight seeds each,
  eight rotating cold replays, A:A and all phase clocks. The complete local
  lab takes26.4s after JITs are warm. This is an admission/coverage fact, not
  an estimate of cold H200 compile time or a performance ranking.

The first concurrent check and weighted-host lab hit GPU0 exhaustion when
other jobs occupied nearly all95GiB. This checker itself held about1.6GiB;
no other process was touched. Synthetic cache generation now uses8192-row
temporary chunks instead of1,048,576 rows. Context sizes, seeds, key counts,
distributions and all gates are unchanged; all80 fixtures were rerun after
that change. The failed local batch was never sent. `local_resource_failure.json`
records it. Independent variants see identical generated tensors in each pair.

## Reproduce and integrate

```
python build.py
python build_lab.py
CUDA_VISIBLE_DEVICES=0 tools/img python native.py
CUDA_VISIBLE_DEVICES=0 tools/img python check.py
python build_screen.py
CUDA_VISIBLE_DEVICES=0 tools/img python screen.py
```

`patch.py <arm>` is ready to generate a selected cell onto exact S14 f1c77ce6.
It routes only full decode16/32 and leaves all other attention dispatch intact.
FP32 projection partials are reduced and rounded at the existing BF16 boundary
for `attn_main`, while the native WG prologue retains its direct partial ABI.
Generated patches are experimental and say promotion=false. Do not integrate
an arm before the native paired result and whole-layer screen/pair.

Queued sender identities, time gate and the exact weighted-host screen are
recorded in `admission.json`. The verify reset sender
PID3171016 /03086413 remains unchanged, priority2. The decode batch uses a
lower queue priority and the same not-before epoch1792972800 (09:00 KST).

The wrapper and generated device/driver files are ignored build outputs;
okc_run has frozen the sent bytes in runs/sent. `build_lab.py` and h200_lab
regenerate the wrapper. `patch.py` is ready for the selected native winner;
no unsent performance-dependent judge payload remains to be built.
