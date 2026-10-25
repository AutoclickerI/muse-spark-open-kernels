# Decode-load projection prefetch (P7)

The H200 `feb7b287e3464e4a` standalone experiment matches its plain-read
control at M1 and loses at M>=8. [PROFILE_MAP.md](PROFILE_MAP.md) maps best6's
actual in-layer projection dispatch, grouped timings and one-read floors.
The 2.5TB/s result is the measured envelope of that implementation/protocol,
not a proof that no future single-launch kernel can improve it. This lab
tests overlap as the next practical lever.

`build.py` freezes best6 source `4516cb97` and parses its receipt `1b80ffc0`.
It emits `../l2_overlap_lab.py` and `prefetch.py`. The payload builds only the
raw E4M3 V9 bank and native code it needs for M1/16/32; it does not rerun the
whole best6 tuner or mutate the host state. It uses the same cuBLAS/Lt/custom,
attention, NT1/NT2 and PDL choices. The log tail lost the M1 boundary geometry;
best6's own three-draw selector supplies those missing settings in this run.

The thirteen recipes combine completed 16-byte loads (4/8/16 CTAs) and bulk
L2 hints (4 CTAs). Each load contributes to a private checksum sink, and the
byte-pointer ABI explicitly converts a CUDA pointer to PTX global space.
No local architecture skip exists: both transports and all stream/event code
run on sm120. The native sm90a audit confirms 128-bit loads, `UBLKPF.L2`, and
zero spills (44/18 registers).

The front fork starts after qkv, overlaps attention, and joins before o.
It fetches o's16MiB and optionally the first16/24MiB of pool/router. Another
recipe fetches that pool/router region during o/boundary instead, joining
before pool/router. The tail fork fetches8MiB during MoE; M1 uses the entire
unpool slice, M16/32 use a bounded prefix of full unpool. One recipe tests24MiB
at M16/32. Maximum intended front footprint40MiB, tail24MiB; all selected
regions are64KiB aligned. Bank sizes are16/72/8-or-64MiB, so keeping both
complete o and pool/router banks in a50MiB L2 is impossible. Cache contention
with attention/MoE is measured rather than assumed absent.

Each call captures the fork and join with a private side stream and separate
event pair per phase. No access-policy window, cache reservation or eviction
priority survives the call. Normal L2 eviction applies. The bulk instruction
is a hint and may remain in flight after its kernel returns; correctness does
not depend on cache residency. See [NVIDIA PTX bulk-prefetch specification](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-prefetch).

The lab compares the exact best6 kernel entry point with the prefetch layer
on the same tensors, both layer types at M1/16/32, with128MiB flushed outside
events before each individual replay. Three public soft-stream draws refresh
attention input, stream and busiest EP range in an80K public cache/length
fixture. All recipes get six coarse AB/BA rounds; two finalists get24 rounds,
eight per refreshed stream, plus an A/A timing control. A native activity
profile follows decision timing and reports Q/attention/O/boundary/pool/route/
MoE/unpool/boundary/prefetch durations and prefetch intersection with attention
and MoE. No timing events are inserted between stages, preserving PDL.
The stage classifier uses native names and execution order: Q includes QK
preparation and any split-K reduction before the first attention kernel.
Activity durations include PDL waits; their sums are diagnostic, while paired
whole-layer events measure the performance decision.

The independent gates cover raw projection outputs (FP32 1e-4, BF16 1%), the
original FP32 router at1e-4, selected route weights, top-eight scores, normalized
expert input, V9 against its FP32 oracle, both residual/norm boundaries, FP32
attention and FP8 stores. Prefetch A/B route IDs must agree exactly. These
gates run before and after graph timing. Numerical A/A between two separately
allocated no-prefetch graphs is printed alongside timing noise.
The mixed-BF16 full-layer A/B bound is0.3% (IDs exact). It is separate from
the raw/router FP32 bound: even a no-prefetch eager-versus-graph comparison
exceeded1e-4 in the FP32 stream after BF16 branch rounding. That failure is
preserved in `runs/screen/p7_l2_eager_capture_gate.log`; raw FP32 stages retain
their1e-4 gate and both residual epilogues have independent FP32 references.
The compact `fp32=` field is the worst projection error against an FP32
reference, including BF16 outputs; it is not the router's separate error.

`screen.py --share-s 80` runs the exact prefill-host wrapper locally. Only
Hopper attention launches have portable stand-ins; prefetch loads/hints and
fork/join are real. No local speed ranking enables a best7 setting. The best7
projection cell accepts `{'l2_prefetch': recipe_id}`; absent/zero disables it.
Only explicit decode q1 rows with M<=32 are admitted. Every enabled row gates
three refreshed full-layer graphs against no prefetch during prepare.

Sent on prefill base cb8939d0 (send key **9367398c624b7796**, payload SHA256
`e62b6b087adeb0c5744c79f3f0597039e799f316908dc82951c9d9ee0e3fda55`); H200 job
`d5a9025890a64141`: its best recipe still slows the layer 0.3-16.6% (FINDINGS).
Local admission:78 coarse recipe/cases,12 final pairs,6 native activity
profiles,5 official host tests and3 host static checks; no rejects. Cached
payload12.0s inside an80s share,2442 output characters. `validation.json`
and `compile_sm90a.json` pin the source and native evidence.

Best7 transport tests are recorded separately in `best7_admission.json`.
The decode owner changed its default NT/attention table while screens ran;
the original per-screen SHA is retained rather than claiming that both tests
were of the same table. `check_best7.py` snapshots the live candidate before
forcing all six decode prefetch rows, so later table edits cannot relabel it.
Frozen source `08bbc92837594c2c` passed13 official tests,12 fresh-input graph
shapes and16 static checks, with all6 real prefetch instances,24 streaming
instances and both transports. This verifies the transport hooks, not later
default-table changes; the decode owner owns the cumulative candidate screen.

An instance owns its stream/events/sink and belongs to one ordered caller.
Concurrent root streams require separate instances. The unsafe historical
`overlap_lab` is never reused; this payload runs the actual path locally.
