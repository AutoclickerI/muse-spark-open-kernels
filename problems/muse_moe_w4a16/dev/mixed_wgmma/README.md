# Grouped NVFP4 RS WGMMA

Free lane. No H200 win is established; no candidate was sent and no commit was made.
The measured losing scalar/plane adapter is in decode `dev/parked/cand_r3_mixed_wgmma.py`.
The packed producer is lab-only until the H200 seed comparison passes and wins.

## Structural diagnosis

Actual sm90a cubins rebuilt by `dump_sass.py`, audited by `sass_audit.py`:

| Producer | K step | Static loop instructions | Instructions / packed weight byte | Registers | Spills | BF16x2 multiplies / step |
|---|---:|---:|---:|---:|---:|---:|
| Scalar |128|2046|63.94|122|0|64|
| Plane, scalar conversion |128|1636|51.13|118|0|64|
| Packed fragments, `vector.cu` N32 |64|212|13.25|78|0|16|

These are static counts including skipped normal/subnormal branches, not executed instruction
or cycle counts. Both old loops have 64 FLO/CLZ operations, 203 branches, 72 BSSY and 72 BSYNC.
The scalar converter repeats E4M3 conversion once per scalar value and uses duplicated halves
of BF16x2. Neither spills nor a wait after each MMA explains its loss: each K128 has eight K16
WGMMA instructions and one group wait0. Weight/scales use two-stage cp.async; activations use
synchronous LDG128/STS128. `diagnostics/sass_counts.json` has complete opcode counts.

`vector.cu` converts a whole packed register fragment at once. A generic 256-entry E4M3 format
lookup is built inside each CTA; timed kernels read raw E4M3 bytes and original E2M1 values
under bijective bit/layout permutations. There is no prepared dequantized weight/scale bank.
It stages Q, raw SF and gathered activations with two cp.async groups, decodes 32 weights into
16 BF16x2 registers per thread/K64, then issues four K16 RS WGMMA and one group wait0. The main
loop has no FLO/CLZ; raw E4M3 lookup replaces repeated scale arithmetic. Temporary 2^-8 product
scaling is undone by FP32 GS*256 in the epilogue. GPU routing/hot-first N splitting and
post-GEMM arithmetic remain shared with the original engine. V9's kernel is unchanged.

The extension explicitly includes sm90a and sm120. Local sm120 executes identical
loads, scale lookup, register layout and epilogue using MMA.sync; Hopper selects WGMMA.
A local pass does not establish Hopper WGMMA math or timing.

## Measurements and pending labs

| H200 receipt | M128 | M256 | Larger M | Status |
|---|---:|---:|---|---|
| Scalar `296fdf45382244dc` |2060 us vs V9 280|2648 vs480|M3012 22950 vs Marlin2818; M16384 123615 vs14754|Correct, rejected|
| Plane `01e74ee69c8845e1` |pending|pending|512/753/3012/16384|Receipt absent at turn's one check|
| Prefill controls `70206a2a501047eb` |—|—|raw17/masked-Humming/dg_w8 vs plane/Marlin|Receipt absent at turn's one check|

`prof/lab/moe_w13_ablation.py` uses exact R0 IDs at128/256 (209/414 pairs,18/21 experts,
hot55/119). It times W13 only: original, conversion replaced by raw bit reinterpret, constant
weights with weight/scales loads removed, no MMA, and132/264/528 CTAs. Wrong-value variants
are prepare-only and never returned as a candidate. SASS confirms conversion remains in no-MMA
and loads disappear from constant. The no-MMA checksum adds31 XORs/K128 and one global store
per thread/CTA to keep conversion live; it retains warpgroup synchronization. Reported B/s
uses nominal Q+SF requests or unique expert bytes divided by time, not hardware DRAM counters.

`prof/lab/moe_vector_wgmma_seed.py` compares fixed N16/32/64 against current raw-E4M3 V9
NT1/NT2 over12 public weight seeds5353+7919*i, plus exact R0 IDs at both M. Each draw uses
three fresh real-router inputs,8 rotated/reversed cold graph timings and128MiB L2 writes.
It prints each variant's mean/median over seeds; no per-seed best is used for a win claim.
FP32 gates require finite outputs, L2<=1%, row<=5%. R0 uses fresh public weights/activations,
not the original captured job's unexported weights. W13/W2 phase graphs are diagnostic.
The old unsent build omitted the explicit90a target and was stopped before a judge job.
The ablation queue source is `4152e9ea747e0a07`; the final dual-target vector source is
`6649d92cf810f3c9`. Payload SHA256 values and screens are in `admission.json`.

The two earlier pending labs still cover prefill/serving512..16384. No serving producer changes
until a legal kernel wins a matched H200 comparison. If it wins, replace only expert GEMMs in
`muse_spark_moe._grouped_call`; keep gather-only normalized input, per-pair outputs and EP
collectives. Serving may have distinct gate/up GS, so that epilogue must accept both FP32 scales.

## Local evidence and reproduction

- Original scalar/plane converters pass130048/1040384 finite format combinations bit exactly.
- `check_vector.py`:27 refreshed graph checks at8/128/256 withN16/32/64 and independent FP32;
  another18 graph checks use signed/zero/subnormal raw scales with all-remote and hot routing.
- Vector seed payload: all26 core/R0 draws, five contenders and three fresh inputs pass FP32;
  nine proj-host official tests and eight static checks pass. The explicit dual-target rebuild
  is screened separately in the current admission manifest. Local timing is not an H200 claim.
- `check_layouts.py` inverts canonical V6 gate/up and down banks (K4096) and image Humming's
  actual MMA/WGMMA repacks in PyTorch. Codes/scales/FP32 per-expert GS recover exactly; FP32
  GEMMs match exactly with varied independent GS, signed scales, zero and subnormals.

The 401.5us `v11` in c8c89f9f comes from `hu__prepare_humming` in
`prof/lab/moe_real_routing_driver.py:334`, not canonical V6 RS. Prefill `raw_v11` is another
path, expanding canonical weights inside each timed call then using grouped BF16 GEMM.
The recovered layouts rule out a general Python permutation/missing-GS error on tested data;
they do not prove Hopper backend agreement or reconstruct the original decode weight seed.

Build with `build_vector_lab.py` / `build_ablation.py` inside tools/img. Screen the assembled
proj-host file with tools/local_eval.py before tools/h200_lab.py send. The .inc files are
assembly fragments; static checks run on generated standalone payloads. Never poll the judge.

## CUTLASS quarantine

`check_cutlass_host.py` reproduces the original CuTe TMA abort on local sm120 even when all
experts are nonempty. Grouped setup left an all-ones dummy shape and zero scale strides;
fixing these exposed an independent eight-byte scale box, below TMA's16-byte minimum.
`decode/dev/omni/mixed.cu` now uses capacity metadata, canonical strides and a16-row scale
atom. The test calls both grouped Gemm::can_implement and real host TMA setup, including
exact R0 counts and empty experts, and rejects malformed scales. It launches no mixed GEMM.
The embedded engine/factory/dispatch are removed from cand_r3_omni, whose normal/forced
screens pass13 official tests; the latter covers12 graph shapes and14 static checks.

Stock CUTLASS directly names float_e2m1_t but broadcasts one scale per tileK64 and rejects
group16. Bypassing that check cannot implement the required math. The prior grouped probe
560d1adb was wrong and slow; it also passed host stride arrays as device data and omitted
FP32 global scales in its alpha=1 epilogue. No new CUTLASS judge slot was spent. The private
group16 specialization remains unvalidated numerically and stays out of candidates.
