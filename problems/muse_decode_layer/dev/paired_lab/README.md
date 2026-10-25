# Paired decode measurements

The current E extension is [decode_scale_bank.py](../../../../prof/lab/decode_scale_bank.py),
SHA-256 `e9b0e04ca74a063db131ad489194c8527c0abd84e460bfc4d4cad435f49e9cfc` (parallel cold builds).
It needs board v9's decode-host state; the host source is `0e94268cf5654012`.
[Manifest](scale_bank_manifest.json) pins the source, public seeds and exact best7/raw NT1 definitions.
No prepared BF16/FP8 expert copy is made; both banks remain packed NVFP4 with scale sidecars.
A lossless recode in prepare is allowed by CONSTRAINTS 5.

## E:C — isolate scale conversion

A is best7's exact raw-E4M3 NT1 reader/bank; B calls the board host's own `v9_kernel_fn`
with its actual prepare-recoded bank. Their input tensors, expert IDs and routing weights are
identical. The payload checks each bank against its own packer and asserts matching NT1,
8-token tile, stage count, grid and shared memory at M8/16/32/128. It leaves host state unchanged.

The fixtures use public `generate_input` at ctx80000/page1/soft, agentic B16/32 lengths,
seeds 172903/180822/188741 and both sliding/full layers. The real QKV/attention/O/residual/router
prefix supplies activations and the busiest EP's routes. Eight cases × three seeds = 24 fixtures.
Each has an independent FP32 gate before/after replay (<=1% L2, <=5% worst row), direct A/B
agreement, six alternating AB/BA rounds, a cold 128 MiB L2 flush before every replay and A/A noise.
Absolute times include equal graph-replay overhead. Every per-seed sample is retained locally;
compact H200 lines report per-case seed ratios, mean us, routing counts and per-M means.
Budget exhaustion prints INCOMPLETE and never supplies a complete verdict.

`E8:C`/`E9:C` are M128 sliding/full. B:A <1 favors recoding. Result (`58ae2e2cf8df45ef`):
recoding wins at every M, B:A 0.914/0.908/0.897/0.880 at M8/16/32/128; S1 adopts it.

## Settled G/F/E

[decode_paired.py](../../../../prof/lab/decode_paired.py), payload `bf3fd04bd323d78a`, completed
in H200 receipt **9c83cdb852fe4576**: 36/36 scored-case/seed fixtures, 216 comparisons,
zero rejections and zero flipped tokens; 114.8 s payload, 160.7 s total host prepare.
- G: fixed new glue wins at decode1/verify1; old glue wins at decode32 sliding; others retain new.
- F: public `layer` q8 scheduling ties every split at large loads and beats explicit splits at verify1 full.
- E: NT1 wins below M256 (M128 317–320 us vs Humming 346–353); Humming wins M256 (417–426 vs 537–547).
The earlier failed receipts supply no stage verdict. The frozen best7_paired candidate applies
these supported changes.

## Reproduce

```bash
python problems/muse_decode_layer/dev/paired_lab/build_scale_bank.py
python tools/h200_lab.py build prof/lab/decode_scale_bank.py --host muse_decode_layer \
  -o problems/muse_decode_layer/lab/lab_decode_scale_bank.py
CUDA_VISIBLE_DEVICES=0 tools/img env MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=12.0 \
  python problems/muse_decode_layer/dev/paired_lab/screen_scale_bank.py --share-s 80
python tools/h200_lab.py send prof/lab/decode_scale_bank.py --host muse_decode_layer \
  --note "E:C board recode vs best7 raw NT1, identical public routes M8/16/32/128"
```

The screen fails if the wrapper catches a lab exception, or if any fixture/seed/round is missing.
Only unsupported Hopper attention launches use portable stand-ins; both MoE readers run natively
on sm120. The first local pass completes all 24 comparisons in **4.4 s**, with **0.167% L2 /
0.176% worst row**, plus **13 official tests, 12 graphs and 5 host static checks**. These are
correctness and budget evidence, not H200 latency. Embedded raw NT1 is separately compiled for sm90a.

## Send and cold-budget status

Accepted **58ae2e2cf8df45ef**, send key **6edfaa87ad96a5ca**, section `decode_scale_bank`,
contains payload **8e01adcec8da5a0f**. Its [send validation](scale_bank_sent.json) pins the screened
24-fixture source. This payload rides alone, with the full remaining prepare budget. The subsequent
empty-cache test exposed serial raw-MoE plus prefix-attention compilation taking **123.5 s**, too
long for an 80 s batched slot. The current builder overlaps both builds before graph capture; its
cold 80 s screen passes **24/24 fixtures in 56.4 s**, including all **254 × 16** finite scale/FP4
represented-value checks. [Validation](scale_bank_validation.json) pins this revision. The arithmetic, fixtures, readers
and timing protocol are identical between the sent and shared versions.
