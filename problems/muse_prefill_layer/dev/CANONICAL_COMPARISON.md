# Canonical V6 reader comparison

All timings are local sm120 measurements on the RTX PRO 6000 Blackwell; these execute the MMA.sync stand-in branches, not Hopper WGMMA. They cannot establish the H200 winner.

Use RS128 with adaptive N=8/16/32/64/128 as the provisional serving grouped reader. It wins every measured shape and both routing distributions. Keep the canonical v3 pipeline in the reset-time H200 diagnostic; the architecture difference makes a permanent H200 dispatch decision premature.

The same canonical V6 bank, BF16 normalized input, FP32 gate/up global scales, BF16 SwiGLU/down outputs and BF16 effective post-norm sidecar are used throughout. Captured total includes routing, both expert GEMMs and finalization; the pipeline also gathers a V6-permuted tiled activation buffer. Pre-expert normalization is common and excluded. The grouped full-layer candidate reuses frontend counts/slot positions instead of repeating benchmark routing.

Cold-L2 measurement: 256 MiB cleared outside each timed CUDA event, two captured graph repetitions, lower median (PyTorch median). Stages are timed separately and do not add exactly to captured total. The separate original warm-L2 JSON files remain as explicitly labeled warm evidence.

| Routing | Tokens | RS adaptive (ms) | RS128 adaptive (ms) | Canonical v3 pipeline (ms) |
|---|---:|---:|---:|---:|
| uniform | 753 | 1.260 | 0.979 | 3.934 |
| uniform | 3012 | 3.503 | 2.352 | 4.218 |
| uniform | 16384 | 14.431 | 9.805 | 18.073 |
| one hot expert | 753 | 0.700 | 0.659 | 1.182 |
| one hot expert | 3012 | 2.580 | 1.866 | 3.454 |
| one hot expert | 16384 | 12.558 | 8.608 | 15.800 |

Uniform routing sends 770 / 2922 / 16189 local pairs at 753 / 3012 / 16384 tokens, touching all 32 experts. The hot diagnostic sends one local pair per token to one expert. These are synthetic routing fixtures, not receipts from the judge routing seed.

All three readers pass independent FP32 dequantized math at all three sizes, with unequal gate/up scales (up = 1.375 * gate). Relative L2 error is below 0.058%, worst row below 0.303%. The same captured graph passes after changing input activations, probabilities and routing; replay is bit exact against eager on the changed data.

Full candidate `cand_canonical_grouped.py` additionally passes the five official local cases, four nearby fresh-input graph checks, and forced SM90 prepare/FA3 tensor and tuple glue through native stand-ins. It asserts one 905,969,664-byte V6 bank. Actual SM90a compilation succeeds for every grouped width/stage, using 70–195 registers and zero spills; execution on H200 remains unverified.

Frozen candidate SHA256: `17206b59f6548cbd999da5cc786aa6ca96066f1c017db67c6730e7eb59b7c07f`.

Frozen adapted RS128 CUDA SHA256: `303a09989ed5aa3c582bd4541e05609d7cf8338d586c565df0e7603fa1d57bd8`.

Artifacts: `canonical_comparison_cold.json`, `canonical_comparison_hot_cold.json`, `canonical_baselines/SHA256.json`, `local_canonical_grouped.json`, `graph_canonical_grouped.log`, `hopper_glue_canonical_grouped.log`, `compile_canonical_grouped_hopper.log`.

Baseline snapshots remain byte-for-byte copies of the sibling sources. The private source adapter adds distinct FP32 gate/up scale pointers, required by the serving layout contract; it does not change the bank bytes. No sibling file was edited.

```sh
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_prefill_layer/dev/compare_canonical.py --tokens 753 3012 16384 --fixed-n 0 --custom --reference --repeats 2 --cold-l2 --json problems/muse_prefill_layer/dev/canonical_comparison_cold.json
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_prefill_layer/dev/compare_canonical.py --tokens 753 3012 16384 --fixed-n 0 --custom --distribution hot --reference --repeats 2 --cold-l2 --json problems/muse_prefill_layer/dev/canonical_comparison_hot_cold.json
```
