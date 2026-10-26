# S19 native one-boundary projection decisions

Completed: **no new G0 nominee**. All24 dispatch point estimates lose in the changed load’s whole-layer time; the closest is statistically inconclusive. P4 receives the negative results in `P4_HANDOFF.json`; its accepted optional native-layout-only arm from `large_split` remains unchanged.

Sizing before the build (`e70024b1`): small-M completion-critical surplus over the actual coded-byte stream target gives **R1.032884 [1.030488,1.035286]**, with2.910/13.278/8.393/2.927us per weighted layer at d1/d16/d32/v1. Large producer surplus plus complete-consumer removal gives an upper scenario **R1.038411 [1.037944,1.038878]**,26.517/52.635us at v16/v32. These model CIs do not validate the assumptions or establish a physical ceiling. See [SIZING.md](SIZING.md) for every selected bank and site.

The larger term’s native oracle (`e4030204`,320layer TASK) recovers only **3.174960 [2.435293,3.914626]us** from ready-partial S3→S1 at v16; v32 is already S1 and gives **−0.051371 [−0.660838,0.558096]us**. Six-load split-only **R1.002444 [1.001132,1.003757]** uses the central larger-A:A-SE interval. Full producer/consumer deletions are larger lab-only bounds, not production claims. Every oracle arm/shape CI is in [ORACLE.md](ORACLE.md).

Built against the native producer term:24 tile/stage/split dispatches at all eight large sites, using S19’s unchanged `_vdsk_gemm` device body and its normal consumers/PDL. Native batch `c87c2d8a` passes **448layer +480strict projection/router TASK**. Best changed-load time ratios across the eight sites range1.001106–1.065163. All arm CIs are in [TUNING.md](TUNING.md); no positive full-layer point was selected for a new held-out G0. These losses close only these24 plans, not native scheduling or the physical gap. No weight bank is added; an isolated selected-portfolio prepare is unmeasured because there is no selected portfolio.

Next by size: small-M, chiefly the coded pool/router producer at d16. Its completion-critical surplus dominates the13.278us d16 model. Keep the same whole-graph/PDL and symmetric per-round protocol; do not equate long PDL residence with removable work.

`large_split/9ab81a3f` does not measure S19's physical activation/tensor/epilogue split. Its observer/launch/synchronization/checksum/boundary residual cannot show the candidate terms are small. Its negative held-out rebuild verdicts remain valid.

This experiment sizes small-M producer exposure above **actual selected coded bank bytes / 4.614 TB/s + 2.876 us**, and large-M native producer excess plus the split-consumer ceiling, as six-load R. It executes frozen S19 `8b415553` in its full graph with PDL unchanged. Capture wrappers only read CUDA graph metadata. CUPTI supplies actual kernel timestamps in separate observation replays; no observer kernels or in-graph event nodes are inserted. Unprofiled A:A cohorts retain every one of 16 palindromic rounds for eight fixed fresh seeds. All rounds enter arithmetic means.

The initial large consumer upper scenario deletes the complete b1/route/b2 residence; split-only savings are unmeasured until the oracle. Producer residence sums overlap predecessors. For example O16F103.818us residence contributes only6.076us to the observed completion path, and gate32F148.398us contributes2.657us. Sizes therefore use completion contributions, not residence-minus-roof sums; these are models, not promised improvements. Per-role critical contributions and whole-wall/event controls remain in the evidence. Logical bank bytes are not physical HBM counters.

The larger term's oracle uses the same whole-layer timing boundary and remaining native kernels:

- A / AA: independent unchanged S19 graphs.
- P_READY: replace selected producers by their same-seed S19 partial buffers.
- SINGLE_READY: same producer deletion, pre-sum partials outside timing and run S19's consumer at S1. Compared with P_READY, this bounds the split reduction under one boundary.
- C_READY: remove consumers using their same-seed S19 outputs, retain native projections.
- BOTH_READY: combined deletion exposes overlap and dependency interaction.

These are explicitly synthetic lab bounds. Every fresh seed first runs real S19 to fill its own buffers, and all oracle outputs must pass the published layer TASK. No ready-buffer caching is a candidate. Removed nodes change the dependency frontier, so deletion is not an exclusive hardware wait counter. Timed graphs contain no extra kernels; cold-L2 reads and event boundaries remain outside the replay. Every per-round observation is retained by a lossless integer-column archive.

`f304369a` exceeded its packet allowance by222bytes, so it has no usable measurements. The repeat retains all projection/consumer observations and primary rounds, omitting only unrelated attention/MoE per-stage breakdowns. It does not alter kernels, seeds or round selection.

Reproduce with `build.py`, `screen.py`, `read_sizing.py RECEIPT`, `oracle_build.py`, `oracle_screen.py PAYLOAD`, `read_oracle.py RECEIPT`. Native Hopper arithmetic uses explicit stand-ins only in local screens. `base_validation.json` records the original official local_eval screen; H200 receipts are required for all native timing claims. Remote sends use only h200_lab, with `--lane P10 --decision ...`.


`24b58b2d` also exceeds the old packet allowance by156bytes and supplies no usable timing result. Bit packing was tried locally but increased the archive. The retained exact integer-column archive factors out only a common event-clock divisor, verified bit for bit. `send.py` calls the normal `tools/h200_lab.py` sender with a **process-local14k reservation**, leaving256marker+2000suffix bytes inside the16k judge tail. Actual wrapper/tail validation is recorded in `tune_validation.json`. No shared tool, candidate, rule or timing boundary changed. The repeated successful batch keeps all original arms/seeds/rounds.

All primary means retain every round and seed, without trimming, retries selected by speed, or A:A mean subtraction. Oracle and dispatch-selection CIs use `tools/promotion.interval` with the larger primary/A:A SE. Selection input/weight/router seeds are distinct from sizing and oracle seeds. No newly selected candidate or held-out confirmation is claimed.

`source_integrity.json` pins the five canonical source/receipt pairs, including both unmeasured packet failures. `sizing_results.json`, `oracle_results.json`, `tune_results.json` retain complete scoped observations and per-round timing arrays. Local screens use explicit Hopper arithmetic stand-ins; native H200 receipts supply all performance claims. `base_validation.json` additionally records13 official TASK cases,12graphs and static checks from `tools/local_eval.py` on the unchanged S19 base.

```sh
python3 problems/muse_bf16_proj/dev/native_boundary/build.py
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_bf16_proj/dev/native_boundary/screen.py
python3 problems/muse_bf16_proj/dev/native_boundary/read_sizing.py e70024b150b942fa
python3 problems/muse_bf16_proj/dev/native_boundary/oracle_build.py
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_bf16_proj/dev/native_boundary/oracle_screen.py problems/muse_bf16_proj/dev/native_boundary/generated/oracle_payload.py
python3 problems/muse_bf16_proj/dev/native_boundary/read_oracle.py e4030204ff2c4e2f
python3 problems/muse_bf16_proj/dev/native_boundary/tune_build.py
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_bf16_proj/dev/native_boundary/tune_screen.py
python3 problems/muse_bf16_proj/dev/native_boundary/read_tune.py c87c2d8aa64b4805
python3 problems/muse_bf16_proj/dev/native_boundary/report.py
```
