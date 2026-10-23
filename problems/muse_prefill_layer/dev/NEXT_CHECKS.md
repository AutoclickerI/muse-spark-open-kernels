# Reset-time prefill checks

**Reset: 2026-10-24 09:00 KST (2026-10-24 00:00 UTC).** This is the frozen local-work queue; no new check from this list was launched today. The prior round's `finish_round.py` still owns its already reserved v18 slot.

At freeze, today's local accounting is 10 completed check receipts (including 2 calibrations), 5 accepted pending checks, v14-v17 waiting admission, and 1 reserved integration = 20 used/reserved. This is local accounting, not a fresh server quota claim. Recount receipts/pending/reservations after reset before launch; use the wrapper's live admission/quota checks.

Every source below passed local screening and graph replay. H200 execution/performance of these new sources remains unverified. Frozen sources are self-contained; builders are not needed at the judge. The first two candidates tune only synthetic prepare-time inputs (R-20261023-7), validate cross-reader agreement, and retain only shape-based configuration choices.

## Launch order

1. **`cand_prefill_autotune.py`** — Complete single-bank candidate: synthetic prepare-time expert and attention selection; prints both stage comparisons.
   - SHA256: `71593f9c5e5467f616ea26d3f8dc7778e3f277106216b62df4dc193797680d9a`
   - Local verification: 5 official tests; forced native tuners; SM90 constructors/FA3 stand-ins; every attention choice fresh full-layer replay.

2. **`cand_canonical_autotune.py`** — Isolate the canonical expert paths with the checked v3 attention/dense stages.
   - SHA256: `9b21cf17386d46de2da7f606c2c024bb26461935dd4bf37a0ec253b0b0e10a14`
   - Local verification: 5 official tests; synthetic flat/hot/four-route tuning; both readers fresh full/sliding graph; SM90 prepare targets.

3. **`cand_attention_fp8_contig.py`** — Isolate gathered FP8 attention, BN128 without SM90 stack spills; v3 experts.
   - SHA256: `e4a9792806182e36765e2ebd9289aac2b12938e26bf6ddc194eaeea68e828e8b`
   - Local verification: 5 official tests; forced-all attention tests/graphs; full 16K fresh attention; SM90 compile and FA3 stand-in.

4. **`cand_attention_fp8.py`** — Isolate direct paged FP8 attention, BN128; compare with gather and FA3.
   - SHA256: `f379b940f6a92caa59367d9fe6460060cce2d91efea2d1b0ffe20a7a9b897e0a`
   - Local verification: 5 official tests; forced-all attention tests/graphs; full 16K fresh attention; SM90 compile and FA3 stand-in.

5. **`cand_grouped_attention.py`** — Fixed local-best serving combination: canonical RS128 adaptive + gathered FP8 BN128.
   - SHA256: `24252850d0c3db5b27c979e7f9234aa62d0bd861ce36ced1cdb32f2ba5faaef8`
   - Local verification: 5 official tests; fresh actual production branch at L2048; forced Hopper glue.

6. **`cand_canonical_grouped.py`** — Fixed canonical RS128 expert reader, reusing prefill router metadata; v3 attention.
   - SHA256: `17206b59f6548cbd999da5cc786aa6ca96066f1c017db67c6730e7eb59b7c07f`
   - Local verification: 5 official tests; fresh nearby replay; separate gate/up scales; forced Hopper glue; SM90a no spills.

7. **`cand_canonical_prefill.py`** — Fixed v3-derived BN128 pipeline reading V6 directly; v3 attention.
   - SHA256: `c4d47509fe7a2a4b172e85167cbf8e80471d73a18f322a58561ebe3ac4cea1f4`
   - Local verification: 5 official tests final source; nearby fresh replay; separate scales/raw FP32 reference at 753/3012/16384; forced Hopper glue; SM90a no spills.

8. **`cand_attention_fp8_contig256.py`** — FP8 gathered BN256 diagnostic: locally faster but SM90 stack 424 B/thread.
   - SHA256: `4b13c714512e3113b35e3ca084669f9c5e7af4e0c726dcde41f8ecc301e35d7a`
   - Local verification: 5 official tests; forced-all BN256 attention tests/graphs; SM90 compile; Hopper glue.

9. **`cand_canonical_attention.py`** — Fixed canonical pipeline + gathered FP8 BN128; isolate the other complete one-bank combination.
   - SHA256: `463b8f678b05dbbcb34ff0169b74e168e35b416c4c91b65cf18549249cf745d5`
   - Local verification: 5 official tests; fresh actual production branch at L2048; forced Hopper glue.

## Execution

Run from the repository root. Verify the frozen sources before sending:

```sh
sha256sum -c problems/muse_prefill_layer/dev/NEXT_CHECKS.sha256
```

Launch each through the receipt wrapper, detached, with a note describing the selected row. For the first row:

```sh
nohup python3 tools/okc_run.py check problems/muse_prefill_layer/cand_prefill_autotune.py --note 'Single canonical V6 bank; synthetic prepare-time expert reader and long-context FA3/FP8 attention tuning. Local official tests, fresh graph replay, forced Hopper glue, and cross-path agreement pass.' > problems/muse_prefill_layer/dev/check_prefill_autotune.log 2>&1 < /dev/null &
```

Collect orphaned jobs through `python3 tools/okc_run.py collect <ids>` in the background. Keep developing while jobs run. A check that beats the current H200 best is promoted unchanged to `submission.py` and gets a scored run within quota, following the existing policy.

The first row prints `CANONICAL_EXPERT_TUNE` (T=753/3012/16384; flat/hot/four-local-route distributions) and `PREFILL_ATTENTION_TUNE` (16K queries, 32K/64K prefixes; FA3/direct128/gather128/gather256). Those are synthetic preparation measurements on the judge, not the scored per-load speedups. The receipt's timed layer comparisons decide promotion.

The fixed rows distinguish tuning errors, asynchronous Hopper errors and stage regressions. Prefix-zero chunks and short/sliding attention retain FA3 in the complete tuner. The local expert winner is adaptive RS128; the H200 winner is still to be measured.

## Evidence

- Expert comparison and serving decision: `../PORTING.md`, `CANONICAL_COMPARISON.md`, `canonical_comparison_cold.json`, `canonical_comparison_hot_cold.json`.
- Attention accuracy/timing/resources: `attention_fp8_verify_long.json`, `attention_fp8_bench.json`, `attention_fp8_compile_tiles.json`.
- Full tuner: `local_cand_prefill_autotune_final.json`, `local_prefill_autotune_forced_native.log`, `verify_prefill_autotune.log`.
- Expert tuner: `local_canonical_autotune.json`, `verify_canonical_autotune.log`.
- Expert boundaries and raw gate/up/down scales: `canonical_edges.json`, `verify_canonical_edges.log` (M=8/128/256, zero/all-local routes).
- The last checked H200 best at freeze remains v3 `82a4e50b54294a2a`, 1.345393x; its scored rerun `bab2f3468fcb4b58` is pending. Existing background promotion may change that before reset; compare against the then-current best.
