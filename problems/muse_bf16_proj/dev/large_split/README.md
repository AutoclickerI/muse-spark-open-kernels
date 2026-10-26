# P10: large projection rebuilds; physical split unmeasured

**Correction:** the 9ab81a3f stripped-twin residual contains observer/launch/sync/checksum/boundary costs. S19’s physical activation/tensor/epilogue split is **unmeasured**. The archived raw-schedule interventions do not show these terms are small. Native one-boundary sizing continues in `../native_boundary/`.

No new G0 winner. Against frozen S19 `8b415553` (expanded `62ea5bf4`), the held-out six-load ratios are shared-weight WGMMA **0.971503 [0.970251,0.972755]**, K-major weight tiles **0.977755 [0.976731,0.978780]**, and the native-schedule layout-only patch **0.999817 [0.999058,1.000576]**. Larger R is faster. All three portfolios were judged despite their stage losses. No load or stage veto was applied.

P4 receives [P4_HANDOFF.json](P4_HANDOFF.json): the two rejected rebuilds and an **unpromoted** optional native-layout arm for an already-planned composition batch. The latter improves verify16 by R1.004128 [1.002075,1.006184], but fails the sole G0 promotion gate. The earlier exact12 sliding gate at T1 remains P4's S20 nominee; the rejected T16 coded QKV is not restored.

## Evidence

- [oracle_split.md](oracle_split.md): eight large shapes, activation/tensor/epilogue factorial allocation with microsecond CIs; all 16 oracle arms and their ratio CIs, including measured norm/route neighbours and direct S19 hot-activation controls.
- [stage_ratios.md](stage_ratios.md): both structural rebuilds at all eight sites and all 24 matched K-major/row-major comparisons. [all_stage_ratios.csv](all_stage_ratios.csv) includes every candidate arm and A:A control, with paired ratio CIs.
- [whole_ratios.md](whole_ratios.md): 16 fresh held-out seeds, all 12 scored shapes, six-load covariance, G0 and prepare receipts. The batch passed 960 published layer TASK checks and 48 strict native-pack projection/router checks.
- `joint_results.json`, `packed_results.json`, `final_whole_results.json` retain every timing sample and derived CI. Stage-selection input seeds and held-out input/weight/router seeds are disjoint. Ratios use paired log Student-t intervals; saved microseconds use arithmetic intervals. The three-axis Shapley allocation averages all factorial paths.
- [source_integrity.json](source_integrity.json) pins each canonical source and receipt.

| H200 receipt | Result |
|---|---|
| `017b445ab14f4ec6` | First oracle timed out; no timings retained. See `first_oracle/README.md`. |
| `03ad7fd4788546ce` | Repaired oracle + SS finished every native check, but the packet needed 8779 bytes for an 8744-byte declaration. No timing result inferred. |
| `9ab81a3f3b054b92` | Complete oracle + 24 SS arms/site, eight seeds and eight cold alternating rounds. |
| `6a7133f514cf4068` | Complete 42-arm/site K-major batch, eight fresh seeds, exact bank round trips and original projection/router checks. |
| `7f110a377fb843a7` | Complete held-out G0 for SS, packed SS and native-schedule layout-only against S19. |

All sends used `tools/h200_lab.py`, `--lane P10 --decision ...`. The first two jobs are preserved as failed/incomplete evidence, not recipe losses. The packet repair keeps every timing sample losslessly, measures the actual wrapper schema and only deduplicates post-check diagnostic metadata.

## Mechanism and limits

The explicit raw-register oracle removes repeated activation TMA/barriers, WGMMA, and the epilogue independently and in combination. Math-free twins retain every vector weight load through a volatile checksum; CTA synchronization prevents the producer warp lapping an mbarrier phase. Shared padding matches active CTAs. Native SASS/resource receipts show the retained loads and zero spills. Synthetic shared-X tiles reuse the first two real tiles; these bounds are never production candidates. BF16 output partials are an epilogue oracle, not an FP32-router precision change.

The operational split is **not S19's physical decomposition**. Its residual includes weight issue, launch/prologue, initial X tiles, checksums, synchronization and timing boundaries. Inner event durations include a descriptive 3.16–3.79 us offset relative to the other phase's outer-event/empty-graph-subtracted controls; different input seeds prevent treating this as an isolated calibration. Physical HBM counters and opaque Lt math-free twins are unmeasured. Within-oracle intervention differences share a timing boundary. A:A controls are retained, including stage offsets.

The resulting shared/shared WGMMA rebuild reuses TMA weight tiles across token warpgroups, writes FP32 partials directly and tests K64/K128, queue depth, split count and issuing weights before the activation dependency. It loses against S19. Making the same weights contiguous in K-tile order improves all 24 matched configurations (ratios 0.886–0.985, every upper CI < 1), but that gain belongs to the slower schedule. The final native-layout arm keeps S19's tile/stage/warp/S3/CHAIN schedule unchanged at T128 QKV and pool/router; only the weight address/layout changes. Its G0 is neutral. The private leader's aggregate 1.36125 component advantage remains unidentified; neither oracle residual nor stage gains can be divided into that unrelated aggregate.

The final combined experimental prepare took 144.599 s. Measured additions were SS 2.813 s, packed SS 2.820 s and native pack 0.316 s; the two native banks took 0.185 ms on GPU. The minimal NP-only patch drops experimental banks; its isolated prepare and its composition with newer P4 builds have not been measured.

## Reproduction

Run builders from the repository root; generated payloads and logs are ignored.

```sh
python3 problems/muse_bf16_proj/dev/large_split/combined_build.py
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_bf16_proj/dev/large_split/compile.py
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_bf16_proj/dev/large_split/ss_compile.py
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_bf16_proj/dev/large_split/combined_screen.py
python3 problems/muse_bf16_proj/dev/large_split/combined_read.py 9ab81a3f3b054b92
python3 problems/muse_bf16_proj/dev/large_split/packed_build.py
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_bf16_proj/dev/large_split/packed_compile.py
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_bf16_proj/dev/large_split/packed_screen.py
python3 problems/muse_bf16_proj/dev/large_split/packed_read.py 6a7133f514cf4068
python3 problems/muse_bf16_proj/dev/large_split/final_whole_build.py
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_bf16_proj/dev/large_split/final_whole_screen.py
python3 problems/muse_bf16_proj/dev/large_split/final_whole_read.py 7f110a377fb843a7
python3 problems/muse_bf16_proj/dev/large_split/report.py
```

Native compilation targets sm90a. Local SM120 runs preserve launch glue and use explicit arithmetic stand-ins for Hopper bodies; their timings are not H200 claims. The final native receipt supplies actual Hopper correctness. `ss_eval.py` and `np_screen.py` additionally run the official `tools/local_eval.py` tests. Minimal unpromoted arm:

```sh
python3 problems/muse_bf16_proj/dev/large_split/np_patch.py BASE.py OUTPUT.py
```

Remote reproduction uses the generated payload, e.g. `tools/h200_lab.py send PAYLOAD --host muse_bf16_proj --lab-only --import-budget 700 --lane P10 --decision 'rank large projection mechanisms'`. The held-out layer payload uses the prefill host's import-time lab allowance.
