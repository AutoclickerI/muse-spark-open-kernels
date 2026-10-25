# Small-M dense GEMVs: excess over a plain read (P7 item 1)

H200 `9cd95e72` (prof/lab/gemv_roof.py, standalone, cold L2): a plain read of the weight takes qkv 10.7-10.9 us,
pool_router 18.7-19.1, unpooler 17.0, unpool_slice 4.1, o 5.8. S8's Triton GEMV (V9 tile) is 16-20% over it
(qkv 12.6-12.7, pool_router 22.5-22.9); cuBLAS at t=32 is 28-43% over (qkv 15.25, pool_router 24.21). The
CUDA-core warp-per-row GEMV was worse than the Triton one everywhere.

- `lab_body.py` + `build.py` -> `prof/lab/gemv_bal.py`: balanced split-K (`_gb_kernel`: runtime k-block
  ranges, S picked so all CTAs sit in one wave and end together), BN 32/64, 4/6 stages, vs S8's choice,
  cuBLAS and the read. Local sm_120 can't rank it (every arm at the local read). H200: send af875a6de1c1dc9e
  (job 26fd09a79d5e42c2).
- `patch_gemv_bal.py <cand> <out> [--all]`: ('bal', BN, S, NS) choices in `_GEMV_CHOICES` (same (P, S) ABI);
  ROWS from the receipt. `--all` forced on every role at t = 1/8/16/32 on S8_front: 13/13 tests, 12/12
  timed shapes ok, static checks pass (`s8f_gb_all_local.log`).
- Next: `../gemv_tile` asks whether the remaining gap is DRAM locality (tile-contiguous weight).
- H200 `26fd09a7` + replicate in `a1927852`: BN 64 wins. qkv S 7: 11.6-11.9 us at t <= 16 and 12.3 at t=32, vs 12.5-13.0 / 15.3 (cuBLAS). pool_router S 6: 20.9-21.3 / 22.0, vs 22.6-23.1 / 24.7-25.6. unpooler S 2: 18.2 / 18.7-18.9, vs cuBLAS 18.9 / 19.0-19.2. o and unpool_slice: no gain. ROWS in `patch_gemv_bal.py`. Paired chain check: `../gemv_pre` (send c2b2da7551560ad2).

## Remaining issue-path gap (P7)

`issue_kernel.py` changes operand order, K-tile width, pipeline stages and the
weight load's cache policy; `build_issue.py` builds standalone and whole-layer
payloads. The standalone graph includes a sum/cast consumer, so its times
include that launch and graph overhead; compare its paired ratios, not its
absolute time directly with the single-kernel stream roof.

H200 `d3666d18197247e8` (`lab_gemv_issue.py` hash `f129d2ad`): pool/router at
two stages instead of four wins projection+sumcast by 0.97740 / 0.95894 /
0.96031 at M=1/16/32. M32 unpooler with `.cg` weights is 0.98120; qkv3 `.cg`
is 0.99243 / 0.97805 at M=1/16. Operand reversal and larger K tiles give no
systematic gain. This is screening evidence, not a promotion.

Whole-layer S11 final attribution attempt `3372cfdca76c41c0` measured zero shapes because setup consumed its deadline. The frozen payload
(`lab_gemv_issue_layer.py` hash `0ad1831b`): A:A, pool NS2, unpool `.cg`, qkv3
`.cg`, all together; eight small-M shapes, 16 seeds x6 rotating cold READ rounds.
Local payload passes all eight shapes and agreement gates. The trial candidate
`5f97f0f8` passes 13 tests, 12 graph shapes and 23 static checks in both normal
and forced modes (`issue_candidate_validation.json`).

`patch_issue.py` applies onto S11 final `acda6c62`. Its ROWS remains empty
until the whole-layer result wins. `--trial` installs the measured hypotheses
for local admission. No micro change is promoted. The lane now targets structural producer epilogues in `../producer_glue/`; new labs use timeline/selected_prepare.py rather than the interrupted, unadmitted small-M setup helper.
