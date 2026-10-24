# Projection next checks

Solo/GPU0. The queue is shared: never poll or wait for a judge. Check an H200 lab
receipt once per turn, and build the next artifact while it runs.

| Order | Artifact | Status |
|---|---|---|
| 1 | `dev/dflash_tp8_lab.py` | Sent priority 0.8; SHA256 `deeb1d446fdc7262a74511f19c4cc9a70da1349586ea5dbcb5217688b3792699`; receipt pending |
| 2 | `serving/sglang/python/sglang/srt/layers/muse_dflash.py` | TP8 portfolio extension and concat-FC helper; 3 graph tests pass on sm_120 |
| 3 | `serving/tests/test_muse_dflash_portfolio.py` | 3/3 exact eager and CUDA-graph agreement tests pass on sm_120 |

The replaced `cand_dflash_diag.py` waiter was killed (PID 3649289); do not resend
it. The proj host is still scored best `f3786341611f4f1f`.

Closed: stock CUTLASS recipe set `45f58e7009c440eb` (all recipes lose at M128/256);
glue fusion `479aab3038c6407c` (all stages/chains lose, picks=0). These closures
do not close the broader kernel families.
