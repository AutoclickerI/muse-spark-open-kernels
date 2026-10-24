# Projection next checks

Solo/GPU0. The queue is shared: never poll or wait for a judge. Check an H200 lab
receipt once per turn, and build the next artifact while it runs.

| Order | Artifact | Status |
|---|---|---|
| 1 | `dev/dflash_tp8_lab.py` | Send via `tools/h200_lab.py` at priority 0.8; receipt pending |
| 2 | `serving/sglang/python/sglang/srt/layers/muse_dflash.py` | TP8 portfolio extension and concat-FC helper; local tests pending in this turn |
| 3 | `serving/tests/test_muse_dflash_portfolio.py` | Exact eager and CUDA-graph agreement tests |

The replaced `cand_dflash_diag.py` waiter was killed (PID 3649289); do not resend
it. The proj host is still scored best `f3786341611f4f1f`.

Closed: stock CUTLASS recipe set `45f58e7009c440eb` (all recipes lose at M128/256);
glue fusion `479aab3038c6407c` (all stages/chains lose, picks=0). These closures
do not close the broader kernel families.
