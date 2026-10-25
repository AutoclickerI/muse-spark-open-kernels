# Ramp oracle (P7 item 0)

Per weight-streaming boundary of S10's small-M chain, the L2-warm ceiling of any pre-wait prefetch, beside the
pre-wait recipes themselves. `chain.py` holds the chain pieces (S10's kernels and exts, gemv_bal, warm kernels,
paired timing). `../gate_split/build_lab.py` reuses it.

- Oracle arm: after `lab_cold_l2()`, an untimed kernel reads the named bytes into L2, then the graph is timed. Its
  gain over the base arm bounds a prefetch of those bytes issued during the predecessor at zero cost to it.
- Recipe arm: the same bytes prefetched before `griddepcontrol.wait`, behind the real predecessor.
  - O after attention: S10's `PREFETCH` (L2 line prefetch of its slice).
  - pool_router after b1: a bulk `cp.async.bulk.prefetch.L2`, one per weight row, of the first J k-blocks.
- Chains, arms and output format: the comment at the top of `chain.py`.

Build, screen, send:

    python3 problems/muse_bf16_proj/dev/ramp_oracle/build.py
    CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_bf16_proj/dev/verify_splitk/local_run.py prof/lab/ramp_oracle.py 600
    python3 tools/h200_lab.py send prof/lab/ramp_oracle.py --host muse_bf16_proj --note "..."

The local screen (`local_lab.log`, sm_120: 128 MiB L2, 188 SMs, a shared GPU) only shows that every arm runs. Its
numbers mean nothing for the H200.
