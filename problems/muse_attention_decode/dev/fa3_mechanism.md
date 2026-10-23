# FA3 versus XQA verify mechanism

The H200 check `32c5b4b287c24c38` reports FA3/custom times of 206.56/675.30 us
at full verify B16 and 380.22/1,182.91 us at full verify B32. The K128 check
`a0511c0001464dbe` reports 207.36/616.74 us and 391.36/1,173.81 us. Full
B16/B32 read 327.68/655.36 MB of raw FP8 KV. Dividing those bytes by the best
custom times gives only 0.53/0.56 TB/s, while the independent random page-1
gather probe reaches 2.558 TB/s. The gap is therefore the fused attention
schedule and tensor work, not a raw HBM ceiling.

The installed image's FA3 binary is
`/home/jihoon/anaconda3/envs/comp_kv/lib/python3.12/site-packages/sgl_kernel/flash_ops.abi3.so`.
`cuobjdump --dump-resource-usage` shows SM90 FP8 `FlashAttnFwdSm90` families
with 64/96/128/192/256 tile variants and static/varlen persistent schedulers.
`cuobjdump --dump-sass --gpu-architecture sm_90a` shows `LDGSTS.E.*` tensor-memory
loads and `HGMMA.64x128x16.F32.BF16` / `HGMMA.64x176x16.F32.BF16` PV tiles.
Our XQA source uses page-row `cp.async` and FP16 V widening/PV; the new portfolio
also measures an FP8 raw-P/V path.

The standalone FA3 tuner check `dd927c6c46264afe` failed before timing because
its strict output-equivalence gate rejected `(num_splits=1, pack_gqa=None)` at
relative error 0.008635. The portfolio treats such settings as rejected while
continuing the complete accepted grid.

The portfolio's `prepare()` emits:

- `PORTFOLIO_MECH`: raw KV bytes, 64 logical query rows per KV page, FA3 and
  best-custom times, effective bandwidth, and the 2.558 TB/s gather reference.
- `PORTFOLIO_MECH_SPLITS`: every accepted FA3 split/pack time, testing split
  count and SM fill directly.
- `PORTFOLIO_MECH_BINARY` and `PORTFOLIO_MECH_TESTS`: the TMA/`cp.async`, PV,
  reuse, and split measurements tied to each shape's graph timings.

The raw-FP8 P/V stand-in agrees with the existing group path at relative L2
0.00231 for B1 sliding and 0.00450 for B16 full. The full local screen passes
16/16 correctness cases and 12/12 graph shapes on sm_120.

The new `cand_attn_fa3_struct.py` adds an eight-stage ring to that portfolio.
Its forced sm_120 path is finite and agrees with the five-stage group path at
L2 0.0 for B1 sliding and B16 full; the full candidate screen remains 16/16 and
12/12. H200 timing is intentionally left to the orchestrator.
