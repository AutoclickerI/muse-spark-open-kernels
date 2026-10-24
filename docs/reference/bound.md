# Bound
Serves: G1, G2. Read when ranking decode-layer work: the rules-compliant kernel lower bounds (`python -m sim.bound`) and the gap from our receipts.


Snapshot: 2026-10-23. Reproduce every modeled number with:

```sh
python -m sim.bound
python -m sim.bound --json
```

## Result

At H200 specification rates, the maximum kernel-board speedups are **2.021x projection,
6.291x attention, 3.288x MoE, and 3.344x decode layer**. Replacing 4.8 TB/s
with our best measured 4.25 TB/s stream rate makes them **1.819x, 5.570x, 2.912x, and 2.960x**. These are
replay-adjusted score ceilings; the raw time tables also charge the replay that the kernel judge
subtracts. [MODEL: `sim/bound.py`; SGLang calibration: `runs/*/*.json`; official aggregation and
loads: `problems/*/TASK.md` and `upstream/api/problems/*.json`]

## What “lower bound” means

The owner's methods—arbitrary ordering, fusion, scheduling, prebuilt libraries, and JIT—are means,
not requirements. The firm outcome is the fastest result that still performs all rule-required
work at legal precision. The model therefore assumes perfect tiling, perfect on-chip reuse, lossless
weight repacking and perfect overlap. It does **not** assume approximate
activations, fewer experts, key skipping or cached results across calls. Those are forbidden by
`upstream/llms.txt` and the official `TASK.md` files.

For one graph call, the replay-adjusted resource floor is

```text
max(
  compulsory_bytes / HBM_rate,
  BF16_FLOPs / 989e12 + legal_QK_FP8_FLOPs / 1979e12,
  nonmatrix_scalar_ops / 1979e12,
  NVFP4_weights * issue_instructions_per_weight / (132 * 1.980e9 * 64),
  communication_floor
)
```

The scalar term gives every norm, gate, softmax, routing and activation operation the entire GPU's
dense-FP8 rate. That is deliberately more generous than any CUDA/SFU implementation and therefore
a valid lower floor; it never binds. BF16 and FP8 matrix times add because they use the same tensor
cores and QK precedes PV. FP8 is used only for the legal FP8 query × FP8 key product; PV and every
other matrix operation remain at BF16 peak. The FP32 router is losslessly represented as BF16 hi+lo,
so it performs two BF16 products and retains four bytes/weight. [Precision rule: `upstream/llms.txt`;
router guarantee: `problems/muse_bf16_proj/TASK.md`]

Raw time then adds 4.224 us for each replay. That is the evaluator's measured
`replay_overhead.overhead_ns` in
`runs/muse_bf16_proj/20261023T065936Z_check_7fb3e93056d34999.json`; the custom projection
diagnostics independently report 4.93–5.31 us as “replay base” in
`problems/muse_bf16_proj/profiles/cand_sweep1.03113fd349d347dc.log`,
`cand_sweep2.62bd26b45aeb4992.log`, and the diag logs. The kernel score subtracts this overhead, so
speedup uses the adjusted time. Raw tables retain the replay cost.

Hardware rates are the H200's dense spec rates: 4.8 TB/s HBM, 989 TFLOP/s dense BF16, and
1,979 TFLOP/s dense FP8. NVIDIA's H200 page lists 4.8 TB/s and lists twice the tensor values in its
sparse footnoted columns; halving those yields the dense rates used by the competition. It also
lists 900 GB/s NVLink per GPU. [NVIDIA H200 specifications](https://www.nvidia.com/en-us/data-center/h200/)
Our achievable HBM rate is the best 4.25 TB/s 52–61 MB streaming result in
`problems/muse_bf16_proj/profiles/cand_sweep2.62bd26b45aeb4992.log`; the same probe measured
4.04–4.25 TB/s HBM and 8.2–9.6 TB/s for 4–16 MB L2 windows, summarized in `docs/FINDINGS.md`.

### Compulsory bytes and FLOPs

All values below are totals for one complete scored load, before dividing by a rate. “NV weights”
counts logical E2M1 weights that must be decoded, not bytes. The byte column includes:

- BF16 dense matrices and FP32 router globals;
- for each touched NVFP4 expert, 0.5 bytes/weight of E2M1 codes, one E4M3 byte per 16 weights,
  and two FP32 global scales;
- input/output activations at their required dtype, returned expert IDs, and cache writes;
- every old FP8 K/V row attended, once under ideal query-block reuse, plus the page-size-one int32
  mapping used to find it.

Intermediate layer activations are allowed to stay on chip in the lower bound. Counting them as
HBM traffic would contradict the allowed arbitrary fusion/order. Large real implementations will
spill some of them or reread weights; that belongs in the practical gap. Expected expert counts
come from `sim/routing.json` for standalone MoE and from `sim.problems.layer_routing`, which backs
the layer expectations out of the official floors. [MODEL: `sim/bound.py`; format/shapes:
problem `reference.py` files]

| Problem | Load | Compulsory GB | BF16-equivalent TFLOP | legal FP8-QK TFLOP | NV weights, billion |
|---|---|---:|---:|---:|---:|
| projection | decode 1 | 8.587 | 0.009 | 0 | 0 |
| projection | decode 16 | 12.327 | 0.196 | 0 | 0 |
| projection | decode 32 | 12.432 | 0.391 | 0 | 0 |
| projection | verify 1 | 8.627 | 0.069 | 0 | 0 |
| projection | verify 16 | 13.063 | 1.564 | 0 | 0 |
| projection | verify 32 | 13.904 | 3.129 | 0 | 0 |
| attention | decode 1 | 0.358 | 0.003 | 0.003 | 0 |
| attention | decode 16 | 5.723 | 0.045 | 0.045 | 0 |
| attention | decode 32 | 11.447 | 0.090 | 0.090 | 0 |
| attention | verify 1 | 0.361 | 0.023 | 0.023 | 0 |
| attention | verify 16 | 5.771 | 0.360 | 0.360 | 0 |
| attention | verify 32 | 11.543 | 0.721 | 0.721 | 0 |
| MoE | decode 1 | 4.495 | 0.016 | 0 | 7.988 |
| MoE | decode 16 | 20.299 | 0.160 | 0 | 36.058 |
| MoE | decode 32 | 27.092 | 0.303 | 0 | 48.107 |
| MoE | verify 1 | 14.371 | 0.082 | 0 | 25.533 |
| MoE | verify 16 | 38.645 | 1.178 | 0 | 68.476 |
| MoE | verify 32 | 41.684 | 2.337 | 0 | 73.655 |
| decode layer | decode 1 | 13.334 | 0.028 | 0.003 | 7.792 |
| decode layer | decode 16 | 31.335 | 0.405 | 0.045 | 23.630 |
| decode layer | decode 32 | 39.603 | 0.794 | 0.090 | 27.992 |
| decode layer | verify 1 | 22.159 | 0.176 | 0.023 | 23.403 |
| decode layer | verify 16 | 51.043 | 3.142 | 0.360 | 57.449 |
| decode layer | verify 32 | 62.662 | 6.264 | 0.721 | 66.551 |

### Per-load lower-bound time

Times are raw milliseconds and include 4.224 us per graph call. Per-load speedups use the mean
paired SGLang time in our H200 receipts. Geomean headlines instead use the organizers' published
`sglang_us`, matching `python -m sim` and `docs/reference/ceiling.md`; every named load has equal weight. All
speedups use replay-adjusted bound time, as the scorer does. [MODEL: `sim/bound.py`; measured
per-load baseline: `sim.calibrate.sglang_mean` over `runs/<problem>/*.json`; official geomean:
`upstream/api/problems/*.json`]

| Problem | Load | 4.8 TB/s raw ms | 4.25 TB/s raw ms | score ceiling at 4.8 / 4.25 TB/s | paired SGLang ms |
|---|---|---:|---:|---:|---:|
| projection | decode 1 | 3.098 | 3.330 | 2.072x / 1.834x | 3.706 |
| projection | decode 16 | 3.878 | 4.210 | 1.826x / 1.617x | 4.691 |
| projection | decode 32 | 3.900 | 4.235 | 1.851x / 1.638x | 4.793 |
| projection | verify 1 | 3.107 | 3.339 | 2.602x / 2.304x | 4.677 |
| projection | verify 16 | 4.031 | 4.383 | 1.955x / 1.731x | 5.321 |
| projection | verify 32 | 4.513 | 4.593 | 1.969x / 1.920x | 6.306 |
| **projection geomean** | | **3.719** | **3.983** | **2.021x / 1.819x** | **4.831** |
| attention | decode 1 | 0.336 | 0.346 | 19.575x / 17.332x | 1.459 |
| attention | decode 16 | 1.454 | 1.609 | 3.728x / 3.301x | 4.445 |
| attention | decode 32 | 2.647 | 2.955 | 3.184x / 2.819x | 7.594 |
| attention | verify 1 | 0.337 | 0.347 | 21.676x / 19.192x | 1.629 |
| attention | verify 16 | 1.464 | 1.620 | 3.851x / 3.410x | 4.631 |
| attention | verify 32 | 2.667 | 2.978 | 3.252x / 2.880x | 7.821 |
| **attention geomean** | | **1.093** | **1.184** | **6.291x / 5.570x** | **3.766** |
| MoE | decode 1 | 1.190 | 1.311 | 4.973x / 4.403x | 4.656 |
| MoE | decode 16 | 4.482 | 5.030 | 2.689x / 2.380x | 11.369 |
| MoE | decode 32 | 5.898 | 6.628 | 2.201x / 1.949x | 12.425 |
| MoE | verify 1 | 3.247 | 3.635 | 2.940x / 2.603x | 8.803 |
| MoE | verify 16 | 8.304 | 9.346 | 3.472x / 3.074x | 27.952 |
| MoE | verify 32 | 8.938 | 10.062 | 4.293x / 3.801x | 37.279 |
| **MoE geomean** | | **4.432** | **4.963** | **3.288x / 2.912x** | **13.448** |
| decode layer | decode 1 | 3.040 | 3.399 | 3.873x / 3.429x | 10.759 |
| decode layer | decode 16 | 6.790 | 7.635 | 2.902x / 2.570x | 18.946 |
| decode layer | decode 32 | 8.513 | 9.580 | 2.668x / 2.363x | 22.016 |
| decode layer | verify 1 | 4.878 | 5.476 | 3.568x / 3.159x | 16.469 |
| decode layer | verify 16 | 10.896 | 12.272 | 3.301x / 2.923x | 35.104 |
| decode layer | verify 32 | 13.316 | 15.006 | 3.882x / 3.438x | 50.683 |
| **decode-layer geomean** | | **7.065** | **7.941** | **3.344x / 2.960x** | **22.620** |

The raw attention floor looks much larger than its adjusted work at concurrency one because 62
replays contribute 0.262 ms while the compulsory work is only about 0.075 ms. The very high
per-load adjusted ratio is therefore a scorer artifact only in the benign sense that the scorer
explicitly removes a real fixed cost from both sides; the bound does not tamper with that removal.

### NVFP4 decode issue floor

Hopper has no E2M1 conversion instruction: PTX restricts the packed E2M1 conversion forms to
sm_100-family and later targets. [NVIDIA PTX ISA, conversion target notes](https://docs.nvidia.com/cuda/parallel-thread-execution/)
The broadest defensible minimum is one packed LUT/load-or-PRMT operation for each byte holding two
codes, or **0.5 issue instructions/weight**. It assumes a 256-entry byte→f16x2 table, perfect scale
delivery, and complete overlap. `docs/reference/profile.md`'s static Marlin/custom-kernel audit finds approximately
**1.25–1.5 integer instructions/weight**; the model uses 1.5 as a feasibility stress test.

Neither floor binds. The closest case is MoE decode-1: the 1.5-instruction floor is 0.716 ms versus
0.936 ms of HBM work, or 76.5%. The theoretical 0.5-instruction floor is one third of the audited floor.
[MODEL: `sim.bound`; issue width/clock:
`docs/reference/hardware.md` and machine record in `7fb3e93056d34999`]

## Inadmissible changes

Lower-precision activations, fewer experts, sparse/key-skipping attention and cached results remain
outside the kernel rules; see [CONSTRAINTS](../CONSTRAINTS.md). They are not candidate speedups.

## Gap to practice and the boards

Bound and practical times are replay-adjusted geomeans. SGLang is the organizers' published
`sglang_us`. “Ours” is the best complete, passing, unflagged H200 receipt currently in `runs/`,
including checks; it need not be a board submission. Leader values are a 2026-10-23 snapshot of the
public `/api/v1/leaderboard/<problem>` endpoints. [MODEL: `sim.bound`; API snapshot embedded there]

| Problem | Bound 4.8 / 4.25 TB/s, ms (speedup) | Practical model, ms (speedup) | SGLang ms | Our best H200, ms (speedup; receipt) | Board leader, ms (speedup) | Where the remaining gap sits |
|---|---:|---:|---:|---:|---:|---|
| projection | 2.390 / 2.656 (2.021x / 1.819x) | 3.211 (1.505x) | 4.831 | 4.232 (1.142x; `9294b3237c4f4e2d`) | 3.339 (1.447x) | cold BF16 matrices; high-M router hi+lo issue and skinny tails |
| attention | 0.599 / 0.676 (6.291x / 5.570x) | 1.091 (3.451x) | 3.766 | 2.938 (1.282x; `26fd635bdf114ced`) | 1.592 (2.366x) | page-1 KV gather efficiency, softmax/split finalization, short-window fixed cost |
| MoE | 4.090 / 4.619 (3.288x / 2.912x) | 4.830 (2.784x) | 13.449 | 9.016 (1.492x; `eefecf1eeb724feb`) | 5.782 (2.326x) | touched NVFP4 expert bytes, dequant pipeline, residency and global phase barriers |
| decode layer | 6.765 / 7.641 (3.344x / 2.960x) | 8.585 (2.635x) | 22.620 | 19.794 (1.143x; `2c112482e81745af`) | 12.134 (1.864x) | verify experts first; then full attention and dependent layer glue |

Receipt paths:

- `runs/muse_bf16_proj/20261023T095458Z_check_9294b3237c4f4e2d.json`
- `runs/muse_attention_decode/20261023T092042Z_submit_26fd635bdf114ced.json`
- `runs/muse_moe_w4a16/20261023T091300Z_check_eefecf1eeb724feb.json`
- `runs/muse_decode_layer/20261023T082030Z_check_2c112482e81745af.json`

The three largest gaps from our measured kernels to the spec bound are:

1. **Attention: 4.91x.** Our winning kernel specializes q_len=1 but loses every verify
   shape; the gap is mainly verify-tile reuse, page-1 gathers and fixed/split work, not missing keys.
2. **Decode layer: 2.93x.** The isolated MoE win has not reached the whole-layer verify path;
   `docs/reference/profile.md` attributes 68.9% of our verify-32 profile to experts.
3. **MoE: 2.20x.** HBM is still the proof bottleneck, but the audited decode issue floor is already
   76.5% of HBM at the smallest shape, leaving little instruction budget for addressing and routing.

Those factors are computed as bound speedup divided by our receipt speedup. [MODEL + receipts in the
gap table] Against the board leaders, the factors are attention 2.66x and decode layer 1.79x.

## What was checked and what remains inferred

Checked:

- `python -m sim.bound` and `python -m sim.bound --json` run with stdlib plus the existing `sim`
  package and reproduce the tables.
- All 68 official scored shapes are consumed; SGLang baselines come from paired H200 receipts.
- The kernel arithmetic includes format metadata, router hi+lo work, FP8 cache traffic/page indices
  and replay cost.
- The public rules and official task files were read.

Inferred or unverified:

- Layer routing expectations are inferred from official floors where the private task data does
  not expose `router_load`/`router_distinct`.
- Perfect fusion, overlap and peak tensor utilization are lower-bound assumptions,
  not achieved measurements of this model.
- Scalar instruction/SFU behavior is bounded with an intentionally impossible whole-GPU FP8 rate;
  it never selects the maximum, so it cannot make a headline falsely slower.

