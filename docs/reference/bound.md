# Bound
Serves: G1, G2. Read when ranking decode-layer work or asking how far a stage can still go: what the judge's floor
counts, the practical ceilings measured on its H200, each stage's floor, and the score with every stage at a given
multiple of its floor.

## The floor

The judge's floor (TASK.md) is the bytes as given at 4.8 TB/s or the math at 989 TFLOP/s. Our lower bound
(`sim/bound.py`) is the same idea applied to all the work the rules require, with perfect tiling, reuse, fusion
and overlap. Per graph call:

```text
max(compulsory_bytes / 4.8 TB/s,
    BF16_FLOPs / 989e12 + legal_FP8_QK_FLOPs / 1979e12,     # QK precedes PV on the same tensor cores
    NVFP4_weights * integer_instructions_per_weight / (132 SMs * 1.98 GHz * 64),
    nonmatrix_ops / 1979e12)                                # deliberately generous; never binds
```

- Bytes: BF16 dense weights once; the FP32 router as BF16 hi + lo (four bytes per weight, two products); each
  touched NVFP4 expert's E2M1 codes (0.5 B/weight), one E4M3 scale per 16 weights and its FP32 global scales;
  activations at their required dtype; every FP8 K/V row attended, once, plus its page-1 int32 index. Intermediate
  activations stay on chip (fusion is allowed, CONSTRAINTS "Boundary cases").
- Not assumed: anything CONSTRAINTS forbids (FP8/FP4 activations, fewer experts, skipped keys, cached results).
  A lossless smaller weight encoding would lower the floor; hardware compression doesn't (below).
- Expected touched experts per layer are backed out of the official floors (`sim.problems.layer_routing`), not
  measured routes.
- The judge subtracts the graph replay's fixed cost (~4.2 us per call, `7fb3e930`) from both sides.

`python -m sim.bound` prints every problem's per-load floor (`--json` adds compulsory bytes and FLOPs). At the
spec rate the score ceilings are projections 2.021x, attention 6.291x, MoE 3.288x and **decode layer 3.344x**.
Its "measured" column uses 4.25 TB/s, from an event-subtracted 52-61 MB sweep; the judge's own protocol streams at
3.61 TB/s (below), where the decode-layer ceiling is about 2.5x (the ceiling scales with the rate, as the 4.25
column shows). `python -m sim` adds three model columns: **floor** (the organizers' bound: 75% of expected expert
work, shortest context), **expected** (all experts the routing model touches, mean context) and **practical**
(expected work at reduced rates plus per-kernel launch costs; no spills, wave quantization, gathers, decode issue
or clocks). Rates and constants live in `sim/hw.py`.

### NVFP4 decode issue
Hopper has no E2M1 conversion (PTX restricts it to sm_100+), so every weight is decoded by integer or permute
instructions at 64 per SM per clock (CUDA guide throughput table, cc 9.0). Per weight, issue time is 0.51 x
(instructions per weight) of the HBM time at 4.8 TB/s. A 256-entry byte LUT gives 0.5 instructions per weight (26%,
the theoretical minimum); `sim.bound` stresses 1.5 (77%). Our production NT1 decoder runs 26-27 integer
instructions per 16 weights (FINDINGS "MoE real-routing labs", NT1 audit), i.e. 83-86% of HBM time at 4.8 TB/s and
65% at the practical 3.61 TB/s. Decode alone doesn't bind, but with addressing, scales and MMA issue the small-M
expert loop sits near the issue limit; this is our inference, consistent with `dc3fea0a` (decode costs up to
~27 us of 117 at M=32) and `963515d3` (a pure stream lost to NT1 at M=1).

## Practical ceilings measured on the judge

Protocols and full rows: FINDINGS "The judge's H200" and "P8 hardware probes".

| Ceiling | Value | What it bounds | Receipt |
|---|---|---|---|
| Plain cold HBM stream | 256 MiB in 74.33 us = **3.61 TB/s = 1.33x the floor's rate**; a 528-CTA ~130 MB stream 3.80 TB/s (raw events); 52-61 MB with event baseline subtraction 4.04-4.25 TB/s | A byte-bound stage reaches ~1.3x its floor at best | `4233660b`, `849bd4b6`, `cand_sweep2.62bd26b4` |
| Hardware memory compression | Granted, but real NVFP4 and BF16 weights read at 0.997-1.002x (zeros 2.03x) | Closed: only software coding can lower the byte floor | `4233660b` |
| Page-1 random KV gather (128 B rows) | 0.46 / 1.62 / 2.56 TB/s at 132 / 528 / 1056 CTAs | Attention's KV read; more independent rows in flight | `849bd4b6` |
| One launch reading one weight (8-75 MB) | 0.96-2.45 TB/s, ~5.2 us launch/ramp envelope | Small dense stages' per-launch cost | `feb7b287` |
| Dense stages with weights already in L2 | warm qkv 9.8-14.9 us (2.4-3.7 TB/s from L2); prefetch saves <= 2-4 us per layer | Dense GEMVs are bound by launch and issue, not HBM | `09d19813` |
| Warm L2 read, 4-32 MiB | 7.3-7.8 TB/s | | `849bd4b6` |
| Dependent kernel pair, PDL vs normal | saves 1.06 us | One stage boundary | `849bd4b6` |
| BF16 at N=8 / 32 | `mma.sync` 504 / 590, WGMMA SS 200 / 608 TFLOP/s | Small-M tensor path | `849bd4b6` |

## Stage floors

Per layer, us: one cold read at 4.8 TB/s, or tensor math where it binds (dense stages at M=256). From
`problems/muse_decode_layer/dev/levers/current.json` (`build.py`); MoE uses the inferred touched experts.
M = tokens per call (decode B = B, verify B = 8B); S / F = sliding / full layer.

| Stage | M=1 | M=8 | M=16 | M=32 | M=128 | M=256 |
|---|---:|---:|---:|---:|---:|---:|
| QKV + gate | 7.86 | 7.86 | 7.86 | 7.86 | 7.86 | 9.77 |
| O | 3.50 | 3.50 | 3.50 | 3.50 | 3.50 | 4.34 |
| Pool + router | 15.73 | 15.73 | 15.73 | 15.73 | 15.73 | 19.54 |
| Unpooler (T <= 8: the 1024-row slice) | 1.75 | 1.75 | 13.98 | 13.98 | 13.98 | 17.37 |
| Attention S / F | 0.11 / 4.33 | 0.11 / 4.33 | 1.77 / 69.3 | 3.55 / 138.7 | 1.78 / 69.3 | 3.56 / 138.7 |
| MoE | 14.7 | 44.2 | 43.9-44.9 | 52.7-53.4 | 108.4-109.1 | 125.7-126.1 |
| Boundaries, route | <= 0.02 | <= 0.02 | <= 0.02 | <= 0.02 | <= 0.02 | <= 0.02 |

Where S4 (`ad858b58`, with the pending S5-S6 lab wins) sits, from `scenarios.py`: dense projections 1.4-2.1x their
floors; MoE 1.3-3.0x; full attention 1.4-1.5x at decode 16/32 and 2.8-2.9x at verify 16/32; sliding attention and
the batch-1 full layers are fixed work (4-184x a sub-5 us floor); glue 6.9-19.7 us of boundaries plus 3.0-6.8 us of
route per layer against ~0.02 us.

## Score vs floor

`python3 problems/muse_decode_layer/dev/levers/scenarios.py` reads `current.json` (rebuilt by `build.py` from the
latest landed S-candidate profile), adds the paired-lab stage wins that candidate doesn't contain yet, and rescores
the six loads. On S4 (`ad858b5875a940a6`, measured 1.528x) the pending S5-S6 wins (`770809cf`, `8483c0c6`,
`c42e9fcc`) give **1.565**. Calibration: the reset model (`c0e74bed` plus lab deltas) predicted 1.457 for S1's
content and 1.507 for S4's; they measured 1.467 and 1.528. S2's content modeled 1.491 on S1 and measured 1.474
(`4e212ec2`); S3 measured 1.466 (`5c7323ea`). The model tracks within ~1.5%, the size of one run's router-seed noise.

| Every stage capped at | Score on S4 | on S1 |
|---|---:|---:|
| 2.0x its floor | 1.956 | 1.942 |
| 1.5x | 2.280 | 2.341 |
| 1.3x (byte-bound stages at the measured 3.61 TB/s stream) | 2.597 | 2.710 |

The two bases differ because a capped layer keeps its profile's residual (adjusted graph minus kernel sum: kernel
durations include PDL waits) and its run's baseline: read these as +-3%. The board leader (2.285x on 10-25, SOTA)
corresponds to every stage at ~1.5x its floor at once. 1.3x is not reachable everywhere: glue and small attention
floors are near zero, so a cap there assumes their fixed work is fused away.

| Lever on S4, capped at 1.5x / 1.3x its floor | Score gain on 1.565 |
|---|---|
| Attention (sliding +5.5 / +5.6%; full verify 16/32 +2.5 / +2.9%) | +9.8 / +11.0% |
| Glue (boundaries, route) | +7.1 / +7.1% |
| Small-M MoE, M <= 32 | +4.6 / +7.3% |
| MoE M=256 | +5.5 / +6.6% |
| MoE M=128 | +4.8 / +6.1% |
| Projections | +2.3 / +6.0% |

Each row is an independent counterfactual; they don't add. Matching the leaders component by component
(attention /1.8, MoE /1.2, projections /1.25 from the overall top 3's component boards on 10-24, MoE discounted for
our newer in-layer kernels; glue /3 assumed; `--match` to change) gives 2.117, below the leader's 2.285x: the leaders'
in-layer stages beat what their component boards imply, or their glue is faster still.

## Whole-layer excess and the megakernel

On the reset profile `c0e74bed`, each batch-1 layer's kernels take 46.7 / 56.1 us (decode 1 sliding / full) and
63.3 / 78.6 us (verify 1) more than a one-read stream at 3.6 TB/s
([megakernel scoping](../../problems/muse_decode_layer/dev/megakernel/README.md)). That excess bounds every fix
together (compute, addressing, occupancy, tails, launches), not a megakernel's share: the scoped fusion estimate is
+1.74% on the score. `scenarios.py` prints the same quantity for the current model (S4 with pending wins:
44 / 48 us decode 1 and 46 / 57 us verify 1 above the stream, by adjusted graph time).
