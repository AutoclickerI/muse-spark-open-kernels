# Ceiling

How fast each problem (and the serving step) can get on one H200, against SGLang's time, and how far
the current leaders are from it. All numbers come from `python -m sim` (kernel problems) and
`python -m sim.serving` (serving); this file says what they mean and how far to trust them. Measured
inputs are in [FINDINGS](FINDINGS.md); hardware parameters and their sources in [HARDWARE](HARDWARE.md).

## Method

A problem's speedup is the geometric mean over its loads of SGLang's load time over ours (each load
is a weighted sum of shapes). SGLang's published `sglang_us` is the same geomean, so a bound's
ceiling is `sglang_us / geomean(bound's load times)`. Three bounds per shape (`sim/problems.py`):

| Bound | Bytes and math counted | Rate | Fixed cost |
|---|---|---|---|
| **floor** | the judge's own: 75% of the expected experts touched, the shortest agentic context | 4.8 TB/s or 989 TFLOP/s, whichever is longer | none |
| **expected** | what the shape really moves: all experts touched (Monte Carlo of the generator, `sim/routing.json`), mean context | same | none |
| **practical** | expected | 90% of 4.8 TB/s; 75% of BF16 peak when compute-bound | 1.5 us for the first kernel + 2.0 us per further kernel |

Practical is priced for a design: proj 1 kernel, attention 2, MoE 2, decode layer 8, prefill layer 10
("multi"), or one fused kernel ("megakernel"). It is a target a strong submission can reach, not a
proof: real kernels lose to tails, wave quantization, dequant issue limits and scattered gathers.

Checks on the model:

- Floors: recomputed for all 70 scored shapes; equal to TASK.md's floor column (0.1 us rounding), MoE
  within 0.6%. The layer problems' routing isn't published, so it is backed out of their TASK.md floors
  (`layer_routing`): 2.50, 7.50, 7.53, 9.00, 18.4, 21.4 experts touched at 1-256 tokens.
- SGLang's measured times (FINDINGS) sit at 1-76% of the floor: the bounds are not exceeded by anything
  measured.
- `bw 0.90` rests on published H200 read-heavy rates (~4.45 TB/s, 92%) and SGLang's own 4.3 TB/s copy
  note; SGLang's cuBLAS reaches 74% on the judge. The fixed costs rest on H100 graph-boundary
  measurements (~1 us) and the judge's ~4 us replay subtraction; no Hopper PDL measurement exists.

## Kernel problems (2026-10-23)

| Problem | SGLang (us) | Floor | Expected | Practical | Megakernel | Leader | Practical / leader |
|---|---|---|---|---|---|---|---|
| muse_bf16_proj | 4,831 per step | 2.02x | 2.02x | 1.50x | 1.50x | 1.447x | 1.04 |
| muse_attention_decode | 3,766 per step | 6.71x | 6.38x | 3.45x | 4.31x | 2.366x | 1.46 (1.82 fused) |
| muse_moe_w4a16 | 13,448 per step | 4.38x | 3.29x | 2.78x | 2.88x | 2.326x | 1.20 |
| muse_decode_layer | 22,620 per step | 3.87x | 3.35x | 2.64x | 2.97x | 1.864x | 1.42 (1.59 fused) |
| muse_prefill_layer | 387,404 per load | 3.18x | 2.85x | 2.10x | 2.14x | 1.695x | 1.24 |

Readings:

- **proj is nearly exhausted.** The leader is at 96% of the practical ceiling; the floor's extra 0.5x
  needs bandwidth no BF16 GEMM reaches. Worth only enough work to keep rank.
- **Attention has the most ratio headroom**, because SGLang's sliding layers are 20-34 us of fixed cost
  against 0.1-3.7 us of data. It is also the most sensitive to the one untested assumption: gathering
  page-size-1 rows (128 B of K, 128 B of V per token) at 90% of bandwidth. FlashInfer measured 60-66% at
  page size 1. Sensitivity (all bytes at a lower rate):

  | Achieved bandwidth | Attention, 2 kernels | Attention, fused | Decode layer, 8 kernels | Decode layer, fused |
  |---|---|---|---|---|
  | 90% | 3.45x | 4.31x | 2.64x | 2.97x |
  | 75% | 3.04x | 3.72x | 2.24x | 2.48x |
  | 63% | 2.68x | 3.23x | 1.92x | 2.09x |

- **MoE**: expected (all touched experts read) is 3.29x; the practical 2.78x needs NVFP4 dequant at line
  rate on CUDA cores (~2 INT ops per weight budget, HARDWARE). Marlin runs at 15-34% of floor today.
- **Decode layer** carries the largest weight on the overall board (below) and 1.4-1.6x headroom over
  the leader; fusion matters here (8 kernels to 1 is +0.33x).
- **Prefill** is compute-bound in the expert GEMM at the first-turn shapes; the gap between expected and
  practical is the 75% tensor-core efficiency assumption. It isn't on the overall board, but under the
  modeled workload it is worth as much serving score as all of decode (Serving below).

### Overall board weight (inference)

The overall board's formula is unpublished. If problems count in proportion to SGLang's time per
decode step (the only published per-problem weight that adds up to a step), the shares are: decode
layer 51%, MoE 30%, proj 11%, attention 8% (prefill isn't on the overall board). Under that reading
decode layer and MoE carry ~80% of the rank; under an equal-weight reading attention is worth as much
as either. Ask or watch the board before trading one against another.

## Serving (model)

`sim/serving.py` prices one step of the 8xH200 node (TP8 dense and attention, EP8 experts, DFlash
verifying 8 tokens per request per step) and whole conversations shaped like LMSYS's OpenHands
simulation: a 74K-token first turn, 12 more turns of ~753 new tokens, 220 output tokens per turn,
prefill served in line with decode. Stock's 62 layers and its prefills are the judge's own SGLang
times from our decode-layer and prefill checks (FINDINGS); stock's LM head, drafter and all-reduces
are modeled at cuBLAS rates. The judge's conversations are hidden (serving RULES), so the workload is
the largest assumption here.

tau 4.16 (LMSYS AgentX), 13 turns, interactivity over the whole conversation:

| Concurrency | Stock step | Practical step | Stock tok/s per GPU | Practical tok/s per GPU | Stock tok/s per user | Practical tok/s per user | Prefill share (stock / practical) |
|---|---|---|---|---|---|---|---|
| 1 | 18.45 ms | 7.80 ms | 21.1 | 48.0 | 169 | 384 | 25% / 28% |
| 4 | 26.84 ms | 9.77 ms | 40.4 | 94.7 | 81 | 190 | 48% / 55% |
| 16 | 40.26 ms | 15.80 ms | 60.0 | 128.9 | 30 | 65 | 71% / 76% |
| 32 | 56.66 ms | 19.58 ms | 65.6 | 142.2 | 16 | 36 | 78% / 83% |

The practical step at c=1 is layers 6.34 ms, drafter + LM head 0.66 ms, all-reduces 0.80 ms (2.77 ms
at c=32: 134 per step at 5.5 us plus size). The drafter's `fc` (5 x 8192 -> 8192, 671 MB) is
replicated on every GPU (`srt/models/dflash.py:705`). Prefill per conversation: stock 3.37 s first
turn + 72 ms per later turn; practical 1.73 s + 30 ms; expected 1.29 s + 22 ms. Stock's times are the
mean of SGLang's times over our judge runs of each shape (two so far; they differ by ~3%).

### Score ceilings

Score = our throughput at stock's interactivity over stock's throughput, averaged over the four rungs.
At c=16/32 stock's interactivity is below anything our curve reaches (we can't run past 32 requests),
so those rungs are either extrapolated along our curve or capped at our c=32 point.

| Design | Score (extrapolated / capped) | c=1 | c=4 | c=16 | c=32 |
|---|---|---|---|---|---|
| Practical: decode only (stock prefill) | 1.60x / 1.60x | 2.39 | 1.59 | 1.23 | 1.19 |
| Practical: prefill only (stock decode) | 1.65x / 1.59x | 1.37 | 1.59 | 1.75 | 1.87 |
| Practical, both | 3.13x / 3.04x | 4.63 | 2.99 | 2.44 | 2.46 |
| Practical + sharded pooler at every size | 3.17x / 3.09x | 4.83 | 3.03 | 2.42 | 2.42 |
| One fused kernel per layer | 3.21x / 3.12x | 4.85 | 3.06 | 2.45 | 2.46 |
| Expected (4.8 TB/s, no fixed costs) | 4.55x / 4.34x | 6.87 | 4.34 | 3.47 | 3.51 |

Readings:

- **Decode and prefill are worth about the same** under this workload (1.60x and 1.65x alone), and
  they compound (3.1x together), because the score compares at equal interactivity and both set it.
  Decode wins at c=1, prefill at c=16-32. The prefill kernel isn't on the overall kernel board but is
  half the serving prize.
- **The workload decides the split.** Single-turn conversations (`--turns 1`): practical 2.05x, almost
  all from prefill. tau 4.5 instead of 4.16: 3.06x (moves little, because prefill doesn't use it).
- **The interactivity definition barely matters**: leaving the user's own prefills out of tokens/s per
  user (LMSYS's 1/ITL style, `--interactivity itl`) gives 3.05x / 2.97x instead of 3.13x / 3.04x.
- **Sharding the pooler** (and the unpooler above 8 tokens), as stock does on GB300 only, saves
  0.46 ms per step at c=1 after paying its all-gather, and loses ~0.4 ms at c=32 if the gather of
  2-4 MB costs like an all-reduce: gate it by token count, with the threshold measured on the judge.
  [SERVING](SERVING.md) lists this and the other Hopper-gated levers.
- Stock's modeled c=1 decode rate is 225 tok/s per user (4.16 tokens per 18.45 ms step). LMSYS
  published 543 at c=1 on 8x B300 (1.7x the bandwidth, FP4 tensor cores, Blackwell-only kernels) and
  nothing for H200, so stock's model is plausible but not validated.

## Limitations

- Practical rates are from published H100/H200 measurements and SGLang's own kernels, not from our
  kernels; the first kernels we write will test them.
- Routing for the layer problems is inferred from floors, not from the generator.
- The serving model ignores host time (`--host-us`), scheduling (chunked prefill of 16K, hierarchical
  cache, cache hits beyond the turn's new tokens, prefix sharing across conversations), p50/p90 effects,
  the TTFT cap, and the extra experts a larger verify batch touches beyond the decode layer's routing.
- Stock's step outside the 62 layers (LM head, drafter, all-reduces) is modeled, not measured; serving
  can't be measured anywhere we have.
- Leaders move daily; the table above is from 2026-10-23 16:22 KST.
