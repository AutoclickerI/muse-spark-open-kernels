Read when interpreting a resource floor, comparing stages with it, or using the score model.
Serves: G1, G2.

# Bound

## The floor

The judge's TASK floor counts supplied bytes at 4.8 TB/s or math at 989 TFLOP/s.
`sim/bound.py` counts all required work under perfect tiling, reuse, fusion and overlap:

```text
max(compulsory_bytes / 4.8 TB/s,
    BF16_FLOPs / 989e12 + legal_FP8_QK_FLOPs / 1979e12,
    NVFP4_weights * integer_instructions_per_weight / (132 SMs * 1.98 GHz * 64),
    nonmatrix_ops / 1979e12)
```

QK precedes PV on the same tensor cores, so their compute times add. The nonmatrix term is
deliberately generous and never binds. Bytes count each BF16 dense bank once, the FP32 router as
BF16 hi+lo (four bytes and two products), touched NVFP4 codes plus group-16 scale bytes and
FP32 globals, required activation IO, every FP8 KV row plus its page-1 index. Ideal fusion
keeps intermediates on chip. Expected expert counts inferred from official floors are a model,
not measured routes. Lossless weight coding changes the byte floor itself.

`python -m sim.bound --json` exposes counts; `python -m sim` distinguishes official floor
(75% expected expert work, shortest context), expected work (routing model, mean context),
and practical assumptions (reduced rates plus launch costs). The practical model omits spills,
wave quantization, gathers, decode issue and clock variation; its ceiling is not a hardware limit.

### NVFP4 decode issue

Hopper lacks E2M1 conversion. At 64 integer instructions/SM/clock, issue time per weight is
0.51 × instructions/weight of HBM time at 4.8 TB/s. A 256-entry byte LUT's theoretical
0.5 instructions/weight is 26%; the model stresses 1.5 (77%). These count a subset of the
loop: addressing, scale handling, copies and tensor issue also occupy instructions. Static
counts alone do not prove achieved issue saturation. Hardware details: [hardware](hardware.md).

## Practical ceilings measured on the judge

[FINDINGS: The judge's H200](../FINDINGS.md#the-judges-h200) owns stream/gather/L2/MMA
measurements and protocols. Stage sections own their measured roofs. A warm-cache bandwidth
number cannot replace cold graph timing, and a stream intercept is not a removable penalty
at every boundary.

## Stage floors

`sim/bound.py` owns work formulas. `problems/muse_decode_layer/dev/levers/current.json`
records its source identity; inspect that identity before applying its stage attribution to
another candidate. Profile residence includes overlapping PDL waits and omitted rows; it is
not exclusive critical-path time.

## Score vs floor

[FINDINGS: Stage accounting and the lever model](../FINDINGS.md#stage-accounting-and-the-lever-model)
owns measured anchors and the score transformation. Stage caps are independent counterfactuals:
they cannot be added. Capping near-zero glue/sliding floors assumes fusion removes fixed work;
it does not predict that an isolated kernel can attain the cap.
