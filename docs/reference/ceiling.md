Serves: G1, G3. Read when interpreting the performance model's headroom or serving score.

# Modeled ceilings

The executable model is `python -m sim` and `python -m sim.serving`; its measured inputs and current H200 results are owned by [FINDINGS](../FINDINGS.md), while the full byte/FLOP lower bound is [bound](bound.md). Hardware rates and their sources are [hardware](hardware.md). This file records the model's meaning and assumptions, not a dated leaderboard snapshot.

## Kernel bounds

For each problem load, `sim/problems.py` forms a weighted sum of shape times. A problem speedup is the geometric mean of baseline load time divided by candidate load time, so each load has equal influence. The three model columns are:

- **floor:** the organizer's bound using 75% of expected expert work and the shortest agentic context;
- **expected:** all experts touched by the routing model and mean context;
- **practical:** expected work at a reduced memory/tensor rate plus launch costs.

The floor is the judge's reference, not a promise of achievable throughput. The practical model prices fixed work for the selected kernel count (multi-kernel or one fused kernel), and therefore exposes the value of fusion and tail removal. It does not model register spills, wave quantization, scattered page-1 gathers, dequant issue limits, clock changes, or correctness failures. The exact rates and fixed-cost constants are in `sim/hw.py`, `sim/problems.py`, and `sim/bound.py`; do not duplicate them here.

Layer routing is inferred from the official floor where the organizer does not publish it. That inference is a model input, not a measured routing distribution. Current leader distance, receipt-backed per-load timing, and closed recipes are in [FINDINGS](../FINDINGS.md) and [SOTA](../SOTA.md), never copied into this ceiling note.

## Serving model

`sim/serving.py` models one 8xH200 TP8/EP8 step with DFlash target verification and a conversation workload shaped from the LMSYS OpenHands description. It combines decode-layer and prefill timings, then reports throughput at stock interactivity for concurrency 1/4/16/32. The serving score is the average ratio of our throughput to stock throughput at those four stock-interactivity points. The judge's hidden conversations, scheduling, cache sharing, host time, percentiles, TTFT limits, and held-out agreement are outside this model; a modeled ceiling is therefore a ranking aid, not a serving claim.

Decode and prefill contributions compound only inside this workload model. A model option such as pooler sharding, a fused layer, or a drafter change is useful only if its complete eight-rank cost and agreement survive the serving check. The current modeled outputs are regenerated from the simulator; do not preserve old tables in this file.
