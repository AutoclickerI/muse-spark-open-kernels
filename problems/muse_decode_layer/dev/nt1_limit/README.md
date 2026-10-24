# Recoded NT1: small-M binding-limit experiment

Owns decode1/16/32 and verify1 (M1/16/32/8). The scalar M1 streamer is closed:
H200 `963515d3f0c04d91` gives its best recipe57.07us versus NT1-raw44.17us,
v9m144.61us on12 real-route draws. Its PDL/grid settings barely matter.

## What is measured

The exact scale-bank comparison `58ae2e2cf8df45ef` establishes a9–12% raw-scale
conversion tax at M8/16/32/128. Recoded NT1 is the starting point. It preserves
packed NVFP4 precision; there is no prepared BF16/FP8 expert-weight bank.

The unchanged NT1 sm90a build uses124 registers, no spills, two-stage36KiB tiles
and normally264 CTAs on132 SMs. Each lane decodes256 weights per tile: sixteen
16-weight groups, with128 paired BF16 multiplies and32 MMA instructions.
`prof/lab/audit_moe_nt1.py` writes source, cubins, line-mapped SASS and counts to
`runs/screen/nt1_decode/`. The exact isolated16-weight probe has:

| Component | Instructions |
|---|---:|
| FP4 extraction |10 LOP3 +7 SHF =17|
| Scale construction |2 PRMT +2 LEA =4|
| Paired BF16 multiplication |8|
| Isolated total |29|
| Production total per16 weights, amortizing scales across32 |27|

Historical H200 CTA phase medians below are from recoded NT1 diagnostic receipt
`ac9dd461f0294e34`, at the recorded1980MHz clock. The fixture uses standalone
MoE-board routing, not the next lab's real-router draws. Phase times overlap
across CTAs; W2 includes dependency wait and norm, so these medians do not add
to cold-graph latency.

| M | W13 µs | W2 µs | W2 dependency wait µs | Grid / stages |
|---:|---:|---:|---:|---|
|1|8.91|9.08|2.91|264 /2|
|8|53.97|30.57|5.81|264 /2|
|16|73.35|36.30|2.95|264 /2|
|32|82.20|40.64|3.50|264 /2|

This identifies W13 as the larger phase at M8–32 and confirms a decode cost.
It does not establish that decode instructions explain the remaining gap to
one read. The new lab separates stage/grid effects from decode/MMA work.

## Original lab experiments

- `lop3`: force the mixed-word Boolean select to one LOP3; NVCC emits two for
  the original expression. Same bytes and math;26 rather than27 normalized
  decode/scale instructions per16 weights.125 registers, no spills.
- `scale16` (inadmissible; diagnostic evidence only): represent the same8-bit E4M3 information as a16-bit integer code
  matching its multiplier bit pattern. Packed FP4 is unchanged. This removes
  the scale shift/add, adds11.1% bank bytes, and uses128 registers without spills.
  No weight is dequantized in prepare.
- `early`: cache four activation K groups in shared memory and four in registers,
  preload the complete packed tile, release its stage, then refill while doing
  the current tile's decode/MMA. A single protected reduction buffer keeps the
  two-stage footprint near114KiB;128 registers and no spills. The extra barrier
  and shared-input loads are real costs measured by the lab.

`Ddecode` and `Dmma` are non-equivalent diagnostic lower bounds. They are never
eligible for dispatch. Removing decode also permits its shared loads to vanish.
SASS shows that `Dmma` removes both MMA and its now-unused decoder/activation
dataflow: it is the streaming/control ceiling, not an isolated MMA subtraction.
It has no HMMA or HMUL2, versus32/128 in the baseline; its register count drops
from124 to62. Neither difference is a pure arithmetic latency.

## Lab and verification

`prof/lab/moe_nt1_limit.py` runs on the MoE host and generates12 public full-model
weight/router seeds. Each fixed recipe uses identical real routes, inputs and
expert weights. It checks valid recipes against independent FP32, rotates/reverses
cold128MiB graph timing order, reports actual stage/grid choices, and emits
instrumented W13/W2/wait/norm/finish CTA cycles. Stage1/2/3 and grid132/264 controls
respect residency before the grid barrier. Summaries come last.

Local admission:432 graph checks and288 forced Hopper calls pass on3 public
weight seeds; max L2=0.242%. Signed, zero and subnormal scales, empty routes,
zero inputs and local-start96 are included. SM120 cannot fit early's two-stage
shared footprint or the standard three-stage footprint: those Python paths use
stage1/stage2 native arithmetic stand-ins. Native sm90a builds of all eight
families pass, with no spill loads/stores. H200 execution of the changed
two-stage producer remains to verify. Logs: `runs/screen/nt1_limit_*`.

```bash
python prof/lab/build_moe_nt1_limit.py
CUDA_VISIBLE_DEVICES=1 tools/img python prof/lab/check_moe_nt1_limit.py
CUDA_VISIBLE_DEVICES=1 tools/img python prof/lab/audit_moe_nt1.py
```

`dropin.py` now excludes widened-scale storage and prepares one selected bank.
It is experimental. No best7 cell is changed before a valid H200 seed-mean win.

Landed as merged MoE job **dc3fea0a20a349bf** (original send0ea5aeba5a3435bf,
merged senda1762556e26931ef). One-LOP3 wins42.74/73.75/79.20/113.61us against
R2 43.08/74.45/81.21/117.22 at M1/8/16/32. Scale16 and early refill lose;
scale16 is also inadmissible for scoring. The exact admitted LOP3 cell and
new fixed-phase work are in [nt1_branches](../nt1_branches/README.md).
The admission JSON retains the original lab's historical local evidence.
