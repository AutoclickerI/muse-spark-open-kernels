# Decode layer focus — G3 first

Goal: raise the overall-board score before the 2026-11-05 17:00 KST close.
Decode carries 51% of the fitted board time weight; serving uses its verify shapes.

Control: `cand_moe7` H200 check `66d5d4c9e4054fa9`, 1.354968x (CI
[1.334644, 1.375293]); exact source hash `eb659038a74c628794b585e4579901dfea16fcd23c50eddf7862b5d84b3dcc6c`.
Do not edit sent sources. The orchestrator owns checks and scored runs.

H200 profile bottleneck: verify16/32 MoE is 286/362/499/503 us (sliding/full),
49.8–76.9% of candidate time. Decode32 sliding MoE is 119 us (54.8%).
Full verify32 attention is next at 377.5 us; projections and boundaries are smaller.

Standalone H200 evidence: Humming3 reaches 1.424x at MoE M=256; v7 wins small M.
The new `cand_r3_moe_dispatch.py` uses the canonical-bank V8 and grouped wide-tile
kernels from the MoE lane, plus v7 for small M. `prepare()` selects by synthetic shape.

Local admission: 13/13 tests and 12/12 fresh-input graphs passed. The extracted V8/grouped CUDA also compiles for SM90a (grouped zero spills; V8 small spill frames), but no H200 timing is inferred. SHA-256
`dfe0d8bbc569f0636e4e41b98952cc80baf469bdb3bc0eb634f0e071b2306adf`.
Local sm_120 timings do not establish H200 speed. Force Hopper glue through a stand-in.

Structural fusion candidates reduced launch count but regressed complete local layers;
keep them diagnostic until a packed variant is neutral or better. Read [NEXT_CHECKS](dev/NEXT_CHECKS.md)
for frozen source hashes and order. No remote run or commit is part of this lane round.
