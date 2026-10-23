# Decode layer focus — G3 first

Serves: G3, G2.

Goal: raise the overall-board score before the 2026-11-05 17:00 KST close.
Decode carries 51% of the fitted board time weight; serving uses its verify shapes.

Control: `cand_moe7` H200 check `66d5d4c9e4054fa9`, 1.354968x (CI
[1.334644, 1.375293]); exact source hash `eb659038a74c628794b585e4579901dfea16fcd23c50eddf7862b5d84b3dcc6c`.
Do not edit sent sources. The orchestrator owns checks and scored runs.

H200 profile bottleneck: verify16/32 MoE is 286/362/499/503 us (sliding/full),
49.8–76.9% of candidate time. Decode32 sliding MoE is 119 us (54.8%).
Full verify32 attention is next at 377.5 us; projections and boundaries are smaller.

Standalone H200 evidence: MoE V9 reaches 1.654x overall and beats V7 at every load; its decode/verify ratios are 2.196/1.717/1.392 and 1.813/1.590/1.356 for loads 1/16/32. The layer candidates `cand_r3_v9.py` and `cand_r3_v9_dispatch.py` replace V7 with V9 using the same lossless tile bank and layer ABI.

Local admission: both V9 candidates pass 13/13 tests and 12/12 fresh-input graphs. V9 source SHA-256 is `ba990a86f6f7f4985345cfc89a3bc56e36730d974ba47642a344c5c77d84279f`; dispatch SHA-256 is `633c96ccffa40c6673c99b808c18b28685db5aafd4541667de68e1f0e6c41b84`. The dispatch synthetic choices on sm_120 were V9 at M=1/8/16/32/64 and grouped wide tiles at M=128/256; this is not H200 evidence. Local sm_120 timings do not establish H200 speed. Force Hopper glue through a stand-in.

Structural fusion candidates reduced launch count but regressed complete local layers;
keep them diagnostic until a packed variant is neutral or better. Read [NEXT_CHECKS](dev/NEXT_CHECKS.md)
for frozen source hashes and order. No remote run or commit is part of this lane round.
