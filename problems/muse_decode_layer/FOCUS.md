# Decode layer focus — G3 first

Serves G3/G2: improve the whole-layer score; serving uses the verify shapes.
H200 control: `cand_r3_v9.py`, receipt `0e94268cf5654012`, **1.428x**.
Current owner-requested candidate: [cand_r3_best.py](cand_r3_best.py), SHA-256
`812f5f369d042c9bb7f80c4e7198bc4883922539c619c10f1e49622f1fe7aad2`.
This build was not sent; Jihoon screens/sends it. Earlier hashes with the same name are historical.

Local normal and forced-Hopper screens: **13/13 tests, 12/12 fresh graphs, 11/11 static checks**.
Cold prepare with all native attention compiles and tuners: **175.121 s** locally;
forced wrapper including extra branch checks: **177.152 s**. H200 cold prepare remains unmeasured.
Details, commands, source hash and per-shape selections: [BEST](dev/BEST.md).

The candidate keeps MoE on V9 at every T and small pool/router on the measured V9 GEMV tiles.
It repairs Lt/XQA names and compilation, restores q1 attn_main, tunes the cost/projection portfolio
plus seven H200-positive Lt recipes, and uses a post-store FP32 attention gate.
PREFIX is token-tiled (TB≤32): sm_90 O/stats 24,576 B, pool/router 42,496 B at T128/256,
below 232,448 B. The local tuner retains regular PREFIX; forced fused graphs also pass.
NVFP4 and E4M3 bytes stay packed; scale conversion occurs inside V9's timed kernel.
V7/grouped/Humming/Marlin, rejected persistent islands and dead dispatchers are absent.

Attention sweep `59e53e746df341a9` had no receipt at this turn's one read.
Its fixed H200 winners are pending; local stand-in choices establish glue, not H200 speed.
Native Hopper attention agreement and this build's speedup need the owner's H200 run.
Other lanes' MoE/projection handoffs and historical receipts are in [NEXT_CHECKS](dev/NEXT_CHECKS.md).
