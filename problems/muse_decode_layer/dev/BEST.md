# Decode best candidate handoff

Current integration: **[cand_r3_best7.py](../cand_r3_best7.py)**, SHA-256
`5a4bdf4138d342a71b0c715af2ca8f3b51ecea686a141510e0046295acf16112`.
[Settings schema, exact-source audit and validation](best7/README.md).
Normal + forced each pass **13 tests, 12 fresh graphs, 15 static checks**. The forced table exercises
both glue implementations, fixed FA3, small-M NT2, all expert alternatives and 24 streaming instances.
The emitted `BEST7_SETTINGS` table is the sole receipt application point; pending alternatives are not
selected from local timings. Humming M256 is selected from the H200 seed-mean/full-layer evidence.

Native control: best6 `4516cb97`, check **`1b80ffc037cc454c`**, **1.379716x**, prepare **161.009 s**.
M256 Humming 419.971 us versus NT2 511.784 us in its prepare, consistent with twelve-seed lab
`de6a56fc` (448.04 vs 545.46 us). M128 retains the best6 router-dependent choice.
Board best remains V9 `0e94268cf5654012`, **1.427904x**; its favourable routing is not a seed-mean model.

[FOCUS.md](../FOCUS.md) owns the next reset's scored/check allocation and per-load score sensitivity;
[NEXT_CHECKS.md](NEXT_CHECKS.md) owns landed/pending evidence. Jihoon sends candidates.
The overall kernel score is the only goal (R-20261025-1); serving work is parked until its round.
