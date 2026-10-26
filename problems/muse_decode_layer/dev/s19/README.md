# S19 integration

Base: frozen **S18 95df42fe** (`cand_r3_best7_s18_final.py`, [../s18/frozen.json](../s18/frozen.json)), scored
`28b5c189` 1.848375x, paired `5d0282bb` R 1.004321 [1.002314, 1.006332] over S17.

Candidate **S19 62ea5bf4** (expanded; shipped `cand_r3_best7_s19_final.py`, sha `8b415553`; [frozen.json](frozen.json)),
composed by `compose.py` on S18's own composer (`../s18/compose.py`).

- Round 2 (this candidate), lab `6ac9317e` on fresh seeds: **R 1.014463 [1.013044, 1.015885]** over S18, 16 seeds,
  672 TASK + 260 nearby checks, all pass; loads d1 0.9973 [0.9914, 1.0032], d16 1.0116, d32 1.0185, v1 1.0400,
  v16 1.0017, v32 1.0183. Host prepare **147.7 s** (round 1: 254.7 s). Arms against the A2 baseline of each load:
  P10 at T=16 costs d16 0.8% (1.0096 vs 1.0018), pdl_M's `_boundary_late` costs v32 0.4% (1.0046 vs 1.0005), FAST
  is neutral (d1 0.9913 vs 0.9923; round 1 +0.09%). The scored decode run waits for the 10-27 09:01 KST reset;
  promotion completes when it passes TASK.
- Round 1 (`4ae2f6b9`, lab `7c0fd3cc`) passed the gate at R 1.007437 [1.006506, 1.008368] over S18 and chose the
  rows below; round 2 dropped W, kept P10 at T=8 only, reverted pdl_M's `_boundary_late` and baked prepare.

| Piece | Rows | Evidence (7c0fd3cc leave-out R, arm/S19; >1 keeps it) |
|---|---|---|
| P10 exact 12-bit QKV, (T,N) (8,2304) | v1 | no_P10 1.00028 [0.99959, 1.00096]: v1 1.0157 for, d16 0.9881 against, so T=16 is out |
| RAW2 two-WG sliding attention | v32S | v32S_noRAW2 1.00154 [1.00069, 1.00240]; PIPE2 0.99971 [0.99901, 1.00041] |
| R8 page-coordinate lookahead (q1 port kernel) | d16F, d32F | no_R8 1.00496 [1.00432, 1.00560] |
| D tensor-core softmax denominator (VQ8) | v16F, v32F | no_D 1.00109 [1.00046, 1.00172] |
| FAST merged into S18's SHARD2+INDEX M1 cell | d1 | no_FAST 0.99992 [0.99924, 1.00060], d1 point 1.0007: kept, re-tested |
| LATE: pdl_M's `_boundary_late` edit reverted | v32S, v32F (T=256 glue) | v32S_no_pdl_M 0.99903 [0.99835, 0.99970]: S18 without pdl_M is faster at v32S |

Dropped on `7c0fd3cc`: P7 W (no_W 0.99731 [0.99658, 0.99804]: faster without it at d1 and v1; W had won on S17),
P3a SPLIT2 at M32 (1.00291 [1.00223, 1.00358]: S19's unsplit M32 is faster). Not composed earlier: P5b d16F K-tile
reuse (R8 replaces that kernel), the gate-audit shape nominees (no patch on the current base), P7 E/P/B/L/X (all lost).

LATE: pdl_M hoisted `_boundary_late`'s four 8192-wide weight-vector loads above the PDL wait. That kernel is the
T=256 boundary (t > 128, one replica), so the revert touches v32S and v32F only; every other pdl_M edit stays.

## Prepare (the binding budget)

Round 1's first lab (`3597359b`) died at 239 s in prepare (`prepare budget before mixed_parallel oracle`); its
retry `7c0fd3cc` measured S19's host prepare at **254.7 s** of 300 (S18's scored prepare: 241.6 s). Timeline
(start s, seconds): `_prepare_weights` (0.1, 103.6) under the compile pool (last build 91.9 s); adapters 3.5; MoE
tuners 6.8; projections 8.6; attention portfolio 16.7; VQ8/cluster/r2/partial-port gates 22.0 (r2's compile was
serial); glue tuner 13.8; banks 5.8; BEST7 gates 9.1; base prepare ends at 202.5; NF port prepare 33.1 serial; delta
chain 15.4 (W's O-only compile 7.5).

`bake()` (every candidate and arm) keeps the compiles and the weight coding and removes the rest:
- Picks become constants from S18's scored log `28b5c189`: projections (PROJ_PINS plus the measured cuBLAS at
  (1,'o')), Short-Lt and Lt bindings, MoE NT plan. Nothing reads the attention, best6-MoE or glue tables (every
  BEST7 row names its attention and expert, and `_s15n_plan` maps every shape onto a BEST7 row), so those tuners
  are gone, not baked.
- Oracles removed (all private FP32 gates; TASK's checker is the only correctness gate): the mixed_parallel and S4
  expert oracles and the fixed-expert/attention gates in `_best7_prepare_settings`; the V9 NT1/NT2 and best6
  oracles; the attention portfolio's admission gates; the VQ8, cluster, r2 and partial-port FP32 gates; the glue
  oracle; the M1 O-cell gate.
- Compiles: r2's DSM kernel and PG_RAW move into the parallel pool; the unread XQA/P8 attention extensions and the
  prefill adapters (and their grouped module) are no longer built; NF port coding runs in a thread beside the base
  prepare (no prepare step captures a graph any more).
- A last delta step (`warm`) runs every timed shape once at ctx 257, so every module loads and every kernel JITs in
  prepare, never inside a capture.

After (`6ac9317e`): **147.7 s**. Base prepare 120.3 s: `_prepare_weights` (V9 repack, then V9's compile, beside
the pool) 106.5 s, the pool's last build (`_r2_build`) 94.7 s, NF coding in its thread 109.6 s; banks 6.6; BEST7
settings 3.5; tuners 0.1; delta chain 27.3 (warm 17.6). The repack/compile split inside `_prepare_weights` is
unmeasured; moving V9's compile into the pool is the next cut if prepare binds again.

## Contracts (static preflight)

`contracts.py` runs before any GPU work and fails the build if a component breaks one: every flags word the
small-M dispatch can pass resolves in the cell it reaches and is bound on `s4_nt1_bank`; no component rebinds
prepare (S19's prepare defs are S18's plus the bake's edits) and each delta step calls its predecessor; every plan
table the base prepare reads is the same text in host and arm (the incumbent reads the host's base state in the lab)
and the baked picks reproduce `28b5c189`'s BEST6_BASE_SHAPE / BEST7_PLAN / BEST5_NT lines; every shared-bank
producer is the same text in host and arm, and each bank key a bind adds has one writer. `CONTRACTS` declares each
component's needs and products. Mutations (a flags word, a prepare rebind, one pin) each fail it.

## Lab

`build_lab.py` builds the candidate, its arms and the import-time pair on the lab-only prefill host
(`prefill_host.py`, 780 s budget). INC = frozen S18 at every shape; each other arm runs only at its rows and takes
A2's time elsewhere. Round 2 arms re-test round 1's row choices on fresh seeds: P10_d16 (P10 also at T=16),
late_pdl (pdl_M's `_boundary_late`), no_FAST. Seeds `610351273 + 104729*i` (16), weights/router
`610351271/610351269`; S18's and round 1's confirmation seeds joined the selection set. Read a receipt with
`report.py`; the arms are optional evidence, so S19 freezes on A-vs-INC even if an arm fails.
