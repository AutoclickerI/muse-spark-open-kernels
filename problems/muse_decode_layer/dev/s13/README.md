# P4: S13 integration

Base: S12 final `cand_r3_best7_s12_final.py` **240248b2** (S11 final + T1 coding + F4, no wide).
`build.py` pins the base, every patch, and all source dependencies; every piece is independently removable.
`patch.py` is the aggregate repository handoff onto that exact S12 final. No S13 final is frozen yet.

Pieces:

- `attention`: snapshot of P5 `patch_partial_port.py` / `partial_port.py`, restricted to decode-1 sliding/full.
  Port source still derives from exact S11 final, pinned as a dependency. H200 c114c35e787f46d1: whole-layer
  new/old 0.97091 at decode 1 (sliding 0.96960, full 0.97415), 16 seeds x8, exact sincos. Only the input
  base assertion changed to accept S12 final; kernels unchanged.
- `cpall`: all-thread cp.async staging, one wait/barrier; `cpall_frozen.py` is extracted losslessly from
  c446fdab's SOURCES['cpall'], replacing S12's v3 module. T1 qkv/pool_router only, k-major encoded banks,
  same split counts 22/11. H200 c446fdab439b44e5: qkv 10.34 vs balanced 11.66 us and v3 k-major 10.86;
  pool_router 18.19 vs balanced 21.32 and v3 k-major 18.49. S11's gate split bypasses full-qkv coding.
  Comparison against the actual S12 coded path is required; the standalone win is not promotion evidence.
- `permw`: snapshot of P3b `patch_s12_permw.py`, preceded by S12's pinned wide-bank patch because S12 final
  dropped wide. One combined piece (wide bank + rotated packed scale words + hoisted descriptors).
  H200 b88d04fe86aa45a7 vs F4/LI4: 0.9862 / 0.9787 at M128/256, 12 seeds x3, all TASK PASS.
  Prepare keeps uint8 scale codes after an invertible 32-bit rotation, no precision expansion.

`audit.py` invokes local_eval with S12's complete retained-state and dispatch inventory. `--reuse-static`
reuses S12 final's successful Hopper compiles only for exact CUDA text and compiler flags (same judge image),
compiling changed sources through the original checker at concurrency 2. Python static checks, all 13
runtime tests, all 12 graph shapes, and forced Hopper glue run normally. This avoids the host memory
exhaustion from concurrent duplicate static compiles; it does not skip changed code.

`s13ab_build.py` composes complete arm namespaces and losslessly compresses them with asserted reconstruction
hashes. Native coded modules are independent per arm. Each representation-changing arm has its bank rebuilt
from identical original weights; unchanged small-M banks/counters remain shared. Every shape has an A:A arm.
16 seeds x8 cold rotating rounds, full correctness gates, 46/16 weighted load ratios, per-load A:A ranges
and paired standard errors. Untouched loads are controls, and each retained piece must clear A:A on every
load it touches. A standalone or one-run host speedup never promotes a piece.

Initial attention+cpall trial **632e5b52** passed all 13 local tests and 12 graph shapes; its duplicate static
compile phase was stopped under host memory pressure when PERMW arrived. It was not sent remotely.
All-three candidate **62eca97e** passes normal and forced: each 13 tests, 12 graph shapes, 24 static checks. The complete wrapper **ef002c71** (1,001,892 bytes; JSON source field 1,023,142 bytes) passes its 12-shape/16-seed/8-round local lab, all gates, 13 tests and 24 static checks. `manifest.json` and `admission.json` pin the exact hashes. The local A:A is wide under contention; no sm120 timing is used for promotion.

Initial wrapper **ef002c71** was rejected before a job existed: 1,001,892 bytes exceeds the server limit of 1,000,000 (not 1 MiB). No check consumed. The revised envelope removes only outer Python comment tokens, asserting an identical AST including every CUDA literal; candidate and embedded arm sources remain exact. Revised wrapper **351907fb**, 989,808 bytes, passes the same complete 12-shape/16-seed/8-round lab, 13 tests and 24 static checks. `admission.json` and `s13ab_manifest.json` pin this revision. Accepted as decode check **3dbcaccd1bb042a4**; collector running. Result pending.
Next: collect and report its per-piece/load ratios against A:A. While it runs, prepare the no-cpall variant for a possible follow-up (a sub-percent standalone gain may not clear whole-layer A:A).
Freeze S13 final only for retained pieces supported beyond A:A. Orchestrator owns the scored send.
