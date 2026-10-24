# Decode layer focus — G3 first

H200 board control: V9 `0e94268cf5654012`, **1.428x**. Serving uses verify shapes.
Current candidate: [cand_r3_best5.py](cand_r3_best5.py), SHA-256
`a651af25cb44550932f497b57eb82fb32c0fbe07c1f21a8592bd0982f937c150`. Jihoon sends it as a decode check; this lane did not send.

Base is exact best `812f5f36` (H200 `7b2165c8`: 1.351x, prepare 130.547 s).
Best5 adds only per-run V9 NT1/NT2 selection at M32/128/256 on the actual loaded router.
Three fresh inputs, independent FP32 gates before/after 12 alternating cold graph pairs;
`BEST5_NT` prints both times, ratio, choice, errors and route counts per M.
NT1 and all non-MoE native sources/stage bodies are unchanged. NT2’s kernel body is byte-identical
to accepted H200 lab `3f2b8da0f1d244d7`; no CUTLASS/reuse/hot-first/NT4 code is carried.
Both paths read the existing packed NVFP4/E4M3 bank; oracle dequant scratch is discarded.

Normal + forced screens: **13 tests, 12 fresh graph shapes, 13 static checks** each.
Cold prepare: **190.343 s**; extra forced checks: **192.029 s**, with ccache disabled and empty
extension/Triton/CUDA caches. One deadline from prepare entry guards all six tuners (audited expired).
Local MoE gate error ~0.17%; NT2 selected locally at all three M, which does not establish H200 speed.
Complete best5 H200 agreement, selections and prepare remain unverified.
Commands, provenance and validation: [best5 handoff](dev/best5/README.md).

Omni `a704a37c` aborted in CUTLASS TMA descriptor construction after compiling all extensions;
its Python gate could not catch a C++ abort. This is the existing FINDINGS failure, not a new rule.
Attention sweep `59e53e74` has a faulty gate; its q1 attn_main/q8 FA3 timings are diagnostic.
The current best5 keeps best’s attention and projection choices unchanged.
Stage lever table, lane handoffs and resolved receipts: [NEXT_CHECKS](dev/NEXT_CHECKS.md).
