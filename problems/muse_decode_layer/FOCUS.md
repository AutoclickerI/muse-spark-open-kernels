# Decode layer: integration and scored runs (P4)
Serves: G1 (our decode-layer speedup is the overall score). Lever table and its refresh: [dev/NEXT_CHECKS.md](dev/NEXT_CHECKS.md).

Incumbent before S9: S8 `cand_r3_best7_s8.py` cb84645d1af8b813529a64cfe746dd5846160860deebbe8afd8154007a23c366, 1.5742x
`a5b2f55b` (10-25; judge prepare 215.0 s). Per-load results of every S run: FINDINGS "Whole layer and integration".
Chain: S1 recoded scale bank (`0a43baec` 1.4667x) -> S2/S3 fused M=256 -> S4-S6 withdrawn (a BF16-valued scale copy
made in prepare, CONSTRAINTS 5) -> S7 H_H on uint8 scales (`2f6471d5` 1.5597x) -> S7R + M=256 R_full (`c1fbd0cf`
1.5735x) -> S8 + attention early release, M=128 verify qkv split-K, cluster glue at t <= 32.
Unchanged shapes differ by up to ~4.5% between runs: promote from paired labs on identical inputs and many seeds.
Before a cell enters a candidate, list what each prepare() tensor holds at what precision (CONSTRAINTS 5).

S9 `cand_r3_best7_s9.py` f55f616254a88e33781e897de0a8bfbfb82a410d1a03f1da8e8cae0f2de23f6e: 1.6073x `b7c59570`
(our board score; no own-time gain at decode 1 / 32 and verify 1, a paired A/B decides). Built by `dev/s9/build.py [--base] [--patch ...] [--flags]` (asserted replacements; reads the
other lanes' builders and cells, so a rebuild picks up their updates):
- BF16 qkv front: `../muse_bf16_proj/dev/attn_front/patch_front_bf16.py` (= S8_front 77f477fe, `87cd3a84`).
- N_full at M=128/256 from `dev/s7n` (0.8842x / 0.8701x, `c01b1916` / `14f942d5`); R_full, mixed_parallel and
  pm_call deleted.
- Small-M flags {1:95, 8/16/32:69} (cell 33e5d7a8, `cf1de160`); S4's NT1 deleted.
- Audit (`dev/s9/audit.json`): every expert bank holds FP4 codes plus uint8 scale codes, 9.0 B per 16 weights;
  nothing widened. Local cold prepare 229.8 s (`CCACHE_DISABLE=1`; ccache hides compile time).

S10 `cand_r3_best7_s10.py` 6688fad9 (sent 12:20; `dev/s10/build.py`): S9 on N_full fd04d8c4 (heavy branch first,
0.9799 / 0.9882 `880c3881`), LT and the layer attention ext in prepare's native pool (local cold 186.0 s).
Next, S11: balanced split-K GEMVs (`../muse_bf16_proj/dev/gemv_bal/patch_gemv_bal.py`) once chain `c2b2da75`
agrees; the gate split (P7); coded dense GEMVs (P10) after a whole-layer paired lab.
Rerun `dev/levers/build.py` on each new incumbent. Parked: megakernel at +1.74% estimated (`dev/megakernel/README.md`).
