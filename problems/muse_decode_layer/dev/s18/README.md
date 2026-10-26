# S18 integration

Base: frozen **S17 fea71760** (`cand_r3_best7_s17_final.py`; [../s17/frozen.json](../s17/frozen.json)), scored
`7ec5f9ab` 1.828637x, paired `da0cd64b` R 1.003310 [1.001006, 1.005620] over S16 ee5fe249. The board's best
entry is the first S17, `5f35323a` 1.837887x (unpaired).

Candidate **S18 95df42fe** (expanded sha; `cand_r3_best7_s18_95df42fe7b3a.py`, shipped sha `e04635ec`),
composed by `compose.py` from pinned cells (`capture.py` -> `cells.json`):

| Piece | Shapes | Evidence | Leave-out arm |
|---|---|---|---|
| INDEX1: P3a SHARD2+INDEX small-M cell | M1 | INDEX/BASE .99252 (SE .0025), INDEX/S15 .99156, `d7fb2d8b`; SHARD2/PAIR .99179, `b57d691d` | no |
| P3a PAIR cell | M8 | c49 PAIR/S15 .98756, `c49dbdff` | no |
| BLOCK: P3a PAIR+BLOCK cell | M16, M32 | COMBO/BEST .99550 / .99312 (SE .0028 / .0025), `c49dbdff` | yes |
| pdl_M: immutable metadata before PDL wait | source-wide | R 1.003169 [1.002493, 1.003846] on S14, `0f74021a` | no |
| pdl_D: prior d1-full glue (GP kernels) | d1F b1/route/b2 | R 1.002841 [1.002001, 1.003683] on S14, `0f74021a` | no |
| Vwait16 / Vwait32: redundant full-attention V wait removed | v16F / v32F | .997130 (SE .00076) / .998618 (SE .00093), `1bab0b2d` | Vwait32 |
| D8 / D4: P5 DSM sliding | d16S / d32S | .995283 (SE .0072) / .989822 (SE .0076), `27a914f7` | both |
| pool128: S16 native pool_router T128 | v16 | S16 leave-out .999307 [.997843, 1.000773] (`37c2879f`); S15 pool-only .999399 [.998044, 1.000755] but C/C_no_pool 1.006161/1.004273 (`60bad72b`) | yes |

Already in S17, so not re-added: CHUNK4 at M128/M256 (P3b `7e382263` .99750 / .90968; CHUNK3 at M256 .94604
is worse) and every other piece of composition C (attention consumer, large-M and VDSK release, O128). On
S17, C adds only pool128; `compose.py` asserts C's other edits are already byte-identical in S17. pdl_M
replaces the half-build's pdl_B: B is M plus `_gb`/`_gi` L2 hints and measured lower on the same seeds
(1.002694), and the gate audit picks source-wide policies by maximum R.

## Fixes to the inherited half-build

- The 13:23 screen died in prepare (`KeyError: 'p10_pool_raw'`): `patches.patch(pool128)` ends by rebinding
  `prepare` to the S16 origin chain, which dropped S17's MT, P10-pool (its `_project_variant` swap during
  tuning) and RT wrappers. `compose.py` strips that rebinding; S17's chain already ends in the global
  `_s16_prepare_delta`.
- Prepare's FP32 expert gate runs `smallm_experts` right after `_s4_prepare_banks`, before the delta: the
  P3a cells now bind there too, with every `SMALLM_FLAGS` word (`{1: 95, 64: 69}`; the half-build indexed
  it by M and raised `KeyError: 8`).
- The three P3a modules and GP compile in prepare's parallel native pool, sized 15 + one worker per added
  build: S17 prepare took 198-255 s of the 300 s cap on the judge, and serial compiles would not fit.
- Lab: each arm rebuilds `r2_rows` from its own `_R2_SELECTION`; otherwise INC (S17) would inherit S18's
  D8/D4 rows from the host's prepared state and time them.
- Local lab screen: no_Vwait32 loads S17's VQ8 for verify 32 as its own `_s18_vwait_off_module`, a Hopper ring
  that sm120 cannot launch (illegal address at v32F). `screen.py` gives it the same local VQ8 stand-in S15's
  screen gives `_vq8_module`; its exact CUDA compiles for sm_90a in `audit_native.py`.

## Validation (judge image, GPU0)

- Exact shipped file: 13/13 TASK and 12 graph shapes in normal (`normal.log`) and forced (`forced.log`, on
  d3a9719e, which differs only in the pool size) modes; forced errors equal S17's.
- `audit_native.py`: 228 undefined-name, call-signature and sm_90a compile checks over S18 and all five
  leave-out arms, including the dictionary-held P5 kernels (`native_audit.json`).
- Exact paired wrapper `a2514bca` (`screen_lab.py`), normal and forced: complete, 1536 TASK + 260 nearby TASK
  checks, no failures; the host passes the 13 official tests. Route-ID agreement diagnostics differ at d16S /
  d32S between D8 / D4 and their controls (2/16, 12/32 rows); TASK passes on every arm and seed.
- `validation.json` records all of the above. Local timings run stand-ins (one-CTA DSM, S14 VQ8 ring, pool128
  CHAIN as torch.mm splits) and rank nothing.

## Runs

- Scored: `28b5c189e7b5469c` passed, **1.848375x** [1.820649, 1.876101], prepare 241.6 s of 300 (S17's single
  scores: 1.828637x, 1.837887x; unpaired, so only the paired check ranks S18 against S17).
- Paired check (`lab/lab_s18_95df42fe7b3a.py`, wrapper `a2514bca`, pinned in `batches/a2514bca91f81922/`): S18
  (A, A2) vs INC = S17 plus leave-outs no_Vwait32, no_BLOCK, no_D8, no_D4, no_pool128; 16 held-out seeds
  `404910273 + 104729*i`, fresh weights/router seeds `404910271/404910269`, 1536 public TASK checks plus 260
  nearby TASK checks. Decode checks for 10-26 are spent: sender 2880917 waits for the reset (10-27 09:01 KST) at
  priority -1 (`runs/detached/a2514bca91f81922.log`); the sender writes the receipt (`tools/okc_run.py collect` if
  it dies).

Gate (`../s16/score_gate.py`): promote S18 when the lower 95% bound of the six-load geomean R (46 sliding +
16 full per load) over the held-out seeds is above 1 and every arm passes TASK. Leave-out R > 1 favours
keeping a piece; it is a diagnostic, not a veto. A composition selected from these leave-outs needs fresh
seeds. Read a receipt with `python3 problems/muse_decode_layer/dev/s18/report.py <receipt.json>`; on
promotion it writes `cand_r3_best7_s18_final.py`.

Rebuild from the repo root: `python3 problems/muse_decode_layer/dev/s18/build_lab.py`, then
`CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_decode_layer/dev/s18/screen_lab.py [--forced]`.
