# Decode checks — order for 2026-10-24 09:00 KST

The previous check window is exhausted. Do not launch another decode check before the 09:00 KST reset. The best completed whole-layer check is `cand_moe7`, 1.354968x (`66d5d4c9e4054fa9`); its scored submission is already in flight. `submission.py` is the exact `cand_moe7` source, SHA-256 `eb659038a74c628794b585e4579901dfea16fcd23c50eddf7862b5d84b3dcc6c`.

The previous `cand_hybrid` check is resolved at 1.307138x and loses v7 overall. Do not duplicate it or spend a new check on an already resolved source.

## Frozen source set

| Source | SHA-256 | Local validation | Role |
|---|---|---:|---|
| `cand_moe7.py` | `eb659038a74c628794b585e4579901dfea16fcd23c50eddf7862b5d84b3dcc6c` | 13/13 tests, 12/12 graphs | current H200 winner; v7 experts, glue dispatch, tuned GEMV/router, FA3 verify |
| `cand_round3_v7.py` | `a3ae91dd3e42bf7edca26824012272e33376c890f41da906929f5b8eedd73d61` | 13/13, 12/12; prepare 18.98 s | v7 plus custom q_len=1 attention for batch 1 as well; FA3 for q_len=8 |
| `cand_r3_moe_dispatch.py` | `dfe0d8bbc569f0636e4e41b98952cc80baf469bdb3bc0eb634f0e071b2306adf` | 13/13, 12/12; prepare 1.83 s; SM90a compile | v7 + canonical-bank V8/grouped wide-tile MoE dispatch, synthetic shape autotune |
| `cand_r3_v7_hybrid.py` | `2102afbad7424620bff19ad75c8765f2d473d2206520ae9930c4e1bbcb727dcd` | 13/13, 12/12; prepare 58.68 s | `cand_round3_v7` plus Humming2 only at T≥256 (verify-32 hypothesis) |
| `cand_tuned.py` | `0d10613d0b92f9b63df158436d521bef8bdc1a3fa6a741a6a5b39f18ae4e789f` | 13/13, 12/12 | control, 1.270858x H200 |
| `cand_v8.py` (MoE lane) | `7ab6818fcbb8f28b53686ed90362e4950fe4a642891836dfab9bcabb56c390be` | MoE local only; no SM90/H200 receipt | hold; V8 is unverified and locally regresses V7 |

Structural sources remain frozen for later diagnostics: `cand_layer_sweep.py` (`08015a0aede234527989a100bb661461dbee843b459844003bb8de59c128d096`), `cand_epilogue.py` (`2c585251f289b140b50d4c0465ec50cf4996bdf8d870ce19c1cb903b1d320598`), `cand_chain6.py` (`6e88cfa39935592a081bb95461ee82610c17b0cd7689209c4ea4e07cc72caf11`), `cand_chain.py` (`0d545fb6fe330e24dbf99f8f21bbb6444a342d945776ab5708f64d447faac4b9`), `cand_chain_occ2.py` (`96cf5ecc7141c2aa2df95ff0724d3dcb79a1cc7a38b5565b0374edfa3e9b5581`), and `cand_chain_nopf.py` (`486356e4e552ebcd4ffca1ead3f808be7144b7c53438009bca4f5374cd554e52`). Their local structural round regressed the complete layer, so they are not production checks until a packed diagnostic finds a neutral stage pair.

## 09:00 order

1. **Orchestrator check `cand_r3_moe_dispatch.py`** (SHA `dfe0d8bbc569f0636e4e41b98952cc80baf469bdb3bc0eb634f0e071b2306adf`). It is the packed large-M diagnostic; do not launch it from this lane: v7, canonical-bank V8, and grouped wide tiles are selected by synthetic `prepare()` timing. It directly targets verify-16/32, where the accepted profile puts 49.8–76.9% of time in experts, and decode-32 sliding, where experts consume 54.8%.
2. **Keep `cand_r3_v7_hybrid.py` as the fallback check** (SHA `2102afbad7424620bff19ad75c8765f2d473d2206520ae9930c4e1bbcb727dcd`). Its Humming2 threshold was locally screened and its old whole-layer check lost overall (1.307138x), so it follows the new canonical-bank dispatch rather than preceding it.
3. **Keep `cand_round3_v7.py` as the attention isolation check** (SHA `a3ae91dd3e42bf7edca26824012272e33376c890f41da906929f5b8eedd73d61`). It tests q_len=1 batch-1 attention with v7 experts and FA3 verify.
4. **Only then pack structural fusions into a diagnostic.** The existing chain/epilogue sources all pass local correctness but regress complete local layers; use them as controls for fused residual/router/norm/gate launches, not as production checks.
5. **Hold V8-only changes** until an H200 receipt and SM90 compile evidence exist. The current V8 occupancy-only source is not independently verified; the new diagnostic uses it only alongside measured v7 and grouped alternatives and records its synthetic choice.

The new dispatch CUDA static compile is recorded in `dev/sm90a_dispatch_compile.log`: the grouped kernels compile with zero spills; the V8 kernels compile but retain small spill frames.

Every candidate in this order must first pass `tools/img python tools/local_eval.py` with 13/13 tests and 12/12 fresh-input graph cases. Force every Hopper-only Python branch through a portable stand-in before remote submission. Remote checks use `tools/okc_run.py` and remain detached; no commit is requested.

## Resolved H200 receipts

| Candidate | Receipt | Overall | D1 | D16 | D32 | V1 | V16 | V32 | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `cand_tuned` | `0fa94cd98ba34c93` | 1.271 | 1.470 | 1.370 | 1.091 | 1.568 | 1.081 | 1.018 | control |
| `cand_glue` | `d32113fa98f946a3` | 1.283 | 1.473 | 1.409 | 1.101 | 1.597 | 1.068 | 1.034 | superseded |
| `cand_all_decode` | `8577b1a2f9054ca2` | 1.289 | 1.524 | 1.350 | 1.093 | 1.565 | 1.142 | 1.065 | superseded by v7 |
| `cand_native_glue` | `13e9ea00088b4bb8` | 1.253 | 1.528 | 1.293 | 0.994 | 1.572 | 1.087 | 1.020 | reject; large-T loss |
| `cand_moe7` | `66d5d4c9e4054fa9` | **1.355** | 1.529 | 1.489 | 1.173 | 1.695 | 1.169 | 1.091 | current best |

`cand_hybrid` receipt `ecc0d2615e294e0d` has now resolved: it passes at 1.307138x (CI [1.287531, 1.326745]) with D1/D16/D32 = 1.493/1.392/1.097 and V1/V16/V32 = 1.591/1.062/1.203. It is rejected as the whole-layer winner because it loses v7 overall, although its V32 result motivates any later per-load hybrid diagnostic.
