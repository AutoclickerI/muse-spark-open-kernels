# Decode checks — order for 2026-10-24 09:00 KST

The previous check window is exhausted. Do not launch another decode check before the 09:00 KST reset. The best completed whole-layer check is `cand_moe7`, 1.354968x (`66d5d4c9e4054fa9`); its scored submission is already in flight. `submission.py` is the exact `cand_moe7` source, SHA-256 `eb659038a74c628794b585e4579901dfea16fcd23c50eddf7862b5d84b3dcc6c`.

The previous `cand_hybrid` check is resolved at 1.307138x and loses v7 overall. Do not duplicate it or spend a new check on an already resolved source.

## Frozen source set

| Source | SHA-256 | Local validation | Role |
|---|---|---:|---|
| `cand_r3_v9.py` | `ba990a86f6f7f4985345cfc89a3bc56e36730d974ba47642a344c5c77d84279f` | 13/13 tests, 12/12 graphs; prepare 1.3 s | V9 persistent expert kernel replacing V7; same packed NVFP4 ABI and custom q_len=1 attention |
| `cand_r3_v9_dispatch.py` | `633c96ccffa40c6673c99b808c18b28685db5aafd4541667de68e1f0e6c41b84` | 13/13 tests, 12/12 graphs; prepare 1.4 s; dispatch synthetic timing | V9 baseline plus grouped wide-tile selection for large M |
| `cand_moe7.py` | `eb659038a74c628794b585e4579901dfea16fcd23c50eddf7862b5d84b3dcc6c` | 13/13 tests, 12/12 graphs | current H200 whole-layer winner; incumbent until V9 checks resolve |
| `cand_round3_v7.py` | `a3ae91dd3e42bf7edca26824012272e33376c890f41da906929f5b8eedd73d61` | 13/13, 12/12; prepare 18.98 s | V7 attention isolation fallback |
| `cand_r3_moe_dispatch.py` | `dfe0d8bbc569f0636e4e41b98952cc80baf469bdb3bc0eb634f0e071b2306adf` | 13/13, 12/12; prepare 1.83 s; SM90a compile | prior V7 + canonical-bank V8/grouped dispatch; superseded by V9 dispatch |
| `cand_r3_v7_hybrid.py` | `2102afbad7424620bff19ad75c8765f2d473d2206520ae9930c4e1bbcb727dcd` | 13/13, 12/12; prepare 58.68 s | fallback Humming2 verify-32 hypothesis |
| `cand_tuned.py` | `0d10613d0b92f9b63df158436d521bef8bdc1a3fa6a741a6a5b39f18ae4e789f` | 13/13, 12/12 | control, 1.270858x H200 |

Structural sources remain frozen for later diagnostics: `cand_layer_sweep.py` (`08015a0aede234527989a100bb661461dbee843b459844003bb8de59c128d096`), `cand_epilogue.py` (`2c585251f289b140b50d4c0465ec50cf4996bdf8d870ce19c1cb903b1d320598`), `cand_chain6.py` (`6e88cfa39935592a081bb95461ee82610c17b0cd7689209c4ea4e07cc72caf11`), `cand_chain.py` (`0d545fb6fe330e24dbf99f8f21bbb6444a342d945776ab5708f64d447faac4b9`), `cand_chain_occ2.py` (`96cf5ecc7141c2aa2df95ff0724d3dcb79a1cc7a38b5565b0374edfa3e9b5581`), and `cand_chain_nopf.py` (`486356e4e552ebcd4ffca1ead3f808be7144b7c53438009bca4f5374cd554e52`). Their local structural round regressed the complete layer, so they are not production checks until a packed diagnostic finds a neutral stage pair.

## 09:00 order

1. **`cand_r3_v9.py`** (SHA `ba990a86f6f7f4985345cfc89a3bc56e36730d974ba47642a344c5c77d84279f`). Send first: V9 is H200-verified at 1.654x standalone and is the largest decode-layer lever.
2. **`cand_r3_v9_dispatch.py`** (SHA `633c96ccffa40c6673c99b808c18b28685db5aafd4541667de68e1f0e6c41b84`). Send second: V9 plus shape-only grouped timing selection. Local choices were V9 M=1/8/16/32/64, grouped M=128/256; do not treat these as H200 results.
3. **Projection + boundary diagnostic** `dev/r3_v9_projection_boundary_diag.py` (SHA `65cb943ff0ba38113c3c76621e663882c1a4ae012930e6b802ee2890a7b5cc82`). It times the portfolio pair at M=128/256 and PDL boundary fusion; its portable sm_120 results are in `dev/r3_v9_projection_boundary_diag.json` and `.log`.
4. Keep `cand_r3_v7_hybrid.py` as the fallback only if both V9 integrations lose.
5. Keep structural chain/epilogue sources diagnostic; prior complete-layer screens regressed and they need a neutral packed stage pair before production use.

The V9 and dispatch sources preserve the layer's routing, output, and cache ABI; only the expert implementation and dispatch state change. The dispatch CUDA paths are shape-only and repack weights losslessly in `prepare()`.

The portable diagnostic found projection GEMV/dense parity at M=128/256 within measurement noise; at M=128 the pooler/unpooler dense stand-in was 1.032x slower, while at M=256 pooler was 0.979x. Boundary fusion regressed the complete local prefix at every tested M (PDL fused/unfused ratios 0.726–0.780 with PDL on), so it remains diagnostic.

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
