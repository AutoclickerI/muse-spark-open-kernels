# Decode checks — next send order after b787866

The V9 and V9-dispatch scored runs are already at the judge. Three decode scored slots remain today; do not wait for their receipts before sending the next information-dense candidate. The latest completed receipt is `85e67f5d6ef945c9`, 1.366737x (CI [1.346236, 1.387237]).

## Frozen source set

| Source | SHA-256 | Local validation | Role |
|---|---|---:|---|
| `cand_r3_v9_autotune.py` | `07d19d3c251e392924c808c709626cd11ccd146f9a2c4f7475c467e32299a000` | 13/13 tests, 12/12 graphs; prepare 2.8 s on sm_120; portable FA3/Lt glue PASS | **ready to send first**; judge-side autotune over V7/V9/grouped MoE, FA3, projection and PDL variants |
| `cand_r3_v9_dispatch.py` | `633c96ccffa40c6673c99b808c18b28685db5aafd4541667de68e1f0e6c41b84` | scored; receipt `85e67f5d6ef945c9`, 1.366737x | already sent; current H200 incumbent |
| `cand_r3_v9.py` | `ba990a86f6f7f4985345cfc89a3bc56e36730d974ba47642a344c5c77d84279f` | scored; receipt pending/queued | already sent; V9-only layer |
| `cand_round3_v7.py` | `a3ae91dd3e42bf7edca26824012272e33376c890f41da906929f5b8eedd73d61` | scored; receipt `579855768bd24be6`, 1.353614x | historical V7 control |
| `cand_moe7.py` | `eb659038a74c628794b585e4579901dfea16fcd23c50eddf7862b5d84b3dcc6c` | scored; receipt `d2bf1ced702948a4`, 1.355430x | historical control |

Structural sources remain frozen for later diagnostics: `cand_layer_sweep.py` (`08015a0aede234527989a100bb661461dbee843b459844003bb8de59c128d096`), `cand_epilogue.py` (`2c585251f289b140b50d4c0465ec50cf4996bdf8d870ce19c1cb903b1d320598`), `cand_chain6.py` (`6e88cfa39935592a081bb95461ee82610c17b0cd7689209c4ea4e07cc72caf11`), `cand_chain.py` (`0d545fb6fe330e24dbf99f8f21bbb6444a342d945776ab5708f64d447faac4b9`), `cand_chain_occ2.py` (`96cf5ecc7141c2aa2df95ff0724d3dcb79a1cc7a38b5565b0374edfa3e9b5581`), and `cand_chain_nopf.py` (`486356e4e552ebcd4ffca1ead3f808be7144b7c53438009bca4f5374cd554e52`). Their local structural round regressed the complete layer, so they are not production checks until a packed diagnostic finds a neutral stage pair.

## Send order

1. **`cand_r3_v9_autotune.py`** (SHA `07d19d3c251e392924c808c709626cd11ccd146f9a2c4f7475c467e32299a000`). Send as the next scored run. One judge run measures and selects V7/V9/grouped MoE, all 27 FA3 settings for each verify shape, custom/cuBLAS/cuBLASLt projection variants, and PDL on/off at both boundaries.
2. Do not resend `cand_r3_v9_dispatch.py` or `cand_r3_v9.py`; their scored runs already occupy the queue and provide the incumbent H200 evidence.

The autotune log is intentionally part of the scored receipt: `FA3_TUNE`, `PROJ_TUNE`, `PDL_TUNE`, and `MOE_TUNE` lines record every accepted timing and selected plan. The local screen uses sm_120; FA3 and cuBLASLt H200 choices remain unverified until this scored run.

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
