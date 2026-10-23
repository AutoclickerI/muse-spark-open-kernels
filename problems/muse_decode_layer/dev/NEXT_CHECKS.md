# Decode checks — next send order after dafac99

The V9, V9-dispatch, and cost-autotune sources are already at the judge; the cost-autotune source is first in the queue from `dafac99`. Do not resend or wait for its receipt before sending the next information-dense candidate. The latest completed receipt is `85e67f5d6ef945c9`, 1.366737x (CI [1.346236, 1.387237]).

## Non-MoE cost ranking from receipt `85e67f5d6ef945c9`

| Load / layer | Largest non-MoE cost | Next | Glue / launch remainder |
|---|---:|---:|---:|
| decode 1 sliding | projections 47.5 us | attention 14.8 us | boundary 12.3 us |
| decode 1 full | projections 46.1 us | attention 33.8 us | boundary 12.7 us |
| decode 16 sliding | projections 42.8 us | attention 20.2 us | boundary 9.1 us |
| decode 16 full | attention 116.7 us | projections 42.0 us | boundary 9.2 us |
| decode 32 sliding | projections 46.9 us | attention 20.6 us | boundary 9.1 us |
| decode 32 full | attention 227.9 us | projections 56.1 us | boundary 15.2 us |
| verify 1 sliding/full | projections 47.2/47.1 us | attention 20.3/32.0 us | boundary 14.0/13.6 us |
| verify 16 sliding/full | projections ~73 us | attention 27/192.6 us | boundary 11.6/11.2 us |
| verify 32 sliding/full | projections ~91 us | attention 30/373.8 us | boundary 19.9/19.6 us |

The fused prefix and persistent island variants target the projection/glue rows at small loads; full attention is the largest remaining decode cost at verify32/decode32/verify16 = 373.8/227.9/192.6 us. The MoE lane owns its V11/V17 verify row-tile work.

## Frozen source set

| Source | SHA-256 | Local validation | Role |
|---|---|---:|---|
| `cand_r3_v9_attention_portfolio.py` | `257674016ee2b5b3ef12a11c2e2af90406d90e9c43efe0e08c5ce2d37373d8f2` | 13/13 tests, 12/12 graphs; prepare 4.87 s on sm_120; Hopper branches forced through layer stand-in | **ready to send next**; adds the attention-lane FA3/XQA/WGMMA/FP8-PV/q1-gather portfolio per scored shape |
| `cand_r3_v9_cost_autotune.py` | `b3fcd1eaa3f7f84c7d74e381ae5564f042ef9324659aef21d9b01ca851fb9d2e` | 13/13 tests, 12/12 graphs; prepare 3.3 s on sm_120; portable FA3/Lt glue PASS | sent first in the queue from `dafac99`; fused prefix and persistent small-load paths |
| `cand_r3_v9_autotune.py` | `099ee216693f890b1bc6b737de5d3441e5f210baa2cca0c8fbb0b4fd65ecbc6e` | sent in commit `8584a56`; scored/queued | already sent; original MoE/FA3/projection/PDL portfolio |
| `cand_r3_v9_dispatch.py` | `633c96ccffa40c6673c99b808c18b28685db5aafd4541667de68e1f0e6c41b84` | scored; receipt `85e67f5d6ef945c9`, 1.366737x | already sent; current H200 incumbent |
| `cand_r3_v9.py` | `ba990a86f6f7f4985345cfc89a3bc56e36730d974ba47642a344c5c77d84279f` | scored; receipt pending/queued | already sent; V9-only layer |
| `cand_round3_v7.py` | `a3ae91dd3e42bf7edca26824012272e33376c890f41da906929f5b8eedd73d61` | scored; receipt `579855768bd24be6`, 1.353614x | historical V7 control |
| `cand_moe7.py` | `eb659038a74c628794b585e4579901dfea16fcd23c50eddf7862b5d84b3dcc6c` | scored; receipt `d2bf1ced702948a4`, 1.355430x | historical control |

Structural sources remain frozen for later diagnostics: `cand_layer_sweep.py` (`08015a0aede234527989a100bb661461dbee843b459844003bb8de59c128d096`), `cand_epilogue.py` (`2c585251f289b140b50d4c0465ec50cf4996bdf8d870ce19c1cb903b1d320598`), `cand_chain6.py` (`6e88cfa39935592a081bb95461ee82610c17b0cd7689209c4ea4e07cc72caf11`), `cand_chain.py` (`0d545fb6fe330e24dbf99f8f21bbb6444a342d945776ab5708f64d447faac4b9`), `cand_chain_occ2.py` (`96cf5ecc7141c2aa2df95ff0724d3dcb79a1cc7a38b5565b0374edfa3e9b5581`), and `cand_chain_nopf.py` (`486356e4e552ebcd4ffca1ead3f808be7144b7c53438009bca4f5374cd554e52`). Their local structural round regressed the complete layer, so they are not production checks until a packed diagnostic finds a neutral stage pair.

## Send order

1. **`cand_r3_v9_attention_portfolio.py`** (SHA `257674016ee2b5b3ef12a11c2e2af90406d90e9c43efe0e08c5ce2d37373d8f2`). Send next. One judge run measures and selects the cost candidate's MoE/projection/PDL/prefix/island paths plus FA3's 27 `(num_splits, pack_gqa)` settings, XQA/WGMMA warp and grouped settings, FA3-structured WGMMA, FP8-PV, and the four-stage q_len=1 gather per `(batch, q_len, window)`; every invalid/resource-wrong/tolerance-failing variant is logged and rejected.
2. Do not resend `cand_r3_v9_cost_autotune.py`, `cand_r3_v9_autotune.py`, `cand_r3_v9_dispatch.py`, or `cand_r3_v9.py`; their scored runs already occupy the queue and provide the incumbent H200 evidence.

The cost-autotune and attention-portfolio logs are intentionally part of the scored receipt: `ATTN_PORT_TUNE`, `ATTN_PORT_SHAPE`, `ATTN_PORT_REJECT`, `ATTN_PORT_PLAN`, `PROJ_TUNE`, `PDL_TUNE`, `MOE_TUNE`, `PREFIX_TUNE`, `PERSIST_TUNE`, and reject lines record every accepted timing and selected plan. The local screen uses sm_120; FA3, XQA/WGMMA, and cuBLASLt H200 choices remain unverified until the next scored run.

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
