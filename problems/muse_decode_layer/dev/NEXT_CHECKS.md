# Decode checks — next send order after 18a5f4e

`cand_r3_v9all.py` is first and `cand_r3_merge.py` is second at the next reset. Both megakernel sources are held: the original persistent qkv failed repeated M=32 agreement
checks, and the regular-qkv repair still rejected a persistent dense island at M=16. The attention v2 source is parked after its FA3 proxy gate rejected faster verify variants;
the attention lane owns vstream. The new ready glue diagnostic folds the O-boundary and
pooler/router prefix with PDL and gates it against FP32 references. The earlier overlap source is
parked after a verify correctness failure. Pure V9's completed receipt is
1.428x overall (rank 85); rank 50 needs about 1.54x. The H200 non-MoE ranking is now full
verify32 attention first, then verify16 full attention, decode32 full attention, projections,
and glue; the per-stage floor table is in `docs/FINDINGS.md`.

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

Full attention is the largest remaining non-MoE cost at verify32/decode32/verify16. Its full-shape
profile gaps are +251/+90/+133 us over the one-read floors. MoE remains fixed to the in-layer V9
winner until `moe_tune.py` lands routing-aware choices. The ready glue diagnostic leaves V9 and
attention intact, fuses the O-boundary/pooler-router prefix with PDL, and rejects any FP32,
resource, or timing failure back to regular V9.

## Frozen source set

| Source | SHA-256 | Local validation | Role |
|---|---|---:|---|
| `cand_r3_attention_v2.py` | `e5414acdac7ae3a56f860369b88d0866567db6230e4ebe89d758e8f13a17cc15` | 13/13 tests, 12/12 graphs; parked after its 3% FA3 proxy gate rejected faster verify variants | **parked; do not send** |
| `cand_r3_v9_glue.py` | `233f0cddb568592ee85a0d72cb29f3f433e8bfd6df2b4f9870fb2bd031b6e3f6` | **13/13 tests, 12/12 graphs; prepare 8.666 s on sm_120**; `GLUE_TUNE` FP32 gates pass, fused path rejected by local timing at T=1/8/16/32 and shared memory at 128/256 | **ready diagnostic after receipt merge**; PDL O→boundary→pooler/router fusion, existing PDL route/top-8/softmax |
| `cand_r3_merge_mega_fix.py` | `f7e1217b840a42a85404b7ab563e9c6805e4d724fdfe068e71e69af5376f9d16` | held: regular qkv removed qkv disagreement, but persistent dense island still rejected at M=16; no timed selection | **parked; do not send** |
| `cand_r3_merge_mega.py` | `98d7e6d6029e9337fbc94806ce99c5e739a97d2d884e1385952ba78714e91a2d` | held: repeated prepares rejected persistent qkv at M=32; one verify-8 sample changed IDs; PDL-off did not make qkv deterministic | **parked; do not send** |
| `cand_r3_merge.py` | `f23a267748791646d9afc4d447068158ecdb71fa74ae134dad0add62d6b5b054` | 13/13 tests, 12/12 graphs; prepare 25.011 s on sm_120; fused-prefix resource rejects logged | queued second; merged non-MoE portfolio with MoE fixed to V9 |
| `cand_r3_v9all.py` | `41c018efaefb57bacaf0f55451f6753e1f96a4a1f0ef678066c8ac2cad5ce4d5` | V9 fixed; queued first | queued first at reset |
| `cand_r3_v9_attention_portfolio.py` | `4be669353a75acd03276f806dfaf6a64a94a6c8f7cf23edcaa67da2212603386` | queued in `c14edac`; V9 fixed | already queued; attention FA3/XQA/WGMMA/FP8-PV/q1-gather portfolio |
| `cand_r3_v9_cost_v9.py` | `ed2912d638563b6b0e7e30953e2aa2dbc096d15ad6c2cc6b347b99ecd53f1b93` | V9 fixed; source ingredient | do not resend; cost-v9 fused prefix/island base |
| `cand_r3_proj.py` | `235434b35d1407cdb61ff79f291b9a962fa4a56369f60277fb666863f5b1174e` | projection lane source; untouched | do not resend; projection bodies ported into merge |
| `cand_r3_v9_autotune.py` | `099ee216693f890b1bc6b737de5d3441e5f210baa2cca0c8fbb0b4fd65ecbc6e` | sent in commit `8584a56`; scored/queued | already sent; original MoE/FA3/projection/PDL portfolio |
| `cand_r3_v9_dispatch.py` | `633c96ccffa40c6673c99b808c18b28685db5aafd4541667de68e1f0e6c41b84` | scored; receipt `85e67f5d6ef945c9`, 1.366737x | already sent; current H200 incumbent |
| `cand_r3_v9.py` | `ba990a86f6f7f4985345cfc89a3bc56e36730d974ba47642a344c5c77d84279f` | scored; receipt pending/queued | already sent; V9-only layer |
| `cand_round3_v7.py` | `a3ae91dd3e42bf7edca26824012272e33376c890f41da906929f5b8eedd73d61` | scored; receipt `579855768bd24be6`, 1.353614x | historical V7 control |
| `cand_moe7.py` | `eb659038a74c628794b585e4579901dfea16fcd23c50eddf7862b5d84b3dcc6c` | scored; receipt `d2bf1ced702948a4`, 1.355430x | historical control |

Structural sources remain frozen for later diagnostics: `cand_layer_sweep.py` (`08015a0aede234527989a100bb661461dbee843b459844003bb8de59c128d096`), `cand_epilogue.py` (`2c585251f289b140b50d4c0465ec50cf4996bdf8d870ce19c1cb903b1d320598`), `cand_chain6.py` (`6e88cfa39935592a081bb95461ee82610c17b0cd7689209c4ea4e07cc72caf11`), `cand_chain.py` (`0d545fb6fe330e24dbf99f8f21bbb6444a342d945776ab5708f64d447faac4b9`), `cand_chain_occ2.py` (`96cf5ecc7141c2aa2df95ff0724d3dcb79a1cc7a38b5565b0374edfa3e9b5581`), and `cand_chain_nopf.py` (`486356e4e552ebcd4ffca1ead3f808be7144b7c53438009bca4f5374cd554e52`). Their local structural round regressed the complete layer, so they are not production checks until a packed diagnostic finds a neutral stage pair.

## Send order

1. **`cand_r3_v9_glue.py`** (SHA `233f0cddb568592ee85a0d72cb29f3f433e8bfd6df2b4f9870fb2bd031b6e3f6`). Hold for the receipt merge: one H200 run can select the PDL glue prefix per shape against regular V9; every FP32 agreement, resource, or timing failure rejects the fused path.
2. Do not send `cand_r3_attention_v2.py` (parked after its proxy gate), either megakernel source (`cand_r3_merge_mega.py` or `cand_r3_merge_mega_fix.py`, both parked), `cand_r3_merge_ovl.py`, `cand_r3_v9all.py`, `cand_r3_merge.py`, `cand_r3_v9_attention_portfolio.py`, `cand_r3_v9_cost_v9.py`, `cand_r3_proj.py`, or the withdrawn synthetic-tuned originals again.

The merged, glue, and megakernel logs are intentionally part of the scored receipt: `ATTN_PORT_*`, `PROJ_TUNE`, `PROJ_REJECT`, `PDL_TUNE`, `MOE_PLAN`, `PREFIX_TUNE`, `PERSIST_TUNE`, `MERGE_BUDGET`, `MEGA_TUNE`, `MEGA_REJECT`, `GLUE_TUNE`/`GLUE_REJECT`, the final compact `MEGA_SUMMARY`, and the
attention candidate's final compact `ATTN_PORT_SUMMARY` record every accepted timing and selected plan. The local screen uses sm_120; H200 attention, projection, and megakernel choices remain unverified.

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
