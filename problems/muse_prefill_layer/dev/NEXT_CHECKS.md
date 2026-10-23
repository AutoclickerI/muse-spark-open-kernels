# Prefill send order

Serves: G2, G1. The orchestrator owns all sends and commits. `send` uses a scored slot when one
remains, otherwise a check, otherwise waits for the 09:00 KST reset. The board retains its best;
there is no check-then-promote handler. This thread launched no judge run or collector.

## Ready, new source

| Order | Source | SHA256 | Local screen | Purpose |
|---:|---|---|---|---|
| 1 | `cand_score_portfolio.py` | `fa202b0272f441d917938481fdecd976278858dbaadabaef089a2f0cf42a0452` | `runs/screen/prefill_round5/muse_prefill_layer.cand_score_portfolio.json` | Complete score portfolio:32 expert choices and multiple lossless banks;15 dense/router shapes with cuBLAS NT/NN, all Lt heuristics, DeepGEMM and persistent Triton;6 attention engines at all8 scored shapes. Synthetic prepare tuning on the judge chooses by speed. |

Suggested send note: “Score-only portfolio: retain raw/canonical/WGMMA/bitplane/Humming/Marlin
banks; cold-L2 fresh-input graph tuning with paired swapped order and measured replay-cost subtraction.
Tune experts at753/3012/16384, all5 dense/router projections and all8 attention shapes. FP32 router,
BF16 activations, FP8 KV and every key retained. Official local5/5 and graph8/8; every expert/dense/
attention family passes forced Hopper stand-ins. No memory-saving preference; prepare budget guarded.”

This single source measures all new components; separate dense-only or expert-only copies would
repeat its H200 measurements. Keep the original queued experiments unchanged so their receipts remain
comparable. Do not send a source hash again if it already has an admission request or receipt.

## Already queued by the orchestrator

These17 sources were sent before this round. Their listed order is the previous recovery order,
**not17 requests to create again**. The server owns admission/quota; check `runs/waiting/`,
`runs/pending/` and the ledger before recovering a missing request. A landed receipt owns its result.
20 prefill checks reset at09:00 KST; `send` handles whether a request uses a scored slot or a check.

| Previous order | Source | SHA256 | Local screen | Experiment |
|---:|---|---|---|---|
| 1 | `cand_expert_portfolio_v2.py` | `b1538a10e562f7f5ba7043300eb787be14159f0f5fb1ec5a1259416a71b14f98` | `runs/screen/prefill_round4/muse_prefill_layer.cand_expert_portfolio_v2.json` | Primary diagnostic: 18 same-bank expert choices, including low-register BN64, BN256 and direct canonical DG; v11 attention/dense control. |
| 2 | `cand_prefill_portfolio_v2.py` | `4ab8bccc3ddde270f499d67974be79c3aa191f66cda71d14732ba07d87919ea4` | `runs/screen/prefill_round4/muse_prefill_layer.cand_prefill_portfolio_v2.json` | Complete serving-compatible stage portfolio: same expert choices plus FA3/direct/gathered FP8 long-context attention. |
| 3 | `cand_canonical_hybrid.py` | `6d10a6804cfa3964c2bdeb91e6708fd8c27f910d8e7342e6f91389199876dfc4` | `runs/screen/prefill_round4/muse_prefill_layer.cand_canonical_hybrid.json` | Fixed v11-style canonical control: BN128 pipeline below8192, direct-bank DG hint above; checks the fixed hybrid family with row hints and BR4 SwiGLU. |
| 4 | `cand_prefill_autotune.py` | `71593f9c5e5467f616ea26d3f8dc7778e3f277106216b62df4dc193797680d9a` | `runs/screen/20261023T145219Z/muse_prefill_layer.cand_prefill_autotune.json` | Earlier two-expert/four-attention complete selector; narrower control for the new portfolio. |
| 5 | `cand_canonical_autotune.py` | `9b21cf17386d46de2da7f606c2c024bb26461935dd4bf37a0ec253b0b0e10a14` | `runs/screen/20261023T143731Z/muse_prefill_layer.cand_canonical_autotune.json` | Earlier canonical pipeline versus RS128 expert selector; isolates the earlier control. |
| 6 | `cand_attention_fp8_contig.py` | `e4a9792806182e36765e2ebd9289aac2b12938e26bf6ddc194eaeea68e828e8b` | `runs/screen/20261023T145219Z/muse_prefill_layer.cand_attention_fp8_contig.json` | FP8 gathered BN128 attention with v3 experts. |
| 7 | `cand_attention_fp8.py` | `f379b940f6a92caa59367d9fe6460060cce2d91efea2d1b0ffe20a7a9b897e0a` | `runs/screen/20261023T145219Z/muse_prefill_layer.cand_attention_fp8.json` | FP8 direct paged BN128 attention with v3 experts. |
| 8 | `cand_grouped_attention.py` | `24252850d0c3db5b27c979e7f9234aa62d0bd861ce36ced1cdb32f2ba5faaef8` | `runs/screen/20261023T143731Z/muse_prefill_layer.cand_grouped_attention.json` | Fixed RS128 adaptive plus gathered FP8 BN128, single canonical bank. |
| 9 | `cand_canonical_grouped.py` | `17206b59f6548cbd999da5cc786aa6ca96066f1c017db67c6730e7eb59b7c07f` | `runs/screen/20261023T143731Z/muse_prefill_layer.cand_canonical_grouped.json` | Fixed RS128 adaptive canonical expert reader, prefill frontend metadata reused. |
| 10 | `cand_canonical_prefill.py` | `c4d47509fe7a2a4b172e85167cbf8e80471d73a18f322a58561ebe3ac4cea1f4` | `runs/screen/20261023T143731Z/muse_prefill_layer.cand_canonical_prefill.json` | Fixed BN128 v3-derived canonical pipeline. |
| 11 | `cand_attention_fp8_contig256.py` | `4b13c714512e3113b35e3ca084669f9c5e7af4e0c726dcde41f8ecc301e35d7a` | `runs/screen/20261023T145219Z/muse_prefill_layer.cand_attention_fp8_contig256.json` | Gathered FP8 BN256 diagnostic; Hopper stack424B/thread. |
| 12 | `cand_canonical_attention.py` | `463b8f678b05dbbcb34ff0169b74e168e35b416c4c91b65cf18549249cf745d5` | `runs/screen/20261023T145219Z/muse_prefill_layer.cand_canonical_attention.json` | Fixed canonical pipeline plus gathered FP8 BN128. |
| 13 | `cand_v17.py` | `624e3fbac3a8761296029d72445d26c4751fb82f0c6b424d721c4e09338daa2e` | `runs/screen/20261023T145219Z/muse_prefill_layer.cand_v17.json` | Bit-plane expert decoder; original prefill bank. Useful mechanism evidence before any further port. |
| 14 | `cand_v16.py` | `134dc86ad528fd002b70314c84c5615a955549d52002c95ac994f2137d9d072d` | `runs/screen/20261023T143731Z/muse_prefill_layer.cand_v16.json` | v11 expert dispatch with expected-row hint; original raw/custom research state. |
| 15 | `cand_v15.py` | `ec53c5566437143ac5fc7a27226ffa6afb2d0603b10f0d37c445d1535c4ad68d` | `runs/screen/20261023T143731Z/muse_prefill_layer.cand_v15.json` | Shared/shared WGMMA expert pipeline; original prefill bank. |
| 16 | `cand_v14.py` | `aeeff090a520043b4685eeeb750397e00b8e8efdeb37e8bb013f034e9668de63` | `runs/screen/20261023T143731Z/muse_prefill_layer.cand_v14.json` | Hopper TMA indexed Humming expert path; distinct research bank. |
| 17 | `cand_combo_ready.py` | `9198aed89bf254e48fb1b49059cecaad123f1cbe9a2d92fa5935917f2f489ec6` | `runs/screen/20261023T143731Z/muse_prefill_layer.cand_combo_ready.json` | Earlier receipt-assembled stage control; may retain multiple research banks, not the serving state. |

## Validation and interpretation

- Exact source passes the five official local tests and all eight timed graph shapes on sm120/GPU0.
- Forced Hopper prepare exercises both constructor ABI and TMA flag construction using portable
  MMA/Humming/FA3 stand-ins. All32 experts at all3 shapes and6 attention paths at all8 shapes
  were measured; dense tuning covers15 projection/shape pairs. Local selection is not an H200 result.
- Fresh graph/reference validation:128 complete expert-layer cases,552 dense/router cases and12
  complete production-attention cases; all five official tests also pass with the selected Hopper glue.
- `score_portfolio_validation.json` ties counts/prepare time to the source SHA; `score_tuning_local.json`
  contains the complete local diagnostic table. The combined tuned selection passes all8 scored
  shapes with two fresh inputs each (`score_selected_validation.json`),16 further full-layer passes.
  The forced complete sweeps take49.18/61.12s with warm local caches; cold H200 timing is unverified.
- Complete C++ expert/Lt source compiles for SM90a; all15 persistent Triton specializations compile
  for SM90. H200 execution and cold-compile prepare time remain unverified until the receipt.
- Tuning retains no synthetic input, output, converted BF16 expert weights or graph between calls.
  NVFP4 remains NVFP4 at rest; only per-shape choices, lossless layouts and library workspace persist.
- FINDINGS owns KV capacity arithmetic and the cached-Humming-config mutation failure. Board
  sources have no bank-count preference. Serving duplication requires measured throughput and actual
  resident-token allocations, not a canonical-bank compatibility gate.

Current H200 control remains v11: check6a9cd943fe404d74=1.367883x,
scored965a9a816d4744ea=1.366811x. `submission.py` is unchanged. No local timing is an H200 claim.

Hash verification from the repository root:

```sh
sha256sum -c problems/muse_prefill_layer/dev/NEXT_CHECKS.sha256
```
