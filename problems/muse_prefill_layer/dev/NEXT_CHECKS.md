# Prefill send order

Serves: G2, G1. Only the orchestrator sends or commits. `send` selects a scored slot while one
remains, then a check, then waits for the quota reset. This thread starts no judge process.

## Ready, new source

| Order | Source | SHA256 | Local screen | Purpose |
|---:|---|---|---|---|
| 1 | `cand_fused_expert_portfolio.py` | `c18fa82edad25daf91d621da88dfd9380eea482757c4333c1c633545e4c9d114` | `runs/screen/prefill_round6/muse_prefill_layer.cand_fused_expert_portfolio.json` |45 expert choices:13 new fused/parallel paths plus the queued portfolio's32 controls. All dense/router and6 attention paths retained. Prepare measures complete stages per shape with fresh paired cold-L2 graphs; chooses speed alone. |

Suggested note: “New expert algorithms: persistent BF16 grouped GEMM with SwiGLU fused in its
 epilogue; FP32 split-K reduction; direct NVFP4 grouped GEMM with packed BF16x2 decoding and exact
 power-of-two accumulator restoration; graph-captured W2 expansion overlapping gate/up GEMM, plus
 a fused-gate/DG-down overlap hybrid.45 expert choices,15 dense/router shapes and6 attention paths
 at8 shapes. Full local5/5 and graph8/8; every path forced through Hopper stand-ins; exact FP4/SF
 codec fixtures. No judge timing claimed locally.”

Expert excess over the modeled first-turn floor is343.325weighted ms, compared with173.870dense
 and150.071attention. Dequant172.280ms plus SwiGLU67.549ms motivates the new fusions and overlap.
 Profiles are single H200 traces; model floors infer routing. `V11_PROFILE.md` owns that comparison.

One file supplies all new measurements. A separate file for each new expert path would duplicate
 its diagnostic table. The queued32-choice source stays byte-identical; do not send it again.

## Validation

- sm120/GPU0 official5/5 and timed graph8/8, exact source hash above.
- Forced Hopper constructors, stage3 configuration and TMA/FA3 branches use portable stand-ins:
 180 fresh expert-layer,552 dense/router and12 production-attention checks pass. The complete
 selected pipeline additionally passes all8 scored shapes with2 fresh inputs each.
- Both forced prepare sweeps measure all45 experts at753/3012/16384, all15 dense/router pairs,
 and6 attention paths at all8 shapes. Warm local prepare43.50/26.54s; cold H200 timing unverified.
- The actual packed decoder preserves130048 represented BF16 weights exactly for each of2
 reconstruction forms, covering every FP4 byte and every signed finite E4M3 scale, including zeros
 and subnormals. Only packed FP4 payload/scales and lossless permutations persist in prepare.
- Cross-graph shared-event fork/join and empty/skewed routing replay pass for all3 overlap paths.
 The captured call contains both branches and joins before down GEMM; no data-derived state persists.
- New GEMM and split epilogue specializations compile for SM90; actual H200 WGMMA execution,
 cold-compile prepare time and serving agreement await receipts.
- Local tuner OOM came from uint8 torch.sum's8GiB cast temporary, not a layer failure. The new
 read-only1GiB flush uses small scratch outside timing events; the complete rerun passes.
- Evidence: `round6_validation.json`, `fused_portfolio_validation.json`, `fused_selected_validation.json`,
 `fused_decode_validation.json`, `fused_sm90.json`, `overlap_expert_math.json` and local benchmark JSONs.

## Already queued: retain hashes, no duplicate requests

The32-choice score portfolio below was queued after commit9112546. The17 older sources were queued
 before it. Their order below is recovery context, not a request to resend them. Match source hashes
 against receipts, `runs/pending/`, `runs/waiting/` and the ledger; a landed result owns its information.

| Previous order | Source | SHA256 | Local screen | Experiment |
|---:|---|---|---|---|
| 0 | `cand_score_portfolio.py` | `fa202b0272f441d917938481fdecd976278858dbaadabaef089a2f0cf42a0452` | `runs/screen/prefill_round5/muse_prefill_layer.cand_score_portfolio.json` | Complete score portfolio:32 expert choices and multiple lossless banks;15 dense/router shapes with cuBLAS NT/NN, all Lt heuristics, DeepGEMM and persistent Triton;6 attention engines at all8 scored shapes. Synthetic prepare tuning on the judge chooses by speed. |
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

Current recorded H200 control: v11 check6a9cd943fe404d74=1.367883x,
scored965a9a816d4744ea=1.366811x. `submission.py` stays unchanged. No new H200 receipt is
inferred from local timings. Bank count and scratch size are resources, not selection objectives.

```sh
sha256sum -c problems/muse_prefill_layer/dev/NEXT_CHECKS.sha256
```
