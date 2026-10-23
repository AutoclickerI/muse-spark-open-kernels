# Prefill reset order

Serves: G2, G1. **2026-10-24 09:00 KST / 00:00 UTC: 20 checks available.** The orchestrator sends; this thread launched no check/submit/collect process. Current H200 control is v11: check `6a9cd943fe404d74`1.367883x, scored `965a9a816d4744ea`1.366811x. `submission.py` stays unchanged.

The first three sources are new this round. Remaining rows retain earlier locally screened experiments; several already have admission requests. Match exact source hashes against `runs/ledger.jsonl`, `runs/pending/` and `runs/waiting/` before sending another request. Existing requests are not new slots to duplicate. The old receipt handler owns its original v1-v17 promotion/integration only.

## Order and frozen sources

| Order | Source | SHA256 | Local screen | Purpose |
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

17 screened candidates are listed against the 20-check reset cap; the orchestrator can add new receipt-driven candidates as they become ready. The service enforces the quota; this document is the per-lane order, not server admission state.

## New candidate validation

- Every new exact source passes the five official tests and all eight timed graph shapes with `CUDA_VISIBLE_DEVICES=0 tools/img python tools/local_eval.py <file> --json runs/screen/prefill_round4/muse_prefill_layer.<stem>.json`. Local shared-GPU times are admission evidence, not H200 speedups.
- The expert portfolio forces SM90 module construction through a native stand-in, runs all 18 synthetic choices at753/3012/16384 and passes 72 fresh full-layer graph/reference checks across all choices. The fixed hybrid passes12 corresponding checks.
- The complete portfolio forces both expert and attention tuners, passes the five official tests through the Hopper/FA3 stand-in, and replays each of four attention paths plus the selected path on two fresh production-branch full-layer inputs.
- All 24 C++ SM90a specializations compile with zero spills. BN64 r1 uses80–92 registers/thread, BN256178–192. SwiGLU BR4 uses 255 registers at4warps and76 at8warps; both are measured contenders. Coalesced canonical dequant K1024/K2048 compiles with80/127registers and no stack; both W13/W2 branches compile.
- Canonical expansion preserves represented BF16 values in exhaustive finite signed E4M3/E2M1 fixtures and changed-count graph replay, and complete expert0 tensors match independent FP32 dequantization before GS/norm.
- Dequant uses timed BF16 scratch; one 905,969,664-byte canonical bank is retained. Sequential workspace aliases reduce modeled16K expert scratch including output8.89→5.60GB.

## Decision/evidence

`docs/FINDINGS.md` owns current prefill facts; `../FOCUS.md` is the short lane entry. H200 per-kernel rankings/floors are in `V11_PROFILE.md` / `v11_profile_ranking.json`. New source families are hypotheses until H200 receipts land. In particular, the old BN256 v12 check faulted; that is not a performance result and does not validate the new128-thread BN256 reader.

Evidence: `verify_expert_portfolio_v2.log`, `cand_expert_portfolio_v2_validation.json`, `verify_prefill_portfolio_v2.log`, `verify_canonical_hybrid_final.log`, `cand_canonical_hybrid_validation.json`, `compile_portfolio_sm90_final.log`, `coalesced_dequant_sm90.json`, `coalesced_dequant_benchmark_local.json`.

Hash verification from the repo root:

```sh
sha256sum -c problems/muse_prefill_layer/dev/NEXT_CHECKS.sha256
```
