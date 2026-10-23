# Prefill send order

Serves: G2, G1. Only the orchestrator sends or commits. This thread launched no judge run.
`send` uses a scored slot while one remains, then a check, then waits for quota reset.

## Ready for the 09:00 KST reset (20 prefill checks then)

At reset, the orchestrator sends every ready source in this order; each source is frozen by its SHA256. A scored slot is used first while one remains, then checks.

| Order | Source | Frozen SHA256 | Local screen | New information |
|---:|---|---|---|---|
| 1 | `cand_fused_expert_portfolio_routing.py` | `4fea734c0b00c006a55b1086ebe3dc240c6e6245a465c94eaa26e5a5ae676b98` | `runs/screen/prefill_routing_round8/muse_prefill_layer.cand_fused_expert_portfolio_routing.json` | 45 expert choices selected with layer-router-derived local counts; persistent/fused/direct/raw controls retained. |
| 2 | `cand_dense_fusion_portfolio_routing.py` | `82fc7bb30cbfbd9a87c9c6d28bdb63cdf1747e1adba3ddddbab60a5d9d5a7349` | `runs/screen/prefill_routing_round8/muse_prefill_layer.cand_dense_fusion_portfolio_routing.json` | Structural dense/router front/back choices plus the same 45 routing-aware expert choices and six attention paths. |
| 3 | `cand_score_portfolio_routing.py` | `d22c87c983983a4575e3c0196c9e4d6d3039308f4f6a2fdf100cfbad19aad186` | `runs/screen/prefill_routing_round8/muse_prefill_layer.cand_score_portfolio_routing.json` | Complete 32-choice expert, five dense/router and six attention portfolio with realistic routing selection. |

Suggested note: “Routing-aware prepare tuner: fresh BF16 hidden rows use the prepared layer router hi+lo and bias, select the busiest 32-expert range, construct local slots/counts, compare the legacy uniform fixture for diagnostics, and dispatch the realistic winner. Cold-L2 paired CUDA-graph timing and independent semantic/public-floor checks run for every choice. Graph-build, oracle, timing and finalist failures emit SCORE_REJECT and are dropped. All choices remain lossless and BF16-legal; no H200 result is inferred from sm_120.”

## Exact-source local admission (sm120, not H200)

- Official5/5 and timed graph8/8; baseline FA3 cannot run locally, so no local SGLang speedup.
- Every new front/boundary/back schedule at753/3012/16384:162 independent fresh checks, plus
 100 whole-layer combination replays and16 nearby-shape replays (1,7,8,9,63,64,65,257).
- Four mixed-output TMA pipelines:40 fresh replay checks at1/139/753/3012/16384; native TMA,
 MMA.sync stand-in for Hopper WGMMA. Cold local extension build35.22s, outside timed calls.
- Forced Hopper constructors, stage3/TMA and portable FA3 glue: all15 dense shapes,45 expert
 choices at3 sizes,10 front/5 boundary/7 back choices at3 sizes and6 attention paths at8 shapes
 measured. Fully selected complete layer:8 shapes ×2 fresh replays pass; warm prepare39.34s.
- Injected graph construction, independent-oracle, replay timing and finalist construction
 failures all emit SCORE_REJECT, drop the variant and continue. The known valid choice survives.
- All25 new Triton SM90 specializations compile;4 native TMA/WGMMA kernels compile sm90a with
 58/90/90/154 registers and zero stack/spills. H200 execution/cold prepare remain unverified.
- Compact complete timing catalog is6989bytes and survives the judge's final16K log tail.
- Evidence: `dev/round7_validation.json`, `dense_structure_validation.json`,
 `dense_tma_validation.json`, `dense_selected_validation.json`, `tuner_rejection_validation.json`,
 `dense_fusion_sm90.json`, `compile_dense_tma_sm90.log`.

Local front timings (synthetic, cold-L2, untuned cuBLAS controls): merged NN217.1/792.6us versus
separate235.6/843.8us at753/3012 tokens. At16K, separate4374.6us beats merged NN4871.7us;
TMA and fused routing also lose locally. Stream-K/boundary and chunk paths are measured jointly
with experts on H200 rather than selected from these emulated/local results. No H200 gain inferred.

Builders preserve the coordinator's rejection fix through `builder_safety.py`; explicit --output
is required by the older builders to keep their queued output immutable. One complete new file
provides every new measurement; separate diagnostic submissions would duplicate that information.

## Already queued; do not resend

The coordinator already owns these sends. Their hashes are recovery context, not new requests.
The fused source hash5241415 includes the coordinator's guards; older admission evidence records
its pre-edit c18fa82 bytes. Do not replace the queued source with a builder's new output.

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

Current H200 best: v11 check6a9cd943fe404d74=1.367883x; scored965a9a816d4744ea=1.366811x.
`submission.py` unchanged. No promotion handler; the board retains the best scored run.

```sh
sha256sum -c problems/muse_prefill_layer/dev/NEXT_CHECKS.sha256
```
