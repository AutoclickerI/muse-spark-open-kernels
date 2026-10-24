# Prefill send order

## New ready: glue portfolio (separate from expert/attention lanes)

`cand_glue_prefill.py` — SHA256 **2a2e3914ea26756fc2df828f8c58cc88d0157171dcff880dec50c55290ebdbef**.
Full board-best layer with receipt-proven controls; new residual/norm, pooler-norm and
router-top8 GEMM epilogues plus cheaper vector/topk choices. No existing candidate edited.
Screen sm1205/5 tests +8/8 fresh graph shapes,48.111s prepare;78 trials/36 finalists reached.
Independent FP32 gates; any optional Exception rejects;230s trial budget; compact table last.
Direct60 configs, forced all-fusion layer5/5, strict Hopper branch/fault tests pass;
60 SM90 specializations compile. H200 speed remains unverified. No judge send/commit here.
[Design and evidence](glue/README.md); [H200 glue inventory](../../../docs/serving/prefill_glue.md).


Serves: G2, G1. Only the orchestrator sends or commits. This thread launched no judge run.
`send` uses a scored slot while one remains, then a check, then waits for quota reset.

## Ready for the 09:00 KST reset (20 prefill checks then)

At reset, the orchestrator sends every ready source in this order; each source is frozen by its SHA256. A scored slot is used first while one remains, then checks.

| Order | Source | Frozen SHA256 | Local screen | New information |
|---:|---|---|---|---|
| 1 | `cand_fused_expert_portfolio_routing.py` | `4fea734c0b00c006a55b1086ebe3dc240c6e6245a465c94eaa26e5a5ae676b98` | `runs/screen/prefill_routing_round8/muse_prefill_layer.cand_fused_expert_portfolio_routing.json` | 45 expert choices selected with layer-router-derived local counts; persistent/fused/direct/raw controls retained. |
| 2 | `cand_dense_fusion_portfolio_routing.py` | `82fc7bb30cbfbd9a87c9c6d28bdb63cdf1747e1adba3ddddbab60a5d9d5a7349` | `runs/screen/prefill_routing_round8/muse_prefill_layer.cand_dense_fusion_portfolio_routing.json` | Structural dense/router front/back choices plus the same 45 routing-aware expert choices and six attention paths. |
| 3 | `cand_score_portfolio_routing.py` | `d22c87c983983a4575e3c0196c9e4d6d3039308f4f6a2fdf100cfbad19aad186` | `runs/screen/prefill_routing_round8/muse_prefill_layer.cand_score_portfolio_routing.json` | Complete 32-choice expert, five dense/router and six attention portfolio with realistic routing selection. |

## New local candidate (not queued)

| Source | SHA256 | Local screen | Change |
|---|---|---|---|
| `cand_firstturn_overlap_grid4.py` | `a97aab65efeedad63f5d8c82aa47a0f85bb87f0d841dabb0136de1a2d1c283f5` | `runs/screen/muse_prefill_layer.cand_firstturn_overlap_grid4.json` | Adds grid-4 persistent grouped expert and overlapped W2 expansion choices; 5/5 official tests pass, prepare 8.2 s on sm120. |
| `cand_expert_tma_cluster.py` | `00d91b14b74f899639b9978dde63b9d42bbca967051f98c0f231ea5bf90b784c` | `runs/screen/muse_prefill_layer.cand_expert_tma_cluster.full.json` | Adds a raw17 TMA/WGMMA expert choice with a 2-CTA cluster-ready grid and fused W13 SwiGLU; 5/5 tests pass. |

Suggested note: “Routing-aware prepare tuner: fresh BF16 hidden rows use the prepared layer router hi+lo and bias, select the busiest 32-expert range, construct local slots/counts, compare the legacy uniform fixture for diagnostics, and dispatch the realistic winner. Cold-L2 paired CUDA-graph timing and independent semantic/public-floor checks run for every choice. Graph-build, oracle, timing and finalist failures emit SCORE_REJECT and are dropped. All choices remain lossless and BF16-legal; no H200 result is inferred from sm_120.”

The table records the frozen send snapshots; `NEXT_CHECKS.sha256` checks the current files after
the receipt-safe `SCORE_LAST_SUMMARY` addition.

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

## Historical queue folded at `3bdbe49^`

The withdrawn pre-routing portfolio sends and their local screens are retained in the parent
commit for provenance. Do not resend those deleted sources; the live queue is the three routing-aware
portfolios in the first table above.

Current H200 best: v11 check6a9cd943fe404d74=1.367883x; scored965a9a816d4744ea=1.366811x.
`submission.py` unchanged. No promotion handler; the board retains the best scored run.

## Attention port for the prefill lane

`cand_attn_prefill.py` / the serving mirror are a standalone attention block,
not a complete judge candidate. Frozen SHA256 `498c8c444a014a738326bd768782fe767a8fd73bdcbf235ac58d4086d6180cca`.
Interface: FP8 `q[T,16,64]`, paged FP8 `k/v[pages,page,2,64]`, int32 page table,
`cache_seqlens`, `cu_seqlens_q`, scales, softmax scale and window; returns BF16
`[T,16,64]`. It gathers pages losslessly to contiguous storage, reuses each
KV row across the GQA group, and selects `fp8_gather128/256/64` against FA3
with CUDA-graph timing and FP32-bound rejection. Serving dispatch is restricted
to uniform q_len >=2048 FP8 prefill and falls back to FA3 otherwise.

```sh
sha256sum -c problems/muse_prefill_layer/dev/NEXT_CHECKS.sha256
```
