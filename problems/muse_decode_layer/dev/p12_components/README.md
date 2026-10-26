# P12: current S15 paths on the component boards

Purpose: get current H200 per-shape receipts against the component leaders. These boards do not determine our decode-layer score. Base is exact S15 `60254883`; `build.py` preserves its source and the NF MIT notice, then appends each board ABI. Candidates are <1 MB and generated, not separate forks of S15.

- MoE: S15 NT1 flags at M1/8/16/32; its heavy/light/NT1 three-stream paths at M128/256; its native NT1 for required nearby sizes. EP8 keeps32 experts from the board's256-expert input, losslessly. Board BF16 output matches the layer expert ABI.
- Attention: S15 last-CTA decode1 sliding, direct-partial decode1 full, attn_main at decode16/32, layer FA3 at verify1/sliding, and register-fold N128 VQ8 at full verify16/32. Required DP-head tests use S15's dynamic-head partial-input kernel. Scored TP8 policies are unchanged.
- Projection: S15's full-QKV, O and unpool paths, including its selected NF rows; pooler uses the fused pool/router producer's coded/balanced recipe at standalone N4096, router uses that balanced kernel on its BF16 hi+lo bank. Board outputs require materializing/reducing/casting. The layer's qkv3/gate concurrency and pool/router fusion cannot be represented by separate board calls: these receipts do not directly measure those fusion savings.

`screen.py` runs tools/local_eval.py in the judge image on GPU0, normal and forced, three fresh inputs per scored shape. `static_cache.py` reuses the pinned S15 admission only for identical CUDA text AND compiler flags; Python checks always run, and a changed CUDA source compiles. Attention stand-ins retain the original launch/allocation glue. Projection forced mode traverses native Lt setup and VDSK's Python wrapper with a local-only MMA stand-in; MoE already runs the selected Python stream/route/finalizer paths with its compiled sm120 arithmetic branch. No stand-in is shipped.

`report.py` requires every test, graph shape and static check to pass on the exact queued SHA. `admission.json` retains counts, hashes and branch calls. Generated detailed results live in screen/; local timing is not H200 evidence. A transient GPU0 OOM from concurrent lane screens is rerun sequentially, never admitted.

Queue policy for this task: scored, priority4, not-before epoch1792972800 = 2026-10-26T00:00:00+00:00 (09:00 KST), behind existing reset labs. No candidate depends on the reset outcome: they measure frozen S15 paths. `compare.py` matches exact source SHA through the ledger, reads actual benchmark records, ranks aggregate time ratios to each component leader, and preserves our per-shape receipt rows. Board snapshots do not publish leader per-shape times; do not infer them. Rank actual layer value with stage fractions and the fusion/protocol differences above after receipts arrive.

Final queued component snapshots: MoE **4127d934**, sender **939577**; attention **f38b9ae3**, sender **939623**; projections **01f80a78**, sender **939656**. They were queued with epoch1792972800 / priority4; current receipts are recorded below and in `leader_gap.json`. `queue.json` records exact full hashes; all normal and forced screens are pinned in `admission.json`.


Reset audit: the MoE receipt is complete; attention and projections still await receipts. The current `okc leaderboard <problem> --json` snapshots are `leader_moe.json`, `leader_attention.json`, and `leader_projection.json`. They contain aggregate value/speedup only; their rows do not publish per-shape leader times. The documented job endpoint is owner-only (`upstream/llms.txt:78`). Consequently the comparison reports our actual per-shape times and aggregate gaps; it does not fabricate leader shape times. Component receipts have the official paired SGLang controls and confidence interval, with no independent A:A arm. They are diagnostics for stage priority, not paired whole-layer promotions.

`watch.py` waits on receipt-directory events, starts `tools/h200_lab.py read <job>` as each of the five P10/P12 jobs is accepted, and stops after all five receipts land. It never sends or promotes code. `leader_gap.json` records the current exact-SHA receipt state.

## Receipt results

Leader per-shape times are not published. Ratios below compare aggregate component times; the official confidence interval covers our SGLang speedup, and no independent A:A was captured.

| Stage | Our speedup | 95% interval | Time / leader | Tests / shapes | Job |
|---|---:|---|---:|---|---|
| moe | 2.02924 | [1.9988011252708926, 2.059678316903508] | 1.25835 | 5 / 6 | `97f5e7f3c3fc4159` |

MoE per-shape us at T1/8/16/32/128/256: **30.878 /72.944 /105.207 /138.927 /207.993 /268.911**. All five required tests and six timed-shape checks passed. This completes the MoE diagnostic; it does not propose a new layer kernel. Attention and projections still await receipts before the stage ranking is complete.


Projection receipt `d6ad694a27954c80` passed9tests/30shapes and scored **0.98059x** [0.96588,0.99530], an aggregate **1.65590x** time gap to the component leader. It includes a measured board-adapter sum/cast/router-add chain, unlike fused in-layer consumers. This is a valid board result, not a direct inference of that layer stage's gap. `projection_reduce.py.inc` uses one Triton output reducer (including router hi+lo), preserving every S15 producer. Normal+forced9/30/25 and the exact30shape8seed paired lab pass locally. Follow-up sender **1147249**, MoE priority4.5, payload `917e4ec6`, wrapper `4502b8bb` (481537 bytes), on compact judged host `ac9dd461`. The oversized default-host waiter1130147 was cancelled before acceptance; it consumed no judge job. The follow-up separates A/A2 old adapter, reduced adapter B and producer-only P, each with FP64/probe checks. Promote the new component adapter only if it beats its A:A envelope; layer integration remains separate.

Attention's scored-only waiter939623 was held to tomorrow after scored quota filled while checks remained. Replaced with exact-source check sender776198, accepted **4e108beb9a0a4cdb**. This is the same admitted f38b9ae3 code and completes the diagnostic without waiting for scored capacity.
