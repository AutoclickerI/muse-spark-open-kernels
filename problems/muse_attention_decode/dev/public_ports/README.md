# Public attention ports behind S19

Authority R-20261026-1. Reviewed detached sources in separate new external
clones, before copying; no donor install/build/test/notebook executed. Exact
pins, source hashes, retained functions and omitted drivers: `review.json`.
Operatorstack MIT and Lee Apache-2.0 license texts are in `NOTICES.txt` and
must remain in every shipped source/patch.

The judge image has Triton3.8.0 Gluon/warp specialization/Hopper WGMMA and
CUDA13 nvcc. Operator full verify emits13 WGMMA instructions, full decode
uses `mma_v2` (no native WGMMA claim); Lee selected families compile sm90a
with52 tensor-core instructions,112/32 consumer/producer register transfer.
Local normal/forced each704TASK gates +16 host TASK tests pass. Lee's local
WGEMU math retains its launches, stage depth2/NL1; operator forced Gluon
host routing uses its reviewed portable math at the same split/output ABI.
This does not prove Hopper math: every native arm has all TASK gates inside
the payload. Transport has6 static+1 lab-only wrapper checks.

The input is S19's **real producer ABI** from116dd8976c844b54, not only
board BF16 qkv. Shapes1280 are16-head QKV without the1024 gate logits;
S19 applies those gates afterwards. Timed split sum/cast pads gate logits
with+inf (sigmoid1), preserving that ownership. FP32 split counts per cell
are1/10/7/3, width1280/2304; v32 uses BF16. No timed values cached or moved
into prepare. New native attention includes its Q/K norm, RoPE, KV store,
attention, output norm/gate and last-arriver LSE merge. Lee's uncovered
nearby/test cells retain the unchanged S19 dispatch; every12 scored cell
is covered. Operator covers all16 official tests.

`out/public_attention.py` batches A,A2,operator,Lee for all12 cells;
8 fresh seeds8671259+7919*i, 4 rotated forward/reverse cohorts. Each arm
appears in every timing exposure position exactly twice. Raw nanoseconds
for all8 rounds retained; no outlier removal. Every replay coldL2, fixed
empty-graph subtraction recorded, graph nonempty and poison/replay checks.
Local timing fixtures are shortened to257 only in the local runner;
the sent source uses ctx80000 and all native branches.

Decision: per cell nominate every correct point<1, preserve CI/SE/seeds and
A:A. Choose fastest point for the combined exact S19 patch; P4 confirms
held-out six-load whole-layer G0 and actual complete prepare cost. A loss
closes only that pinned implementation/config on this producer ABI.

Rebuild exact S19 namespace first with `CUDA_VISIBLE_DEVICES=0 tools/img python3
problems/muse_attention_decode/dev/full_attention/build_phases_base.py`,
then this directory’s `build.py`, `build_lab.py`; judge-image `screen.py` and
`screen.py --forced`; `analyze.py <receipt>`, then `build_candidate.py
--selection selection.json`. Exact source hashes in `out/manifest.json` and
`out/lab_manifest.json`. Native compile evidence is preserved separately.

Native receipt **26d5054ffce94231** (proj check; sender3771b9736fbd86cf):
all704native TASK checks pass, 260.06s lab/296s total, complete packet with
all3072cold-round times, no device errors. The lab-only transport intentionally
stops at host prepare; its outer `passed:false` is not a failed attention gate.
Every A:A CI covers1. A:A point range .989612–1.009940; decode1 intervals
are widest (up to±3.9%). No failed-control timing is treated as a loss.

Ratios are port/S19, lower better, mean paired-seed ratios with95% t7 CI.
No seed/round discarded. SE and all seeds are in `result.json`; raw rounds
in `receipt_data.json` and the checksummed receipt. O=operatorstack, L=Lee.


| Cell | O/S19 [95% CI] | L/S19 [95% CI] | Preferred |
|---|---:|---:|---|
| 1x1s | 1.262517[1.222107,1.302928] | 1.232249[1.194636,1.269863] | S19 |
| 1x1f | 1.028181[1.016495,1.039868] | 1.008982[0.999721,1.018243] | S19 |
| 16x1s | 0.996939[0.963572,1.030305] | 0.954256[0.919953,0.988558] | L |
| 16x1f | 0.991556[0.985235,0.997878] | 1.112085[1.103138,1.121032] | O |
| 32x1s | 0.956675[0.940380,0.972970] | 0.860893[0.845972,0.875814] | L |
| 32x1f | 1.026920[1.022488,1.031352] | 1.107322[1.100255,1.114388] | S19 |
| 1x8s | 0.780899[0.773075,0.788723] | 1.105711[1.089843,1.121578] | O |
| 1x8f | 1.015518[1.009180,1.021855] | 0.757719[0.752012,0.763426] | L |
| 16x8s | 0.779475[0.766451,0.792498] | 0.763592[0.752210,0.774975] | L |
| 16x8f | 1.138588[1.128975,1.148201] | 0.968115[0.961687,0.974543] | L |
| 32x8s | 0.799316[0.788602,0.810030] | 0.750479[0.742199,0.758760] | L |
| 32x8f | 1.377356[1.367078,1.387634] | 1.076995[1.073106,1.080885] | S19 |

`P4_HANDOFF.json` nominates all12 correct point<1 pieces (6O,6L), with
8preferred cells. `candidate.patch` **97c86143**, reconstructed source
**4a60c4e3**, applies exactly on S19 **8b415553**. Independent OP-positive
and Lee-positive patches84c9e409/299d9d8f are alternatives for P4 per-cell
composition, not runtime fallbacks. Full notices embedded in every source.
Final selected wrapper normal/forced:28cases each, eager+poison/repeated graph
checks (56TASK each),41static pass. Handoff lists actual identities and sources.

Launch counts S19/O/L with observed producer ABI: d1S/F1/2/2;
d16S1/2/2, d16F2/2/2; d32S1/2/2, d32F2/2/2;
v1S1/2/2, v1F5/2/2; v16S5/2/2, v16F2/2/2;
v32S1/1/1, v32F2/1/1. Both ports' own attention/merge is one launch;
FP32 sumcast/padding adds one, BF16/full-width input is a view. The native
one-launch prologues remain inside every timed graph, not cached.

Mechanism verdicts: OP's two-head block-diagonal decode wins modestly at
sliding32 but loses d1 after paid packing. OP keys-on-M full decode is only
.991556 at d16 and loses d32; no universal native-WGMMA inference there.
OP producer/consumer Gluon full verify loses1.139/1.377. Lee removes FA3's
five-launch structure at v1F/v16S (to2) and lowers v32S mainloop1->1;
its producer WG, early query/own-KV loads, tensor-core row sums and balanced
splits win those cells. Lee v16F wins3.19% while v32F loses7.70%: only this
pinned dispatch closes there, not the whole producer/consumer mechanism.

No six-load whole-layer G0 claim: chosen component points on these seeds
are attention-only improvements. P4 measures held-out composition including
prepare<=300s and PDL boundary effects. This patch bypasses old FA3/row-
merge interceptors on selected cells; keep external split gate ownership.
d16F OP is vsS19, so P4 compares it to its already nominated C3 as well.
The remaining d1 packing launch and full32 mainloops are open structural
work. Nothing further queued. P5's producer_finish README records the
winning v1S fused preprocessing and warns against duplicate fusion.
