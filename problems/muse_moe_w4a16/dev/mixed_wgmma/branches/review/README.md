# P3b reset receipt review

`report.py` matches the embedded payload name and SHA, including automatic merges and reroutes.
[results.json](results.json) contains the raw paired receipt rows and per-arm qualification;
[recalibrated.json](recalibrated.json) compares the original forecasts and current evidence.
`receipt_index.json` preserves exact accepted jobs without depending on ledger recency.
All five batches are terminal. No new arm is nominated to P4.

| Batch / receipt | Primary S14 paired ratios, M128 / M256 | A:A signed / absolute / max us | Decision |
|---|---|---|---|
|Stage3 `8792099636654bac`|cluster1 1.2039/1.1427; normal 1.1965/1.1381; serial 1.2585/1.2992; deep serial 1.1976/1.2273|+.043/.519/3.120|All measured losses; closes these topology policies.|
|Round2 `dd539517bc8344dd`|TRANS64 1.4350/1.5270; TRANS128 1.7416/1.7781|−.051/.428/2.448|TRANS measured losses. CTA2/S3 unmeasured: missing all-consumer shared-X retirement join.|
|Round3 `b93dbbadd1014984`|No valid pair: first native N16 gate fails L2 .359844, row 1.0|Unavailable|Unmeasured; missing native BF16 N16 dispatch and shared-X join repaired in R6.|
|Round4 same `b93dbbadd1014984`|barrier 2.0471/2.0381; pipe 2.0320/2.0248|−.039/.421/1.936|Measured losses for this 2-slot/3-decode-warp/two-consumer recipe.|
|Round5 `7ea5e070b8f744ea`|RS raw-share 1.3017/1.3472; SS decode-share 1.4114/1.4931|−.105/.498/3.408|Both measured losses; actual two-CTA occupancy does not rescue these fixed-N32 pair forms.|

Stage3 has16 seeds x3/M, round2 has22, round4/5 have24. Every qualified arm passes TASK;
max errors L2 .004134, row .004756. Round4/5's primary S14 controls are valid even though
secondary CTA2 references are unqualified. Neither a failed implementation nor a passing
numerical sample proves a missing asynchronous ownership contract.

## Model recalibration

Stage3 DIRECT ratios agree with the old 1.1970/1.1322 calibration within .6%; heavy isolated
service agrees within2%. Ordinary launch does not fix the loss; cluster1 already admits3/2
CTAs/SM. These anchors survive. Original round2-4 model files remain unchanged.

TRANS64 predicted260.2/381.1us, observed262.7/388.2. After normalizing to the current S14 control,
observed/predicted is1.020/1.043. This apparent agreement contains cancellation: fixed heavy
service including pack is182.69/303.23us versus152.73/249.68 predicted. Pack alone costs9.82/10.54us,
far above its .92/1.83us raw-byte floor. TRANS128 predicted201.0/281.7, observed318.8/451.3,
normalized error1.602/1.641. Its one-CTA occupancy and publication/padding/issue cost invalidate
linear byte scaling; the receipt does not isolate a single culprit.

CTA2's177.8/248.7 forecast cannot be fitted to unsafe source. Round3's assumed25/50% schedule
recovery has no native measurement and is withdrawn. Warp PIPE predicted187.4/193.9, observed
374.6/517.4: normalized error2.008/2.716. Its resource prediction of one CTA/SM was right;
hiding all heavy stage work under the weight stream was wrong. PIPE saves only2.83/3.32us of
raw means versus the gated arm, insufficient to repay the total loss. This is a net dependency
ablation, not an identified decode/epilogue partition. Round5 had no point forecast; its
predicted two-CTA occupancy is confirmed, but RS/SS still lose. Preserve these as empirical
priors for the exact recipes rather than transferring arbitrary correction factors to R6.

The S15 timeline cross-check `7eaa262a30ec483e` has MoE owner177.53/236.12us, combined unique
roof122.90/132.06us and GPU idle5.45/5.86us. Other MoE work covers28.14/105.56us of late NT1
dependency gaps. Different inputs and isolated service sums do not identify causal launch,
tail, decode or overlap components. A new latency point estimate is not supported by this data.

## Follow-up applied

[Round6](../round6/README.md), payload **bd2192e2**, repairs both implementation defects and
packs safe full-width CTA2S3/static/queue/queueS3 controls with CHUNK64 S3/S4 at two-CTA resource
admission. It removes full-width register pressure without RS token-padding inflation or
SS decoded-A publication. All six official forced screens and1729 graph/1380 protocol checks pass.
A source-coupled negative witness now rejects the old shared-X protocol; the earlier checker
mistakenly assumed a single retirement frontier. Native SASS must contain N16/32/64 BF16 MMA.
R5's local checker was also corrected to capture each plan immediately after creation so its
OP_ARM mode cannot silently change. The1036 checks pass again; its sent payload is unchanged.

R3/R4 were packed unchanged in sender4ca41bdc7a25cce7, avoiding the prefill base that displaced
lab output and spending one slot. R5 senderf0c92b1867e9aa50 was sent on the stage3 loss.
R6's first wrapperf8009c7cc24c5abe was rejected422 (>1MB), before any job. Its retry uses only
the H200-passed round2 MoE host prefix and a checked275130-byte wrapper; payload is unchanged.
See round6/queue.json for the actual retry sender and job. No accepted run was duplicated.
