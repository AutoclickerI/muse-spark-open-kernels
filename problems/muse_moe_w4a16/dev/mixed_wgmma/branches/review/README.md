# Reset receipt review

`python3 .../review/report.py` reads exact embedded payload hashes, so automatic batching and
host rerouting retain identity. `results.json` is the current receipt/control/model comparison.
`watch.py` uses one filesystem completion subscription, reads landed lines through h200_lab,
and exits after the four original batches plus round5 finish; there is no polling timer.
Concurrent incomplete metadata writes are reported and retried on their close event.

Stage3 **8792099636654bac** is valid:16 seeds x3/M, all TASK and48/48 bit gates pass. A:A
signed/absolute/max difference+0.043/0.519/3.120us, S14 controls182.90/256.67us. DIRECT cluster1,
ordinary, serial and deep-serial ratios are1.2039/1.1427,1.1965/1.1381,1.2585/1.2992 and
1.1976/1.2273. Ordinary is almost identical to cluster1, and reported cluster capacity matches
the3/2 block-perSM limit. This closes the tested topology fix; it does not close all reuse.

The prior DIRECT calibration1.1970/1.1322 agrees within0.6%; heavy-phase sums remain within2%.
`stage3_calibration.json` replaces those phase anchors with the measured values. Do not infer
a causal decode/epilogue/launch/overlap partition from isolated phase service and a mean over
different inputs. The original round2-4 model files remain intact to score their forecasts.
`report.py` compares both raw us and control-normalized predictions, separating clock/input
drift from recipe-model error. A failed or incomplete lab is unmeasured, never a performance loss.

Decision applied: no stage3 arm to P4. Deeper serial is4.9/5.5% faster than serial but still
loses, so CTA2S3 needs its queued direct comparison. Round2 unchanged67eca44e was auto-batched
with P3a as **de47736b9f588673**, now job **dd539517bc8344dd**; do not send a duplicate.
Unchanged round3 abc1631d and round4 80c5d1c8 are packed in **4ca41bdc7a25cce7**, attention
import, total150s/priority2. Only own unsent senders were stopped under the dispatch lock,
before accepting any original job. The identical prefill base lost all LAB_BEGIN in a05f9863;
the packed attention run avoids that loss and spends one slot for both batches. Local shared
deadline rehearsal passes22/24seeds in106.4s; no native compile-time inference.

New response **f0c92b1867e9aa50**, job **7ea5e070b8f744ea** ([round5](../round5/README.md)):
N32 raw-share vs decode-share, paired with both S14 and CTA2,24seed entry, all local gates pass.
It attacks issue/decoded-smem cost at lower accumulator width. Native results pending.

S15 timeline cross-check `7eaa262a30ec483e`: representative MoE owner177.53/236.12us versus
combined unique roof122.90/132.06us, GPU idle5.45/5.86us. Other MoE work covers28.14/105.56us
of the late NT1 dependency gaps. This is observed overlap, not a causal branch assignment;
do not label those busy intervals as launch overhead or sum per-branch roofs. These data
constrain the residual model, but use a different input distribution from the fixed fixture.
