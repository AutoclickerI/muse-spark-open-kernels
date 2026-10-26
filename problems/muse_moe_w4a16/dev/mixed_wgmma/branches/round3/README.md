# P3b compact expert-major tail

Target:132.6 /158.5us at M128 /256. No new H200 result or production promotion is claimed.
The calibration uses8ee003db's185.43/261.03us S14 mean and its fixed-fixture phases; contextual
182.4/256.4us S14 numbers are from the earlier N16 receipt, not an identical timing sample.

## Predicted times and remaining cost

`model.py/json` is the executable prediction. All times below are us, M128 /M256. The point estimates
retain the old model's measured-rate calibration. The design bounds assume full heavy/light overlap,
subject to the combined unique-weight floor at4TB/s. They are not measured H200 latencies.

| Arm | Point prediction | Full-overlap design bound | Point minus132.6/158.5 target |
|---|---:|---:|---:|
|S14 control (observed calibration)|185.4 /261.0|—|52.8 /102.5|
|CTA2|177.8 /248.7|146.0 /144.5|45.2 /90.2|
|CTA2S3 (queued round2)|177.8 /248.7|146.0 /144.5|45.2 /90.2|
|TAIL_STATIC|169.9 /222.7|146.0 /144.5|37.3 /64.2|
|TAIL_QUEUE|161.9 /196.6|146.0 /144.5|29.3 /38.1|
|TAIL_QUEUE_S3|161.9 /196.6|146.0 /144.5|29.3 /38.1|

S3 has no assumed timing advantage before a receipt. STATIC assumes compact scheduling/N16 fusion
removes one quarter of the remaining scheduling deficit; QUEUE assumes one half. These fractions are
falsifiable hypotheses, not measured recovery factors. The corresponding deficits are31.8/104.3us.
The largest modeled avoidable opportunity is M256's lost overlap; hence this schedule rebuild.

The component ledger below sums to CTA2's point prediction after subtracting overlap. It is an
allocation model, not independent causal measurements:

| Component | M128 us | M256 us | Basis / limitation |
|---|---:|---:|---|
|Unique packed weights at4TB/s|120.3|127.4|481.30/509.61MB,17/18 active experts on the fixed fixture|
|Heavy activation service|13.0|20.8|CTA2 scheduled125.83/276.82MB at3e8996f8's per-phase L2-inclusive stream rates|
|Light activation service|16.5|16.5|~24MB NT1 refresh plus50.33MB N16 reads, at4.5TB/s; borrowed/address-mix estimate|
|Heavy stage/decode/epilogue residual|29.0|47.0|Anchored heavy service minus weight and activation allocations|
|Light stage/decode/epilogue residual|21.5|20.0|Light service minus weight and activation allocations|
|Route, finish and unattributed tail|16.0|17.1|Minimum16us shell; M256 keeps the positive calibration residual|
|Heavy/light overlap credit|−38.5|0.0|Derived calibration credit; not a measured critical-path increment|
|Incremental launch-topology win|0 assumed|0 assumed|Launch and terminal-CTA tail are not separately isolated; included in shell/residual|
|**CTA2 prediction**|**177.8**|**248.7**| |

The kernel/epilogue residual combines instruction issue, dequant, fences, reductions, scheduling and
tail. Existing receipts cannot split these further. Launch topology is unmeasured until reset lab
f867a8b lands; zero denotes no assumed incremental win. The broader3e8996f8 mean unique weights were
467/558MB (~116.8/139.6us at4TB/s), a different route mix. Do not substitute them into this fixed-fixture ledger.

The old133.3us M256 full-overlap bound omitted the combined stream cap. Correcting that and the
17.1us shell yields144.5us; `round2/model.py/json` and its arm table now enforce it. Even the corrected
full-overlap bound leaves M12813.4us above target: its fixed light service is130.0us plus the16us shell.
M256's corrected bound is14.0us below target, but attaining it requires the modeled overlap.

## Built design

`build.py` **6057094f** generates `tail_payload.py` **abc1631d**, CUDA **a5ef94e6**, with five arms:
S14, exact CTA2, TAIL_STATIC, TAIL_QUEUE, TAIL_QUEUE_S3. Every new arm is paired directly against
both controls using separate AB/BA x8 timings,24 seeds x3 fixtures/M. The S14/CTA2 device source prefix
is unchanged (`resources.json`, CTA2 prefix **5aff055b**); all arms share one compiled extension.

The existing live route CTA also writes a compact, expert-major list of9..64-token experts. It preserves
sorted expert order and includes a final <=64-token chunk of larger nearby shapes. No extra route launch.
The two consumer WGs share each X slice and own adjacent output tiles, preserving CTA2's one read/decode
per expert weight tile. N16 experts join that compact schedule instead of occupying a third stream.
Experts<=8 stay on the original NT1 path; larger width classes keep CTA2's existing kernels.

STATIC assigns compact tiles by a strided grid. QUEUE claims one full expert/channel tile with one
atomic ticket, not one per K stage. Separate W13/W2 heads reset in the route on every replay. The compact
W13 kernel completes before compact W2 on the same stream; there is no readiness DAG, spin or grid barrier.
S3 changes only the compact copy ring. Two N16 launches and their fork/join edge disappear. Their X reads
also halve (~25.2MB less on this fixed fixture); this alone is only~5.6us service at the light-rate assumption.
The larger predicted gains therefore require the scheduling recovery stated above, not just launch removal.

This differs from ea794250's losing shared queue: that replaced NT1 with N8 WGMMA and grew activation
reads to~91/93MB. This queue preserves NT1 and tests work claiming separately from the same static compact
schedule. It also avoids the cluster barriers/gather of8ee003db. Neither prior loss is treated as a general
closure of queues or multicast. Here the choice targets the modeled lost overlap without adding another
activation transport path.

## Screen and proof

All local work used GPU0 in the judge image. `screen_{static,queue,queue_s3}.json`:
three adapters, each5 official tests +5 static checks pass under `tools/local_eval.py --tests-only`.
They force the new plan on sm120, replacing only WGMMA by matching mma.sync fragment math.

`check.py` / `validation.json`:1,235 checks,122 captured cases, sizes33/127/128/255/256/257, counts around
8/9/16/17/32/64/128/192/256, empty/hot/real routes, poisoned ACT/Y/SS/output and work-list/counters,
two replays/case, plus15 separately captured local_start64 checks. All arms bit-equal to S14 on122/122
cases. Queue membership and exact final ticket counts are checked after every replay. Worst L2/row:
0.004212/0.004782. `entry.json`: all five arms pass72 fixtures/M/arm on24 seeds, weights=None;
worst L2/row0.004132/0.005018. Local timing was not cold-flushed and is not H200 evidence.

`audit.py` / `resources.json`: O3 sm90a compile +cuobjdump; no stack/spills/local-memory SASS or
C7512/C7513/C7519/C7520 warning. Group wait1 retained. Queue W13/W2 uses139/137 registers at S2 and
127/133 at S3. Scratch is51,200/76,800B plus the queue ticket's1KB static allocation on Hopper.
These footprints permit an NT1 co-residency hypothesis; the lab must measure actual overlap.

`schedule_check.py/json`:1,355 coverage/progress checks across expert counts0..1024, empty/full lists,
MT32/64 and grids1/2/4/132/264. Each successful ticket names one tile; every CTA exits on its first
out-of-range claim. No occupancy-dependent progress assumption. Route completion precedes the stream fork;
W2 follows all compact W13 stores; each tile drains WGMMA and crosses a CTA barrier before scratch reuse.
The unchanged S2/S3 ring is covered by the parent's19,278 copy-order checks. This is a protocol model,
not an exhaustive hardware race test.

## Queued

Packed attention import sender **4ca41bdc7a25cce7**, PID **237268**, priority2, total budget150s,
with unchanged round4 after this payload. First gets75s; the second receives all remaining time.
No H200 job ID yet. Original prefill77150f5f and intermediate attentiona62f4dfb were stopped
only while unsent, under the dispatch lock. The identical prefill host lost all LAB_BEGIN
in completed `a05f986396d744e7`; attention avoids that output loss. Packing round3/4 uses one
judge slot, with no duplicate job. `review/packed_entry.json` rehearses the exact shared deadline:
round3 passes22seeds, round4 passes24,106.4s total locally (native compile time unmeasured).
Current metadata: `queue.json`; progress `runs/detached/4ca41bdc7a25cce7.log`.

Decision: compare every arm's paired whole-MoE ratio to both S14 and CTA2 at each M. STATIC vs QUEUE
isolates whether claiming repays its cost; S3 gets promoted only on measured gains. Require whole-layer
integration evidence before promotion. This batch is independent of both pending outcomes.

Rebuild: `python3 .../round3/build.py`; screen: `CUDA_VISIBLE_DEVICES=0 tools/img python tools/local_eval.py
.../round3/screen_<arm>.py --tests-only`; graph/metadata checks: `CUDA_VISIBLE_DEVICES=0 tools/img python
.../round3/check.py`. Retained sources are build/runtime/tail/check/audit/model/schedule_check and their
current JSON evidence. Generated CUDA, adapters and payload are ignored.
