# P3b native repair and bounded-token reuse

Payload **bd2192e2**, CUDA **8c476193**, built by `build.py` onto S14 **f1c77ce6**'s nf cell.
No production patch or native performance claim. Target whole MoE132.6/158.5us at M128/256.

The original CTA2/S3 numerical gates passed, but WG0 could copy shared X after retiring only
its own WGMMA reader. WG1 had no joined retirement frontier. The old checker assumed that
join and therefore admitted an unproven protocol. The compact-tail native dispatcher also
omitted BF16 N16, which portable math had supported. These are implementation defects;
neither the unsafe raw timings nor round3's failed first fixture closes the proposed schedule.

## One payload, seven arms

| Arm | Mechanism and role |
|---|---|
|S14|Exact incumbent primary control.|
|SAFE_CTA2S3|Full-width two-output-WG reuse, joined X retirement; required repaired control.|
|SAFE_STATIC|Correct native N16 and retirement join, compact expert-major static tail.|
|SAFE_QUEUE|Same safe tail, one device ticket per expert/output tile.|
|SAFE_QUEUE_S3|Same queue with deeper small-class ring.|
|CHUNK64_S3_2CTA|Two output WGs share X, expert token rows in at-most64 chunks; 3-slot ring.|
|CHUNK64_S4_2CTA|Same bounded64-token ownership, 4-slot ring.|

All joins are CTA-wide and execute uniformly after both WGs' wait1 (or S1 wait0), before
copy reuse. Native N16 delegates to the established `mwbr::mma<16>` helper. Six native tail
specializations must contain BF16 N16/32/64 MMA; checking portable results alone is insufficient.

CHUNK64 bounds accumulator width and uses `__launch_bounds__(256,2)`. Two WGs own adjacent
output tiles, with no new activation pack, cluster/multicast, separate decoder, decoded-A
smem round trip or queue. This differs from the lost cluster and WS recipes in ownership and
publication cost. It differs from R5 by preserving S14's roundup32 token math rather than
executing two padded32 groups on every small expert. W13/W2 each use one kernel instead of
four width-class launches. A naturally ordered static expert loop is used, not a shared DAG.

On the fixed route fixture,64-token groups/32-token groups=4/5 and8/11. Thus scheduled packed
weight multiplicity1.25/1.571 becomes1.0/1.143 and decode iterations .8/.727 of S14. Valid X
requests are halved across the adjacent output WGs; padded instruction geometry is counted
separately. These are schedule counts, not DRAM counters. Native two-CTA admission restores
four consumer WGs, but does not establish a larger issue budget than S14. Source evidence
supports neither a fixed occupancy speedup nor a causal overlap credit.

## Model and decision

The reset receipts invalidate byte-proportional transport and fully hidden stage-residual
point models: see [review](../review/README.md). Original CTA2/S3 predicted177.8/248.7us, but
has no qualified native pair. Original queue169.9/222.7 static and161.9/196.6 dynamic points
have no qualified measurement and their25/50% scheduling-recovery assumptions are withdrawn.
A calibrated R6 point latency is **unknown**, not fitted from unsafe times.

An explicit optimistic sensitivity scales the whole isolated S14 heavy service .8/.727,
while retaining everything else:~168/220us from round2 controls and fixed phases. This mixes
isolated fixed inputs with multi-seed means, assumes proportional heavy-stage scaling and
unchanged overlap, and is not an identified prediction. Even it misses132.6/158.5. The actual
receipt must test whether shared-X savings, narrower accumulators and fewer launches repay
extra joins and token-chunk weight rereads. Do not treat a raw-byte floor as a forecast.

Decision rule: require native TASK gates for all tested shapes and completed paired controls
with A:A. If safeSTATIC/QUEUE beats safeCTA2S3, that is evidence for this repaired tail schedule;
otherwise retain only the measured non-winning conclusion for it. Compare S3/S4 chunks toS14
and safeCTA2S3 separately at each M. Nominate validated primary-S14 point improvements by shape to P4 with A:A and uncertainty;
P4 then decides on held-out whole-layer six-load evidence. No per-load significance veto applies. A failed gate/protocol is unmeasured.
No additional speculative batch waits on the old failed implementations.

## Local admission

GPU0, judge image:

- All six forced `tools/local_eval.py --tests-only` adapters pass5 official +5 static checks.
  `screen_*.json` pins their source hashes. Hopper math uses matching portable fragments locally.
- `validation.json`:1729 graph/queue/poison/shift checks across122 cases, sizes33/127/128/255/
  256/257,21 local_start64 checks. Bit-equal counts toS14 are122/122/122/121/122/122; the queueS3
  difference is within TASK tolerance. Worst L2/row .004212214/.004781695. Native S4 uses S3
  locally to fit sm120 smem; it is an explicitly forced stand-in, not native S4 execution.
- `protocol.json`:1380 source-coupled least-completion and row-padding checks. Negative witnesses
  reject the original missing both-consumer join for S1/S2/S3/S4; source occurrence assertions
  bind each repaired tile copy. This is a protocol argument, not a hardware race detector.
- `resources.json`: native sm90a S3/S4 chunks at128 registers and76800/102400 dynamic smem,
  allowing2 CTAs/SM. Up to8 bytes stack and4–16-byte metadata spill loads/stores occur only
  in outer expert/row bookkeeping; CFG checks show no LDL/STL in either N32/N64 K loop.
  Accepting these outer spills is a deliberate resource tradeoff, not suppressed evidence.
  All six tail specializations now contain native BF16 N16/32/64 instructions.
- `entry.json`: exact weights=None entry,1GiB cold-read callback,24 seeds x3=72 fixtures/M,
  all TASK gates pass in83.7s of150s locally. Cached local compilation and portable timing
  cannot predict native H200 compile time or throughput.

A shared GPU0 allocation caused an initial OOM; all exact adapters subsequently passed.
The final builder was re-run after adding reproducible host extraction; payload and CUDA
hashes are unchanged. No native inference is made from this local screen.

## Dispatch

Sender **50c82e7edb5e83aa**, PID **1372784**, host **muse_moe_w4a16**, import budget150s,
priority1, job **b433947196f94ef4**, accepted after preceding reset/follow-up runs. [queue.json](queue.json) pins exact
snapshot and payload. The first wrapperf8009c7cc24c5abe exceeded1MB and was rejected422 before
a job. The retry is275130 bytes and uses the exact previously H200-gated round2 host prefix,
SHA **84eb2932**, without embedding its previous lab wrapper. No accepted job duplicated.
The generated host file is ignored; builder reproduces it from the archiveddd539 receipt source.

Rebuild: `CUDA_VISIBLE_DEVICES=0 python3 .../round6/build.py`. Local checks: `check.py`,
`protocol_check.py`, `audit.py` and `entry_check.py`; official adapters use `tools/local_eval.py`.
Send only through h200_lab; the queue already contains this exact payload. Next: read its
native gates, actual resources and S14 pairs before changing the production nf cell.
