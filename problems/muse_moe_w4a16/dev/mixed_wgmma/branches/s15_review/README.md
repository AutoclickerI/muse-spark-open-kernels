# P3b S15 review: point nominations, packets and block readiness

Base exact S15 **60254883**, NF CUDA **0a31e251** (identical to S14). Targets132.6/158.5us.
[ATTRIBUTION.md](ATTRIBUTION.md) separates whole-MoE completion, isolated branch clocks,
logical bytes and unmeasured wait cycles. It explains the failed forecasts before this batch.

## Current H200 nominations to P4

S15 check **7e38226389a848eb**, sender **97722cb0449771fd**, MoE host,24 fresh seeds x3/M.
All TASK gates pass, worst L2/row .004128/.004955. Controls176.552/241.940us. A:A on seed
means signed/absolute/max +.100/.253/1.653us atM128,−.098/.306/.874us atM256. These are
24 seed means, each averages3 fixtures; their max is not a max over all individual fixtures.

| Piece | M | Paired time/S15 | SE across seeds | Seeds | Decision |
|---|---:|---:|---:|---:|---|
|CHUNK4|128|.9974995|.00985397|24|Nominate point gain, even inside uncertainty.|
|CHUNK3|256|.9460363|.01304716|24|Nominate point gain.|
|CHUNK4|256|.9096806|.00785751|24|Nominate point gain; default point choice.|

R6 discovery b433947196f94ef4 had S4 .9791/.9083 andS3/M256 .9448; its original SE was not
retained. Fresh seeds reduce the M128 expected gain to.25%, with substantial uncertainty;
M256 remains~9%. This evidence changes the expected size, not the point-nomination rule.
Earlier qualified round2-5 means are all>1, so no older A:A-vetoed point winner changes verdict.
Native chunk admission remains2CTAs/SM; S15 remains4. Exact S15-applicable patches, cell and
candidate hashes, per-seed ratios andSE are in[P4_HANDOFF.json](P4_HANDOFF.json). All pieces
are handed toP4 without a per-load orA:A veto; only its held-out composed six-load R promotes.

Both minimal chunk patches pass normal and forced whole-layer screens:13TASK cases,12shapes,
26static checks including actual sm90a compilation and eager/captured changed glue. The R7
native component result confirms the extracted S15 tile, not just the earlier R6 image.
The fixed-width receipt archive retains every independent seed mean and A:A pair without
entropy-dependent compression;[read.py](read.py) recomputes the published mean/SE and nominees.

## New independent arms

- **PACK5:** S15 heavy Q is packed16-output/full-K tiles,32768 code bytes plus4096 scale bytes.
  Scale words are PERMW-rotated. Untimed inverse rotation exposes the original code bytes,
  compressed in P3a's exact256-scale packets; timed unpack recreates the original rotation.
  Original light/NT1 banks are unchanged. The heavy copy ring transfers80 compressed bytes
  instead of128 per half-packet, or128 for a raw packet; per-stage metadata stays in smem.
  P3a proof/format is read from `dev/nt1_timeline/review/README.md` and its two pack5 include
  files. No widened scale or decoded weight is prepared. Native static loop image grows
  188→309.75 instructions (includes alternative raw/compressed paths), integer116.25→213.5;
  registers96/95→103/104, no spills. This cost is measured statically, not presumed hidden.
- **READY:** heavy W2 currently waits for the entire W13 grid on one stream. There is no
  heavy `done[e]` counter to reindex. Add one completion counter per expert/32-token block;
  each W13 output tile fences all writers, joins the CTA, then adds1. W2 acquires its own
  block against128 contributions before reading ACT. Reset is timed before the route fork.
  Two streams overlap producer and consumer, with all joins retained before F4.
- **READY_SERIAL:** matched sequential W2-grid control, oneCTA perH200 SM. READY uses the
  same grid, so capacity and dependency effects can be read separately. One producer and
  consumer fit together on H200; with onlySM-count consumer CTAs, they cannot block every
  producer slot. On sm120 the matched control/READY grid isSM/2 because the pair cannot fit;
  this explicitly forced stand-in still executes all stores/fences/publications/acquires.
- **CHUNK3/4:** independent S15 rebases for the measured R6 point pieces, not combined with
  packets or readiness. All arms retain original NF light kernels and finish math.

Two-base decoding is dropped: P3a's actual37→406 integer expansion is not carried forward.
The new independent recipes have now landed:

| Arm | M128 ratio ±SE | M256 ratio ±SE | Decision |
|---|---|---|---|
|PACK5|1.168932 ±.006967|1.260287 ±.006467|Measured loss; exact compression survives but timed cost dominates.|
|READY_SERIAL|1.039774 ±.009193|1.155520 ±.013055|Measured capacity cost relative toS15.|
|READY|1.074708 ±.006290|1.074582 ±.005278|Measured net loss for this capped two-stream/block-publication form.|

All retain4CTA native resource admission and no spills. PACK5 does not lose occupancy; its
extra decode/metadata path is a supported cost hypothesis, not a separately timed sole cause.
AtM256 READY recovers some of the matched grid control's penalty, but remains7.46% slower
thanS15. The same-seed normalized READY/SERIAL contrast is inresults.json and is exploratory,
not a separately timed directABBA pair. This closes this capped readiness recipe, not readiness
without reducing consumer concurrency. P3a's PACK5 small-M losses are recorded inclaims.json.

All10 normal/forced whole-layer screens pass. Full-bank proof verifies805306368 code bytes
and100663296 scale bytes exactly,4.05816% smaller including metadata.684 graph checks across
57 cases make all changed outputs bit-equal toS15;9 additional complete raw-scale graphs pass.
Exact150s cold-read entry completes24seeds in106.4s,5722characters. Its original send note's
6849-character claim is a typo; entry.json andqueue.json preserve the actual5722.
The203933-byte source passed H200; native entry finishes55.9s, no lost or truncated lines.

Rebuild cells: `python3 .../s15_review/build.py`; batch: `build_lab.py`; local normal/forced:
`run_screens.py`; full-bank/captured tests:`check.py`; native count/resource audit:`audit.py`.
All commands using GPU run withCUDA_VISIBLE_DEVICES=0. The payload is import-time and synthesizes
weights; it pairs all arms with S15 and independent A:A in one host run, excluding prefill.

The timed packet decoder also passes7458 independent packed/raw groups (954624 scale bytes),
covering every u8 code, every5-bit delta and packed alignment. Native full PACK5 was compiled
on sm90a; this exhaustive raw/packed helper execution is local sm120, not an H200 raw-branch
claim. Source hashes and proof are inpacket_decoder_validation.json. Non-winning production
patch files are generated/ignored; only the two chunk patch handoffs are retained.
