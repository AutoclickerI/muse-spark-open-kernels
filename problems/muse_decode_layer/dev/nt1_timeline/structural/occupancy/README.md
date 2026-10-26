# Three full raw slots with two CTAs

The earlier48dc8b04 depth3/4 recipes lost while occupancy fell2→1. The loads-only360810b1 oracle also
removes math/readiness/finalizer;it motivates this test but does not isolate a depth bottleneck. This batch
holds predicted occupancy at2 while exposing the cost of reclaiming scratch. The parent is c49 PAIR M8,
PAIR+BLOCK M16/M32. M1 remains S15. All weight/scale precision and token/expert contributions remain.

## Cheapest sizing

`count.py` compiles the exact sm90a images with ptxas and invokes CUDA13's `cuda_occupancy.h` locally.
Inputs:H200 sm90,256threads/block,65536registers/SM,233472shared bytes/SM,1024driver-reserved bytes/block,
232448opt-in per-block limit. Thus two CTAs get at most115712 useful shared bytes each,with128-byte
allocation granularity. The register budget is128/thread after allocation rounding. The header hash,
exact CUDA/cell hashes,zero stack/spills and all depth2/3/4 resource rows are in `counts.json`.
H200's runtime `epi_info` is the authoritative residency check in the receipt.

| Layout | Depth3 dynamic shared bytes,M8/M16/M32 | Registers,M8/M16/M32 | Predicted CTAs | Reduction storage |
|---|---|---|---|---|
| BASE depth2 |84176/84496/85136|125/125/125|2|2x4096B|
| Old BASE depth3 |121056/121376/122016|125/125/125|1|2x4096B|
| SINGLE3 |115488/115552/115680|125/121/121|2|1x4096B|
| HALF3 |114848/115104/115616|122/122/122|2|1x2048B|
| DOUBLE3 |115488/115552/115680|125/125/125|2|2x2048B|

Four full slots alone need147456B,already above the useful115712B budget before any scratch. All exact
four-slot rows predict1CTA and receive no H200 arm. Every fitting three-slot layout is included,plus its
two-slot counterpart and exact BASE/S15 controls. This separates the scratch-layout tax from deeper lookahead.

## Scratch lifetimes and ordering

SINGLE aliases routing's32running offsets,8x32warp counts and expert-byte list into the reduction region.
Those values die when deterministic token lists finish;an added CTA barrier ends their lifetime before
reduction writes. Persistent expert offsets become exact uint16 and reverse flat indices exact uint8;
M<=32 has at most256flat slots indexed0..255. These bounds include M9/M24 and all8local-expert tests.
The inverse `pos_of` table also occupies dead reduction memory:it is rebuilt from the reverse permutation
inside the timed kernel after math/norm completion,with barriers before initialize/scatter/read. No input
work moves into prepare. One reduction buffer retains the original8warp sum order but needs a barrier
after each epilogue before overwrite. Persistent Route shrinks1904→688B;reverse route is onlyM*8bytes.

HALF uses one2KiB buffer. Warps0–3 store partials;after a CTA barrier,warps4–7 add their matching partials;
a second barrier publishes them to the finalizer warps. The final4-way sum changes only legal FP32 addition
order. A final CTA barrier prevents overwrite. The expert-byte routing list aliases the reduction region,
while the original forward/reverse route tables and1904B Route remain persistent.

DOUBLE combines SINGLE's route lifetimes/reconstruction with two2KiB buffers andHALF's paired partials.
Double buffering avoids the per-tile end barrier;the existing next-tile barriers bound the producer lead.
It retains the barrier between the lower/upper four-warp writes and the publication barrier before reads.

The original W13 store-fence-publish/W2-acquire and final grid synchronization are untouched. Actual raw
copies are still3x36864B and every NVFP4 scale/decode/MMA is inside the timed call. Aliases never overlap
an active raw-copy destination or a live reduction read.

## Admission and verdict

All6selectors pass348 native TASK/current-input/capture checks,including real/empty/zero/all8 routes,
M8/9/16/24/32,local starts96/224 and changed captured inputs. Local sm120 cannot launch three full slots;
`local_depth.py` explicitly maps requested3→2 and is absent from candidates/payloads. Native code still
executes every new reduction/route/alias path. Each three-slot layout also gets normal/forced whole-layer
screens. The two-slot controls have identical CUDA and the same dispatch glue,covered by component checks.

`build_lab.py` packs BASE/S15 andall6arms,16fresh seeds at M8/16/32,624paired comparisons48A:A,TASK before/after,
read-flushed AB/BA. Weight seeds675353+7919*s,routes679000+M+97*s. Exact packed local lab passes all16seeds,
8071characters/44.3s. `manifest.json` pins full S15 patches;`cells.json` pins the exact6CUDA/cell bodies.
Accepted MoE check **cdd07707a6974cac**,sender3240664,wrapper5daf36f333fec1e5.
`queue.json` pins the exact admitted975131byte source. H200 verdict is complete below. Every valid per-shape point win goes to P4 with exact patch,ratio,SE,seeds and95%CI;
only P4's held-out six-load R lower confidence bound>1 promotes the whole layer.

## Measured H200 verdict

Receipt **cdd07707a6974cac** passes all16seeds/TASK/source gates,624pairs48A:A,worst total/row.002372/.002532.
`resource_validation.json`:all18actual runtime rows equal offline predictions;each three-slot layout really
has2residentCTAs andzero local bytes. Candidate/BASE time and95% Student-t15 CI:

| Arm | M8 | M16 | M32 |
|---|---|---|---|
|SINGLE2|1.030108 [1.023816,1.036401]|1.025072 [1.020376,1.029768]|1.033352 [1.029474,1.037230]|
|SINGLE3|1.058302 [1.050464,1.066139]|1.066341 [1.062101,1.070581]|1.057700 [1.054786,1.060615]|
|HALF2|1.046150 [1.039883,1.052417]|1.037873 [1.031985,1.043760]|1.042459 [1.039524,1.045395]|
|HALF3|1.067235 [1.061502,1.072969]|1.078475 [1.073110,1.083839]|1.071193 [1.069244,1.073142]|
|DOUBLE2|1.010738 [1.005133,1.016343]|1.008580 [1.004133,1.013027]|1.008283 [1.005613,1.010954]|
|DOUBLE3|1.026589 [1.018336,1.034841]|1.044325 [1.039645,1.049005]|1.035255 [1.032020,1.038491]|

A:A mean absolute .487/.275/.357us,max1.904/.848/1.120. All6arms lose directly. Every3slot arm loses to S15 too.
DOUBLE2/S15 alternatives .983936[.978023,.989850]/.981001[.972536,.989467]/.981782[.975246,.988319]
go to P4 with16seeds,SE .002774/.003972/.003067,exact `double2.patch`4cfcbab3. They lose to stronger BASE.
No3slot nominee andno additive composition claim;P4 confirms held-out six-loadR.

The same-layout BASE-normalized3slot/2slot contrasts all exceed1 (`depth_contrast.json`),but are not direct
depth-paired timings. This evidence rejects occupancy loss as a sufficient explanation for deeper-slot losses.
The oracle is not depth-only. These exact barrier/alias/reconstruction schedules close;L1/shared carveout,
producer issue timing andless serialized reduction remain open. A next structural depth path is timed
warp-cooperative L2 lookahead at2rawslots,avoiding an additional full shared slot. It needs an issue/order audit.
