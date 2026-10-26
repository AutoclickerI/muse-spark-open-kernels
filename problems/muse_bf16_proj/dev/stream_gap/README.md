# P10 projection leader gap

The 1.36125 time ratio is a component-board aggregate (`08cc94050a4e4564`), not a measured ratio between the two teams inside a layer. The leader publishes neither its per-shape times nor its implementation. This report identifies our costs and tests transferable changes; it cannot attribute an exact algorithm to that team.

`2015fc90abf94e14` completes all 30 projection shapes, eight independent activation seeds and A:A. It compares exact P12/S15 control A, a generic exact 12-bit decoder D storing BF16 directly, identical generic FP32 output plus cast F, and the current producer P without the component adapter. M1 in this first batch uses one-hot timing; its random-input confirmation remains a limit.

## Bytes and stream rate

The checksum-verified cold NC stream has a fitted marginal rate of **4.614 TB/s** and **2.876 us** intercept over 32–128 MiB; maximum end-to-end rate is 4.203 TB/s. Compare with the earlier judge stream 4.58 TB/s and TASK peak 4.8 TB/s. This is a size regression, not a confidence interval. Receipt: `2015fc90abf94e14`.

B/W is the complete one-pass logical weight representation including tables/escapes/padding. It is **not a hardware DRAM counter**: repeated CTA requests, cache reuse, transactions, activation loads and output/partial writes can change physical traffic. The rates below divide that bank size by the measured current producer time P, so they are useful-throughput rates. The CSV/JSON retain board A, D, cast costs, all intervals, graph node counts and stream-fit residuals.

| M | Role | A B/W | D B/W | P us | Useful TB/s | /4.58 | /4.8 | A / D launches |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | qkv_gate | 1.5045 | 1.5083 | 10.224 | 2.777 | 60.6% | 57.9% | 2 / 1 |
| 1 | o | 1.5664 | 1.5664 | 5.792 | 2.269 | 49.5% | 47.3% | 2 / 1 |
| 1 | pooler | 1.5045 | 1.5083 | 15.888 | 3.177 | 69.4% | 66.2% | 2 / 1 |
| 1 | unpooler | 2.0000 | 1.5166 | 4.240 | 1.978 | 43.2% | 41.2% | 2 / 1 |
| 1 | router | 4.0000 | 4.0000 | 5.600 | 1.498 | 32.7% | 31.2% | 2 / 1 |
| 8 | qkv_gate | 2.0000 | 1.5083 | 11.408 | 3.309 | 72.2% | 68.9% | 2 / 1 |
| 8 | o | 1.5664 | 1.5664 | 5.488 | 2.394 | 52.3% | 49.9% | 2 / 1 |
| 8 | pooler | 2.0000 | 1.5083 | 19.056 | 3.522 | 76.9% | 73.4% | 2 / 1 |
| 8 | unpooler | 2.0000 | 1.5166 | 4.344 | 1.931 | 42.2% | 40.2% | 2 / 1 |
| 8 | router | 4.0000 | 4.0000 | 5.664 | 1.481 | 32.3% | 30.9% | 2 / 1 |
| 16 | qkv_gate | 2.0000 | 1.5083 | 11.552 | 3.268 | 71.3% | 68.1% | 2 / 1 |
| 16 | o | 1.5332 | 1.5664 | 5.840 | 2.202 | 48.1% | 45.9% | 2 / 1 |
| 16 | pooler | 2.0000 | 1.5083 | 19.112 | 3.511 | 76.7% | 73.2% | 2 / 1 |
| 16 | unpooler | 1.5083 | 1.5166 | 13.976 | 3.621 | 79.1% | 75.4% | 2 / 1 |
| 16 | router | 4.0000 | 4.0000 | 5.872 | 1.429 | 31.2% | 29.8% | 2 / 1 |
| 32 | qkv_gate | 1.5166 | 1.5083 | 12.488 | 2.292 | 50.0% | 47.8% | 2 / 1 |
| 32 | o | 2.0000 | 1.5664 | 6.128 | 2.738 | 59.8% | 57.0% | 1 / 1 |
| 32 | pooler | 2.0000 | 1.5083 | 20.040 | 3.349 | 73.1% | 69.8% | 2 / 1 |
| 32 | unpooler | 1.5161 | 1.5166 | 16.000 | 3.180 | 69.4% | 66.2% | 2 / 1 |
| 32 | router | 4.0000 | 4.0000 | 6.096 | 1.376 | 30.0% | 28.7% | 2 / 1 |
| 128 | qkv_gate | 2.0000 | 2.0000 | 13.552 | 2.785 | 60.8% | 58.0% | 2 / 2 |
| 128 | o | 2.0000 | 2.0000 | 7.472 | 2.245 | 49.0% | 46.8% | 1 / 1 |
| 128 | pooler | 2.0000 | 2.0000 | 25.784 | 2.603 | 56.8% | 54.2% | 1 / 1 |
| 128 | unpooler | 2.0000 | 2.0000 | 18.960 | 3.539 | 77.3% | 73.7% | 1 / 1 |
| 128 | router | 4.0000 | 4.0000 | 9.424 | 0.890 | 19.4% | 18.5% | 3 / 1 |
| 256 | qkv_gate | 2.0000 | 2.0000 | 19.400 | 1.946 | 42.5% | 40.5% | 1 / 1 |
| 256 | o | 2.0000 | 2.0000 | 9.136 | 1.836 | 40.1% | 38.3% | 1 / 1 |
| 256 | pooler | 2.0000 | 2.0000 | 30.528 | 2.198 | 48.0% | 45.8% | 1 / 1 |
| 256 | unpooler | 2.0000 | 2.0000 | 24.256 | 2.767 | 60.4% | 57.6% | 1 / 1 |
| 256 | router | 4.0000 | 4.0000 | 9.968 | 0.842 | 18.4% | 17.5% | 3 / 1 |

The control already codes QKV M1/M32, O M1/M8/M16 and unpool M16/M32. Its coded totals are roughly 1.50–1.57 B/W; router remains two BF16 banks, 4 B/original FP32 weight. Raw remaining BF16 sites are 2 B/W. A blanket extra 25% reduction therefore overstates the available saving.

## Fixed work and output boundaries

Empty graphs with 0/1/2/4/8/16 no-op nodes take 0:3.008, 1:3.952, 2:4.640, 4:6.256, 8:8.000, 16:12.128 us. The 2–16-node fit is **0.519 us/additional launch**; the first node adds 0.944 us. The 2.876-us stream intercept is consequently not all removable launch time. Receipt: `2015fc90abf94e14`.

Most small component shapes have a producer plus a reduction/cast node; the large router has three nodes. Generic direct BF16 output and the exact hi/lo router reduce those to one. F minus D isolates 1.22–1.79 us for the generic output boundary. Current A minus P is about 1.2–2.1 us at split/cast sites (independent captures, not an additive critical-path measurement). These conversion wins explain part of a component-board gap; the real layer generally passes FP32 partials directly to a fused consumer.

D/A wins at pool M1/8/16/32 are .91118/.75085/.76407/.80414; router wins at M1/8/16/32/128/256 are .59450/.64688/.68029/.74258/.63111/.74565, with per-cell CIs in decomposition.json. O1/8 direct-store wins are .77826/.81828, while the identical generic FP32+cast recipes are .96295/1.01834: much of this apparent gain is a board-output boundary. Generic coded QKV loses at every tested small M (1.039–1.154); unpool loses (1.060–1.249).

The optimistic same-receipt per-shape component selection is 1.10711x. Its joint samples were not retained, so no composition CI exists. It is selection data, not a G0 result.

## Fusion and large-M reuse

`646822ab3d164be8` uses 16 independent random activation/residual seeds, eight cold alternating rounds and A:A, retaining every joint sample. O1 last-CTA fusion combines the projection, residual and two norms in one launch. The current NF O1 + cluster boundary costs 8.102 us; WR4/2/1 fusion costs 12.984/15.971/23.872 us. Ratios [95% paired-log t15 intervals]: **1.602651 [1.588091,1.617346]**, **1.971324 [1.954076,1.988723]**, **2.946596 [2.927237,2.966083]**. A:A is .986591 [.976728,.996554]. This serial finalizer loses despite saving a launch.

The same receipt tests 20 full-M-tile/split-K alternatives at M128/M256 against each current projection with the same route/norm or residual/norm neighbour. Every arm loses: 1.102–1.701 at M128 and 1.521–3.734 at M256. These recipes request each logical weight bank once across their disjoint K splits, but pay a larger accumulator/shared-memory footprint and changed grid. Native builds use 73.8–98.3 KB shared and 90–190 registers, with zero spills. Which resource causes the remaining slowdown is not isolated; footprint and occupancy are source evidence, not a causal timing decomposition. `followup_results.json` retains every interval and A:A sample.

The board pool N4096 and standalone router are not interchangeable with the layer fused pool+hi+lo N4608. P4 must retain fused routing, QKV3/gate concurrency, the FP32 partial ABI and PDL ordering. The already-measured small-M pool dispatch is in S17/S18; it is not a new nomination from this batch.

## Current build and P4

The new build tests exact 12-bit QKV at M8/M16, both full N2304 and split N1280. Six geometries vary WK4/8, S1/2/4 and one/two pipeline stages. It changes neither the decoder precision nor the consumer format. All 24 arms and 16 fresh activation seeds are batched on the projection host; samples include independent A:A and separate TASK one-hot checks. `qkv_admission.json` pins local normal/forced 9 TASK tests + 30 graph shapes, all 24 extra cells, 24 zero-spill sm90a units and 24 native host-dispatch stand-ins.

Read the accepted receipt with `read_structural.py <job>`, then regenerate this report. P4 handoff is `P4_HANDOFF.json`; no layer promotion is claimed without held-out paired six-load R and lower 95% bound >1.

Rebuild: build.py -> build_lab.py (first decomposition); build_o1.py / build_followup.py (structural recipes); build_qkv.py (new batch). The router port is retained with its full MIT notice, so rebuilding no longer depends on a previous agent temporary checkout. Generated candidates, wrappers and local outputs are ignored.

QKV dispatch receipt: `2b45988c88fb42a5`; outer host passed: True. Ratios are candidate/control time; point winners require P4 held-out confirmation.

| M | Role | Arm (WK,S,stages) | Ratio [95% CI] | A:A [95% CI] |
|---:|---|---|---|---|
| 8 | qkv3 | [4, 1, 2] | 1.964073 [1.944676,1.983663] | 0.984575 [0.975835,0.993394] |
| 8 | qkv3 | [8, 1, 2] | 1.181058 [1.175907,1.186232] | 0.984575 [0.975835,0.993394] |
| 8 | qkv3 | [4, 2, 1] | 1.289301 [1.278240,1.300458] | 0.984575 [0.975835,0.993394] |
| 8 | qkv3 | [4, 2, 2] | 1.249864 [1.241747,1.258034] | 0.984575 [0.975835,0.993394] |
| 8 | qkv3 | [4, 4, 1] | 0.896534 [0.888875,0.904259] | 0.984575 [0.975835,0.993394] |
| 8 | qkv3 | [4, 4, 2] | 0.898269 [0.893551,0.903011] | 0.984575 [0.975835,0.993394] |
| 8 | qkv_gate | [4, 1, 2] | 1.731481 [1.718754,1.744302] | 1.000580 [0.991067,1.010184] |
| 8 | qkv_gate | [8, 1, 2] | 1.117849 [1.111932,1.123797] | 1.000580 [0.991067,1.010184] |
| 8 | qkv_gate | [4, 2, 1] | 1.176473 [1.167331,1.185687] | 1.000580 [0.991067,1.010184] |
| 8 | qkv_gate | [4, 2, 2] | 1.146200 [1.139541,1.152899] | 1.000580 [0.991067,1.010184] |
| 8 | qkv_gate | [4, 4, 1] | 0.905404 [0.897972,0.912897] | 1.000580 [0.991067,1.010184] |
| 8 | qkv_gate | [4, 4, 2] | 0.941740 [0.932115,0.951464] | 1.000580 [0.991067,1.010184] |
| 16 | qkv3 | [4, 1, 2] | 2.074076 [2.051709,2.096688] | 0.996334 [0.987218,1.005535] |
| 16 | qkv3 | [8, 1, 2] | 1.211386 [1.201747,1.221103] | 0.996334 [0.987218,1.005535] |
| 16 | qkv3 | [4, 2, 1] | 1.306028 [1.292787,1.319404] | 0.996334 [0.987218,1.005535] |
| 16 | qkv3 | [4, 2, 2] | 1.296019 [1.287053,1.305047] | 0.996334 [0.987218,1.005535] |
| 16 | qkv3 | [4, 4, 1] | 0.917619 [0.910580,0.924713] | 0.996334 [0.987218,1.005535] |
| 16 | qkv3 | [4, 4, 2] | 0.929133 [0.919951,0.938406] | 0.996334 [0.987218,1.005535] |
| 16 | qkv_gate | [4, 1, 2] | 1.739733 [1.725710,1.753869] | 1.005548 [0.997452,1.013709] |
| 16 | qkv_gate | [8, 1, 2] | 1.144422 [1.137159,1.151732] | 1.005548 [0.997452,1.013709] |
| 16 | qkv_gate | [4, 2, 1] | 1.194829 [1.187565,1.202138] | 1.005548 [0.997452,1.013709] |
| 16 | qkv_gate | [4, 2, 2] | 1.163281 [1.154839,1.171785] | 1.005548 [0.997452,1.013709] |
| 16 | qkv_gate | [4, 4, 1] | 0.931355 [0.922068,0.940737] | 1.005548 [0.997452,1.013709] |
| 16 | qkv_gate | [4, 4, 2] | 0.978316 [0.968563,0.988168] | 1.005548 [0.997452,1.013709] |

P4 receives arm4 (WK4/S4/ST1) at M8 full QKV, M16 full QKV and M16 QKV3. The M8 QKV3 win is retained as a diagnostic because that site is inactive in the current layer. patch_qkv.py prunes untimed prepare to those three instances/two banks. Applied to shipped S18 e04635ec it produces f9e4661d; layer_qkv_admission.json records 13 TASK tests and 12 graph shapes in normal and forced local modes, executing every changed site, plus three zero-spill Hopper builds. P4 G0 remains unmeasured; the additional H200 prepare cost on that base is also unmeasured.

## P5 S19 producer-finish sizing handoff

P5's [sizing record](../../../muse_attention_decode/dev/producer_finish/README.md)
and [machine handoff](../../../muse_attention_decode/dev/producer_finish/P4_HANDOFF.json)
pin exact S19 expanded62ea5bf4, prefill import check116dd8976c844b54
(sender1933097), eight fresh seeds and all192 layer+96 ready-output TASK passes.

The component useful-throughput figures above do not establish dynamic ALU
slack. A global last-CTA finish follows all producer partial writes, so it
cannot hide under that preceding weight stream. Exact d1/v1/d32 coded SASS
shows16-column output ownership, four tiles per64-channel normalized head;
K-cluster reduction alone does not complete a head. d16 and v16 emit10/3
FP32 K partials; v32 is cuBLASLt BF16, not this editable CUDA epilogue.

No production producer patch is nominated: the proposed free global tail
fails dependency sizing. The reusable native `producer_finish/finish.cu`
probe completes actual FP32/BF16 producer output, RMS/RoPE/E4M3 Q and current
KV in a separate timed launch; its paid sliding tails are4.124/8.017/4.209/
3.745/5.147/2.761us at d1/d16/d32/v1/v16/v32. These are not lower bounds
for register-resident fusion. All paired CIs, controls and source hashes
are in the handoff. QKV3/gate concurrency must survive a future producer edit.

The prior RAW2 ready-BF16 deletion estimates do not time actual S19 FA3
(v16) or FP32 reduction. Deleting those four supplied costs with zero added
work models+0.563% six-load for sliding, +0.760% if blindly extended to all
62 layers, d1/v1 unchanged. Neither is a six-load upper bound. A future
joint design needs head-local readiness while other heads still stream,
plus a real ready-consumer oracle across all six S/F paths before sizing
it above1%. P5 proposes parking independent builds pending that concrete
mechanism; this does not close P10 producer fusion generally.
