# Matched S19 small-M ladder

P3a is **active**. Final receipt **5ee35b941ccb4171** assigns the residual in one exact S19 cohort.
[REPORT.md](REPORT.md) contains the complete us/CI ladder, calibrated contrasts, work/finish observations,
score ceilings and build decision. `read.py` audits the receipt; `report.py` reproduces report/ceilings.

Frozen expanded S19 is `62ea5bf4836277ceffb344e4792c92dd6d1e5980ea315fcbf96ae037b5ebc01a`.
Original BASE is unchanged in each translation unit: rvc_nfidx0/95 M1,rvc_nbase1/69 M8,
rvc_nbase2/69 M16/32. Same build,264CTAs×256threads,depth2,shared reservation and seeds.
All60 runtime resource rows match exact sm90a offline counts;every arm keeps two-CTA capacity,
zero spills. Local replay alone clamps the common grid to Blackwell residency so BASE cannot deadlock.

Eight seeds×four symmetric palindromic rounds follow P4 ec754e5:all-round arithmetic means,
equal mean exposure position,no selected medians or selective retiming. Weights1375353+7919*s,
routes1379000+M+97*s. All2,048 round times survive at nearest1ns,encoded losslessly as32ns tick
deltas plus remainders. All96NVML snapshots retain clocks,temperature,power and monotonic time.
The four most variable BASE/AA/BSINK contexts retain all host-enqueue durations;none are excluded.
This exploratory stage cohort is not held-out whole-layer G0 confirmation.

| Arm | Work retained |
|---|---|
| BASE / AA | Exact original S19,independent workspace |
| L0 | Same TMA schedule/depth and all raw shared code/scale reads,checksum sink;no activation/decode/MMA/epilogue/readiness |
| L0S | Raw transport with common volatile sinks,calibrating sink tax |
| L1 | L0S + all exact decode/scale products;live sinks prevent deletion |
| L2 | L1 + real activation transport +32HMMA sites;no epilogue/readiness/finalizer |
| L3 | L2 + original epilogue/publication/finalizer,ready/grid spins removed |
| BSINK | Exact S19 + common sinks;measures normal sink tax |
| MATH | Synthetic on-chip raw operands and register activations,decode/MMA/epilogue;no global raw-weight copies;includes initialization/small norm/GS/output accesses |
| STREAM | Pure two-slot TMA of exact touched experts' unique packed W13/W2 bytes;same grid/shared budget;checksum and simple tile pointer loop |
| META | Route/metadata construction,lists live |
| LOOP | META + outer tile schedule/partial-store barriers,no raw/activation/math |
| ACT | LOOP + real activation transport and live operand sinks once per segment |
| ROUTER | Separate actual router/input producer |
| HIST | Minimal BASE observer:CTA entry/route/phase/end,SM id,tiles and expert mask |
| EARLY | Correct slot-release build described below;M1 is a control alias |

The ladder telescopes per seed with both raw and normal sink calibrations explicit. Adjacent cuts are
conditional schedule contrasts,not independent hardware-pipe latencies. ACT−LOOP includes operand
sinks. MATH is an on-chip raw-operand control,not a universal compute floor. No cut nominates to P4.

Every seed/M keeps joint per-CTA (tiles,experts) histograms and exact entry/W2-end/finish order
statistics,including last CTA. All8,448 observations survive. A fixed extra seed0 phase in the same
run retains per-SM work/last finish and joint work/finish bins at2.048us. HIST/BASE tax is measured;
tails overlap the ladder terms. `results.json` preserves all arrays and telemetry.

Largest added terms:epilogue M1 4.09us;MMA+activation M8/16/32 8.79/12.10/19.71us (modeled
combined six-load R1.04014). M32 L0−STREAM is15.70us (R1.01234 alone). BASE−STREAM across four M
gives modeled R1.11685,one third R1.03532. These are full-wall-credit sizing ceilings,not measured G0.

EARLY moves slot release/next TMA before the last decode/MMA group and partial-store barrier,
after explicit volatile last raw reads and warp sync. It keeps eight math warps,two slots,two CTAs,
128registers,zero stack/spills and store-fence-publish-acquire ordering. 135native TASK gates,
normal+forced integration screens (13TASK/12shapes each) and remote before/after controls pass.
Measured EARLY/S19 M8/16/32 is1.01450/1.01998/1.02053;every CI is above1. No new winner goes to P4.
`early/manifest.json` pins exact patch/candidate. EARLY_PRIVATE is an unmeasured offline prototype.

The next ranked bound is a pack-free/coalesced BF16 activation-operand layout oracle. ACT−LOOP's
M8–32 reference ceiling isR1.02126,but includes sink cost;a correct timed packing design needs an
oracle first. At full reference credit,common added packing cost must be<3.57us for>1% modeled R.
Reserve at most8MiB per expert/token-block operand bank,16MiB for both phases;this is allocation,
not an assumption of padded traffic. Keep two CTAs,precision and all packing inside timed producers.
M32 raw-load scheduling remains another material opening. The lane is not parked.

Two earlier local capture/grid failures were fixed before sending. c43111f584854b05 is unmeasured
because schema/output transport failed. da1fb2eeb0d3443d has valid controls but its scalar STREAM and
per-tile ACT sink were costlier implementations,not the requested floor/reference;`initial_*` retains
them. Final5ee uses corrected STREAM/ACT plus LOOP and EARLY in one new matched cohort.
Final exact local packets7480/2143B fit declarations8600/2800;all checksummed remote packets verify.
The lab-only host prepare stop is expected;correctness-bearing native controls all pass.

Reproduce:build.py,early/build.py + early/ship.py,count.py/early/count.py,build_lab.py,
run_local.py and early/check.py/screen.py. Do not regenerate payload identities before reading a receipt:
cells.json/payloads.json pin the measured source. `read.py <receipt>` then `report.py` performs no GPU work.
