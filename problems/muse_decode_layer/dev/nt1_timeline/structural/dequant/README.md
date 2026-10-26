# Exact byte-pair lookup decoder

This is the structural dequant step after depth/phase receipt48dc8b04902b40b1. The parent is PAIR M8,
PAIR+BLOCK M16/M32. M1 retains S15. The oracle360810b1 trivial decoder saves7–17% but changes math;
this candidate retains all original NVFP4 codes/scales and BF16 output math.

The old shared1KiB LUT lost in dd539517bc8344dd;random code pairs contend for shared banks and need
address work. This version replicates all256 possible byte-pair values64ways in a fixed64KiB shared table,
initialized inside every timed kernel. Each lane reads its bank:one PRMT forms the address,one LDS returns
the two BF16 base patterns. Scale HMUL2 remains. The table is independent of weights. Prepare only permutes
exact4-bit codes into byte pairs;it never decodes weight values or widens their precision.

`table_validation.json` proves all256pairs x32lanes x4byte positions=32768 native checks. Initial local
probes exposed the incorrect assumption that the shared table starts at address0. The corrected address
adds the table's byte1 bias after PRMT and uses256-byte alignment. The final exact64KiB helper passes.
`component_validation.json` verifies all1,610,612,736 weight codes and raw scale codes exactly,plus116 TASK
checks on real/empty/zero/all8 routes,nearby M9/M24,nonzero expert starts and changed captured inputs.

`static_audit.json` binds exact CUDA/cell hashes to sm90a SASS. Both cells use127registers/zero spills,
64KiB static shared,32HMMA/128HMUL2 sites. PAIR+BLOCK's SHF164/LOP3245/PRMT27/LDS132 becomes
SHF60/LOP3124/PRMT155/LDS260. The core15 SHF/LOP3 per16weights becomes8PRMT+8LDS;scale issue remains.
These static counts do not measure dynamic issue or stalls.

LUT1 uses1 raw stage to leave room for two resident CTAs;LUT2 uses2 stages and pays lower residency.
The real H200 resource rows decide. `local_standin.py` is test-only:sm120 cannot use64KiB static shared,
so local full-kernel screens execute the same decoder math with32KiB/32lane replicas and explicit address
arithmetic at1 raw stage. Its native math proof and the full64KiB address-helper proof are complementary;
the actual64KiB one-PRMT image executes on H200. The stand-in is absent from candidates and payloads.

`build_lab.py` packs BASE/S15/LUT1/LUT2,16fresh seeds at M8/16/32,240paired comparisons and48A:A pairs.
TASK gates run before/after,AB/BA is read-flushed. Weight seeds575353+7919*s,routes579000+M+97*s.
Every valid per-shape point win nominates with exact patch,ratio,SE,seeds and descriptive95% CI;
P4's held-out six-load R lower confidence bound>1 confirms whole-layer promotion.

The cost being tested includes table initialization,shared reads,address bias and residency. A loss closes
this exact table/layout/depth recipe;other dequant encodings and smaller-footprint issue cuts stay open.

Admission is complete:4 normal/forced screens each13TASK/12shapes/27static,116 native TASK checks,
exact bank bitproof and32768-case64KiB address proof. The exact packed local16seed lab passes4417chars/62.1s.
Accepted MoE check **11b3849eb86749df**,sender2944970,wrapper8f7217c145421dd0;`queue.json` pins bytes.
No prefill host is used. H200 receipt is complete;verdict below.

## Measured H200 verdict

Check **11b3849eb86749df** passes all16seeds/TASK gates,240pairs48A:A,with exact source audit in `results.json`.
Worst total/row .002377/.002512. Ratios below are candidate/BASE time,95% Student-t15 CI across paired seeds.

| Arm | M8 | M16 | M32 |
|---|---|---|---|
|LUT1|1.185044 [1.174661,1.195427]|1.145672 [1.122240,1.169103]|1.123305 [1.113185,1.133425]|
|LUT2|1.174630 [1.161741,1.187520]|1.190589 [1.169105,1.212072]|1.222294 [1.208673,1.235915]|

A:A mean absolute .569/.344/.390us,max1.072/1.056/.752. All6direct comparisons lose;all6versus S15 lose too.
No P4 point nomination. LUT1 really has2residentCTAs;LUT2 has1,127registers andzero local bytes in both.
LUT1 dynamic47.3–48.3KiB plus64KiB static;LUT2 dynamic84.2–85.1KiB plus64KiB static. Thus residency alone
cannot explain the table losses. Table initialization/shared service,raw depth andissue are not separately timed.
Close only these exact table/depth recipes. Smaller-footprint direct decoding andoverlap remain open.

P4 S18 receives no decoder patch nomination. The earlier measured FAST/INDEX M1 andSPLIT2M32 alternatives
remain in `../../point_nominations.json`,with exact patches anduncertainty;only held-out whole-layerR promotes.
