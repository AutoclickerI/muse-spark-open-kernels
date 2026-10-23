# MoE focus

Serves: G3, G2, G1.

Measured H200 receipts drive this round. V9 check `f6028a478d594b31` is 1.654x overall:
2.196/1.717/1.392 at decode M1/16/32 and 1.813/1.590/1.356 at verify M8/128/256.
The prior V7 scored 1.535x; hybrid V9-native/Humming-wide check `318227bc382d465c`
was 1.536x, with 1.406 at M128 and 1.439 at M256. Humming6k alone reached 1.222/1.163
at M128/M256. Thus V9 is native control; Humming is the wide M256+ path.

`cand_v11.py` combines one V9 canonical bank with Humming and times fixed-shape plans in
prepare(); its sm120 screen passes 5/5 tests and 6/6 graph shapes at 1.262x local GM.
Serving ports V9 native through M=128 plus the same-bank grouped wide path from M=256; layer agreement passes
M=1,8,16,32,64,128,256,753,3012,16384. The canonical bank is 864 MiB/GPU.

Next: `cand_v9_phase.py` is first in the 09:00 KST 20-check queue; its plain kernel is V11,
while prepare() prints H200 clock64 spreads and ranks V9/V12/V16/Humming on all six shapes.
V9 headroom model (routing means from `sim/routing.json`, direct distinct experts): total bytes
are 74.9/338.3/451.5/239.5/644.1/694.7 MB for M=1/16/32/8/128/256; achieved bandwidth
is 2.14/3.11/3.08/3.06/2.17/1.50 TB/s. Weighted excess over 3.075 TB/s is ~323 us
at verify-32 and ~138 us at verify-16; tensor floors are 19.85/39.38 us at M128/M256.
The diagnostic sm120 trace with H200-like local pairs reports M128 median p1/p2/wait/barrier
770.6K/384.4K/15.5K/61.3K cycles and 416 atomics; M256 734.5K/362.3K/13.6K/30.5K
and 656 atomics. M256 has 387 pairs, 24.39 experts and 60.53 V9 token-blocks on average;
V9's 188 CTAs receive 163.0/163.0/164.0 tiles (max/mean 1.01x over 200 routing samples),
so CTA imbalance is not the explanation. The pair-scaled cost is block padding: `ceil(n_e/8)`
token blocks rise 36.76 -> 60.53 from M128 to M256 (1.65x), repeating 512/256 weight tiles;
padded norm atomics are about 386/320 per local pair. V16 partial reduction is embedded for
evidence; it passes local correctness but adds global traffic, so serving remains unchanged until H200 data.
