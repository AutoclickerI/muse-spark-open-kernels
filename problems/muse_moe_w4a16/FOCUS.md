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

Next: decode-layer integration consumes `dev/V9_ABI.md`; wait for V9 scored receipt,
then compare its per-load table with V11 and port the winning dispatch into the layer.
