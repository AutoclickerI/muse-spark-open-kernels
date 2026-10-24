# Decode layer focus — best6 attention correction

Board control: V9 `0e94268cf5654012`, **1.428x**. Serving uses verify shapes.
Current build: [cand_r3_best6.py](cand_r3_best6.py), SHA-256
`4516cb979dac623a6b6a2347be54cb5ecb416be23d7631aeae092a403beb9528`.
Owner holds its send until the separate native H200 adapter seed lab passes M128/256.

Best5 check **`691bff0bcff44e99`** passes at **1.353449x**, prepare **158.992 s**.
Verify32 full FA3 main regressed **377.9 -> 501.5 us**; decode full attention improved.
MoE verify16/32 differences match routing-seed variance. [Full stages](dev/best6_attention/BEST5_STAGES.md).

Best6 now compares q1/q8 attention settings on three exact public load draws by mean cold timing.
Inputs use soft caches, ctx80000/page1, fixed B1 and agentic B16/32 lengths, with loaded QKV weights.
The q8 `layer` choice restores the board's exact FA3 call, forwarding public scheduler metadata (None).
Incomplete trials retain `layer`. q1 gets the same sampling without changing its kernel choices.
All **114** other helper implementations and all native sources remain exact best6 `c582f342`.
Its raw17 / humming_masked / dg_w8 portfolio at M128/256 and V9 NT1/NT2 remain unchanged.

Normal + cold forced screens each pass **13 tests, 12 fresh graphs, 14 static checks**.
Cold prepare **196.634 s** (**198.312 s** with extra forced checks). All 12 attention shapes sample
three public draws with `layer` timed. All **36 complete fixtures match the public generator bitwise**.
[Current handoff and validation](dev/best6_attention/README.md); native H200 adapter safety remains pending.

Glue check `7370b18221964412` is **1.367496x**; routing/FA3 changes explain the difference from best5.
Paired lab [decode_paired.py](../../prof/lab/decode_paired.py), send key **`731545a5def446ad`**, tests
all 12 loads x 3 public seeds: whole-layer glue A/B + A/A noise, q8 FA3 layer/splits, NT1/NT2 and experts.
Cold forced payload **115.283 s / 124 s**, **216 paired comparisons**, no rejections; 13 tests/12 graphs/5 static.
Paired H200 means decide glue/attention; native adapter safety remains the only best6 send hold.
[Protocol and validation](dev/paired_lab/README.md). Earlier proj glue batch: **`b6b6b6f04beaad02`**.
