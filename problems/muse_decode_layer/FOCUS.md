# Decode layer focus — G3 first

Board control: V9 `0e94268cf5654012`, **1.428x**. Serving uses verify shapes.
Current build: [cand_r3_best6.py](cand_r3_best6.py), SHA-256
`c582f3427684b69b67b13be3f1bcb9f43c68fc8b32e9556bffe4377687d85491`.
Owner holds its send until the separate native H200 adapter seed lab passes M128/256.

Base: exact glue `99120569`, sent by the owner as a priority-0 check.
Best5 `a651af25` is check job **`691bff0bcff44e99`**. Neither receipt existed at this turn's one read.
Glue lab queue key **`bd66426465d8b072`** was queued. No judge polling/waiting this turn.

Best6 adds the verified prefill raw17 / humming_masked / dg_w8 adapters at M128/256 beside V9 NT1/NT2.
Three fresh loaded-router synthetic routings, FP32 gates before/after captured replays,
and mean cold timing over seeds select the backend; rejected/unmeasured choices retain measured V9.
Optional construction, admission, timing and replay RuntimeErrors reject only that alternative.
All 75 base helpers/kernels and 27 embedded prefill definitions retain their exact AST/native bodies.
Adapter setup is restricted to M128/256; its packed layouts/scale encoding are preserved as requested.
No prepared BF16/FP8 expanded expert bank; dg_w8's expansion runs inside each timed call.

Normal + cold forced screens each pass **13 tests, 12 fresh graphs, 14 static checks**.
Cold candidate prepare **174.799 s** (**176.472 s** with forced checks). **18 RuntimeError cases** pass,
plus all optional build rejections and expired defaults. Native library GEMMs still need the H200 lab.
[Best6 handoff](dev/best6/README.md) records final evidence, provenance and the submission hold.

The glue base has three PDL launches, no cross-CTA barriers/counter clears; its <=8 us H200 target is pending.
[Board glue inventory](dev/glue/PROFILE.md): 17.2/17.7 us at decode 1, 31.1/29.8 at verify 32;
profile durations include dependency waits and overlap compute. Both base/glue receipts remain to fold into
[NEXT_CHECKS](dev/NEXT_CHECKS.md) on owner resume. Existing megakernel/overlap holds remain unchanged.
