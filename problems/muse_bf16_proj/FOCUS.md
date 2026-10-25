# Projections and glue in the decode layer (P7)
Serves: G2 (dense GEMVs, verify dense GEMMs and glue inside the decode layer), for G1. Levers at 1.3x floor on S4:
glue +7.03%, projections +5.93% ([lever table](../muse_decode_layer/dev/NEXT_CHECKS.md)).

Incumbent: best7's per-role dispatch (cuBLAS/Lt and custom GEMV) in S1-S6, plus O config6 at decode 1 in S6:
`dev/gemv_fixed/projection.py` (Best7ProjectionCell), SHA-256
c42e9fcc02a6ce57652507e763e1c5dbd67fc459b32a14684f379ac2ffe9d9fe, 15.39 -> 15.04 us with its consumer
(`95ab383d`, near noise). Glue: best7_paired geometry.

In flight: equal-work bulk-ring dense batch `87b2667f` (send `ef2048bf`; `dev/bulk_ring/`): 16 role/M pairs, 21
configs, O/boundary/pool segments, cold AB/BA and A:A. Pool+router at M=32 has the largest gap to a 3.6 TB/s
stream (12.7 us per layer).
Measured losses (exact recipes; FINDINGS): cooperative glue `9eeb39ba` 1.17-9.5x (mechanism not isolated); fused glue
`80b70f7f` (a bug: 576 norm copies per token); 15/16 register-MMA GEMV finalists `95ab383d`; persistent chain
`87dc944b`; verify split-K `19f3a13b`; side-stream L2 prefetch `d5a90258`. PDL prefetch `7426e17c`: 1 of 8 cases
measured, the rest unmeasured (our 0.3% gate). L2 oracle `09d19813`: later-stage prefetch caps at ~2-4 us per layer.
Hardware compression `4233660b`: real weights read at ~1.0x.
Next (PLAN P7): read `87b2667f` once and hand any winning role/M cell (path + SHA-256) to decode FOCUS; native route
top-8; the T<=8 zero-tail boundary; router + top-8 beside the pooler.
