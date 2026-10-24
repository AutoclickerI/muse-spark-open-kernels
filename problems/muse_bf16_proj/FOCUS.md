# Projection lane focus (P7 / G1)
Serves: G1, G2. G0 is the overall decode-layer board score (R-20261025-1).

P10 hardware-compression recipe closed on H2004233660b1bff4d18.
Send3110dc39be425573; support1/create success/granted1.
Zero/pattern BW2.0265x/1.7008x; real raw/recoded NVFP4 and BF16~1.00x.
NT1 M1/M8 and QKV/O time ratios1.0025/1.0009/0.9996/1.0024: no gain.
Mapped bytes, W2, ordered MoE and dense outputs bit-exact.
Incumbent norm atomics vary even AA (M1 10/24, M8 16/24); raw failures retained.
No compression drop-in emitted. `dev/memory_compression/h200_result.json` owns it.

P7 fixed-cost95ab383de42a4fae landed:15/16 role finalists lose.
Native observer entry-first47-101cycles; PDLwait10-11; mainloops dominate.
M1 pool removes2.49us from consumer but adds3.01us producer time: loses3.4%.
Next implementation lever is mainloop transport/weight reuse, not fewer CTAs alone.
Measured candidate: M1 O/config6, BF16, full key(1,1,0).
Producer+consumer15.39->15.04us (r0.9783); O/boundary/pool41.95->41.29 (r0.9846).
6 live full/sliding cases: exact routes and official gates; max hidden drift0.1833%.
Cell: `dev/gemv_fixed/projection.py`, Best7ProjectionCell.
SHA256 c42e9fcc02a6ce57652507e763e1c5dbd67fc459b32a14684f379ac2ffe9d9fe
Decode handoff: `dev/gemv_fixed/handoff.json`; no scored dispatch changed.
Whole-layer H200 score gain/sliding latency unmeasured; bundle with larger change.
Full decomposition: `dev/gemv_fixed/h200_result.json`; README owns reproduction.

Oracle09d19813 caps later-stage prefetch~2-4us; chain87dc944b loses38.3->57.7us M1.
PDL7426e17c measured one M1 case, noise; remaining cases failed the0.3% gate.
