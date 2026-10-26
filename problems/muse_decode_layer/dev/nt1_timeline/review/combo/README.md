# P3a winner composition

Base S15 `60254883`; selection follows measured point estimates in retry `b57d691d689046a8`.
Original `1da2153a0f5543b9` is unmeasured because the wrapper dropped its archive.

| M | COMBO | Best independent control |
|---|---|---|
| 1 | PAIR+SHARD2 | SHARD2 |
| 8 | PAIR | PAIR |
| 16,32 | PAIR+BLOCK | PAIR |

All eligible independent point winners,including alternatives,are handed P4 in
`../../point_nominations.json` with ratio,SE,seed indices and exact S15 patches.
M1/M8 composition CUDA is byte-identical to SHARD2/PAIR. The new BLOCK+PAIR image retains
store-fence-barrier-publish-acquire order,512B readiness region,125regs,zero spills,32HMMA/128HMUL2.
CANCEL and PACK5 are excluded because their independent implementations lose at every measured M.
The identity and smaller lossless encodings remain open via other implementations.

`manifest.json` pins the exact candidate6884c929 and S15 patch `combo.patch` (6b3990b2).
The composition is measured by c49dbdffe92d491a and nominates at all4 M versus S15. `admission.json` verifies:

- 90 native TASK checks per variant,270 total: real/empty/zero/all8 routes,M1/8/9/16/24/32,
  nonzero expert starts and changing captured inputs.
- Normal and forced `tools/local_eval.py`:13 TASK cases,12 shapes,28 static checks each;
  every variant called eagerly and captured. Hopper-only glue uses explicit local stand-ins;
  changed MoE math is native on GPU0.
- Exact packed lab16seeds/64AA,all TASK gates pass,4371characters,79.9s locally.
  Local timing is not H200 evidence.

`../build_combo_lab.py` makes `prof/lab/moe_p3a_combo.py`: S15/PAIR/BEST/COMBO,16 fresh weight seeds
275353+7919*s and route seeds279000+M+97*s,M1/8/16/32,AB/BA read-flushed timings,TASK before/after,
64 independent A:A pairs and direct COMBO/BEST. The bounded receipt retains every seed ratio and A:A.
P4 confirms the whole-layer composition by held-out six-load R; no additive gain is assumed.

Sent on MoE as a check,priority1.2,import budget240s: sender1548101,wrapperda4bfcfab9f0b0db.
Accepted judge job**`c49dbdffe92d491a`**;`queue.json` pins the exact submitted source. Receipt complete;all16seeds/TASK gates pass. No prefill host is used.

Measured COMBO/S15 at M1/8/16/32:.997261/.988757/.970993/.979561;direct COMBO/BEST at M16/M32:.995504/.993125. `results.json` contains SE,seeds,A:A and exact-source audit;every point winner is handed P4.
