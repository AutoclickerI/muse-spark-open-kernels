# MoE inside the decode layer (P3a, P3b)
Serves: G2 (MoE at the decode layer's M = 1-256), for G1; the MoE board itself doesn't count.

Incumbents (cells and SHA-256 in [decode FOCUS](../muse_decode_layer/FOCUS.md)):
- M=1-32: one-LOP3 NT1 on the recoded bank, 42.74/73.75/79.20/113.61 us at M1/8/16/32 (`dc3fea0a`); in S4-S6.
- M=128: H_H 234.44 us (`770809cf`, 12 seeds x 3 fixtures; D32g 272.70, NT1-recoded 289.62): NT1 for experts with <=16
  tokens and u16-scale WGMMA32 (config1, grid1056) above, in both W13 and W2, then the distributed finish; in S5-S6.
  W13 ~133 us against a ~75 us unique-byte model is the target; W2 ~82-85 us.
- M=256: mixed_parallel 344.5 us (`8483c0c6`; recoded C1g2112 366.2, masked Humming ~448): parallel N16/N32 W13
  branches, recoded W13/W2, distributed W2/finalizer; in S6.

In flight: M=128 threshold 8/16/24/32 and heavy N32/N64 sweep `4c469bf9` (proj host; `dev/mixed_wgmma/m128/mix_sweep/`);
M=256 per-expert NT1/WGMMA32/64 mix `ec44b322` (MoE host; `dev/mixed_wgmma/m256_mix/`).
Next (PLAN P3a, P3b): read both receipts once and hand any winning fixed cell (path + SHA-256) to decode FOCUS; delete
M=256 `route_init`'s dead zero-fill; heavy-expert producer/consumer (`dev/mixed_wgmma/m128/producer_consumer/`, built,
unsent); at M=1, single-timeline stamps and direct top-8 dispatch.

Closed recipes (FINDINGS): one-read M=128 `640d67e7` (362 us; changed tiles too, so reread cost is not isolated), fused
producer `471f6d62` (serialized W2 combine), raw64 `63c980e8`, grouped TMA `c8c89f9f`, CUTLASS mixed input `3a98cfc8`
(stock mainloop can't do group-16 scales).
Pointers: `dev/mixed_wgmma/README.md` (M=256 producer), `dev/mixed_wgmma/m128/phase_mix/README.md` (H_H),
`../muse_decode_layer/dev/nt1_branches/README.md` (small M), `dev/V9_ABI.md` (bank ABI).
