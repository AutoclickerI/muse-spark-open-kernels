# MoE inside the decode layer (P3a, P3b)
Serves: G1, G2 (the decode layer's MoE at M = 1-256; the MoE board itself doesn't count).

Incumbents in S7R `cand_r3_best7_s7r.py` 56c51fa026210357… (1.5735x `c1fbd0cf`; S8 carries the same MoE):
- M=1-32: one-LOP3 NT1 on the recoded bank (small-M lane: `../muse_decode_layer/dev/nt1_branches/`).
- M=128: H_H on uint8 scale codes (NT1 <=16 tokens, WGMMA32 above, per phase; distributed finish). Cell
  `dev/mixed_wgmma/m128/phase_mix/hh_cell.py` 365b623992367a1d…; 234.44 us on `770809cf` (u16 predecessor, bit-identical).
- M=256: R_full (live 1-CTA route; one fork: <=16-row groups on N16 || full groups on N32, W13->W2): 0.9227 vs
  mixed_parallel (`110599ba`, bit-equal).
Ready at M=128: N_full 207.40 vs exact S7 H_H 234.78 us, 0.8842, bit-equal 36/36 (`c01b1916`; HH_u8 0.9702, U16 words
0.9931: S7's in-register scale word costs 0.7%). Ready at M=256: N_full 298.68 vs mixed_parallel 342.80 us, 0.8701 (R_full 0.9230, `14f942d5`):
NT1 <=16 W13->W2 on a side stream || heavy-only N32 W13->W2, one join; reads S7's H_H base + NT1 banks, no new bank.
One cell for both M (heavy grid 1056 at M<=128, 2112 at M=256): `dev/mixed_wgmma/branches/nf_cell.py` de34914e47242d68…;
on S7R `cand_r3_best7_s7n.py` a2a52a108e97d65b…
(mixed_parallel's engine and its 1.69 GiB bank deleted; 72 fixtures within 6e-8 of the timed arm; normal + forced 13 PASS).

In flight: M=128 branch lab `c01b1916` (MoE host: exact S7 control, U16 = cost of S7's in-register word, HH_u8,
N/NR/NRm/B8/B8R branches); producer/consumer `9cd95e72` (proj host, u8 control).
Next: hand the M=128 winner as a cell on S7N; put the typed-smem WGMMA (HH_u8: 0.948x idist at M=256) into the
M=128 heavy path and N_full's heavy branch; at M=256 sweep N_full's heavy grid/tile and the W2 NT1 limit (B8_full 0.8768).

Closed: one-read `640d67e7`, fused producer `471f6d62`, raw64 `63c980e8`, grouped TMA `c8c89f9f`, CUTLASS mixed input
`3a98cfc8`; same kernels on one stream (1.016) and N8 remainders (0.9958, `110599ba`); heavy remainders on N16 once NT1
owns the small experts (NR 0.9305, NRm 0.9725, `14f942d5`). zero_()-flushed labs (before 8e6a7e6): paired ratios only.
Pointers: `dev/mixed_wgmma/branches/README.md`, `m128/phase_mix/README.md`, `m128/producer_consumer/README.md`, `dev/V9_ABI.md`.
