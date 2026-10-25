# Decode layer: integration and scored runs (P4)
Serves: G1 (our decode-layer speedup is the overall score). Lever table and its refresh: [dev/NEXT_CHECKS.md](dev/NEXT_CHECKS.md).

Incumbent: S7 `cand_r3_best7_s7.py` bd3932d1cc3c0c82d60145772148fd19fc779465dd360bda610d214385401390, 1.5597x `2f6471d5`
(scored 10-25, our board score). Per-load results of every S run: FINDINGS "Whole layer and integration".
Chain: S1 recoded scale bank (`0a43baec` 1.4667x) -> S2/S3 fused M=256 (`4e212ec2` 1.4742x, `5c7323ea` 1.4665x) -> S4
LOP3 NT1 M1-32 + D32g M128 -> S5 H_H M128 -> S6 mixed_parallel M256 + O config6 M1 -> S7 H_H on uint8 scales.
S4-S6 (`ad858b58`, `e2f45792`, `fecc29af`) were withdrawn: D32g/H_H read a 16-bit scale copy made in prepare (CONSTRAINTS 5).
Unchanged shapes differ by up to ~4.5% between runs: promote from paired labs on identical inputs and many seeds.
Judge prepare: S1 170.2 s, S4 200.8 s of 300 s; each new native module adds cold compile time.
Before a cell enters a candidate, list its prepare() tensors (dtype, bytes per 16 weights; FINDINGS "Failures").

Cell handoffs in S7 (path, SHA-256):
- LOP3 NT1 M1-32: `dev/nt1_branches/nt1_lop3_cell.py` 3f84a8a4e12b41bd7257712d8656d3f417f6a9e238f00e481b3be526da69cc40
- H_H M128 (u8): `../muse_moe_w4a16/dev/mixed_wgmma/m128/phase_mix/hh_cell.py` 365b623992367a1d45d3dae06a2096ddd0e6879deffab2e3924678a47997b621
- mixed_parallel M256: accepted payload of `8483c0c6`, cb20858c5de9bfaca16ee289529cc2f46554ed575e32a7ce16dcccd31b1cd179 (`dev/s6/build.py` extracts it)
- O config6 M1: `../muse_bf16_proj/dev/gemv_fixed/projection.py` c42e9fcc02a6ce57652507e763e1c5dbd67fc459b32a14684f379ac2ffe9d9fe

Ready, not in S7:
- S7R `cand_r3_best7_s7r.py` 56c51fa026210357f9865d114e6bf7e609c52049e40b263f589d23ade95a8101: M=256 on R_full (0.9227x
  mixed_parallel's MoE, paired `110599ba`, bit-equal); sent as a scored run.
- verify-1 sliding single launch `dev/verify1_sliding_pick.py` 46b940320048047cace11d16807e160c11cdd84f364ff902cd04bacf81ee263e:
  19.55 vs FA3 path 20.16 us at partials=0 (`a1679f9c`).
- small-M flags `dev/nt1_timeline/smallm_var_cell.py`: per-M picks pending from `849e4017`.

Next: S8 = S7R plus the small-M flags and each landed lab win; rerun `dev/levers/build.py` on each new incumbent.
Parked: megakernel at +1.74% estimated (`dev/megakernel/README.md`).
