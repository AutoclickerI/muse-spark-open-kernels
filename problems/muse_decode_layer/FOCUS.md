# Decode layer: integration and scored runs (P4)
Serves: G1 (our decode-layer speedup is the overall score). Lever table and its refresh: [dev/NEXT_CHECKS.md](dev/NEXT_CHECKS.md).

Board: our best scored run counts (v9 `0e94268` 1.428x before this reset). Scored runs this reset, cumulative (path, SHA-256, job):
- S1 `cand_r3_best7_scale.py` 42f543a0e25f98058099e417ad2f8b95a71e1ab33c7371b3af6a4499905cbad7, `0a43baec` 1.4667x: best7 + recoded scale bank M1-128
- S2 `cand_r3_best7_scale_w2fix.py` b080544708544fd67ee23b82aca111bf0b9bb82c30a5eb86315044cd9bf92337, `4e212ec2` 1.4742x: + raw CTA1g2112 M256
- S3 `cand_r3_best7_scale_recoded_m256.py` cadef19a6eced1ae1fa0f3deaffa9ee21e083aa5ba6fe46784707bb8df52a7c7, `5c7323ea` 1.4665x: recoded C1g2112 M256
- S4 `cand_r3_best7_s4.py` 9b2b2a48337aa13ba1daa45eec8a30b566cc3e59c7e1bb317d39695413d3404e, `ad858b58` 1.5279x: + LOP3 NT1 M1-32, D32g M128
- S5 `cand_r3_best7_s5.py` f7064fc0127e80a955c7dc9aa5c68eb6aefada179c2841c5c58bce8dabd20284, `e2f45792` pending: + H_H M128
- S6 `cand_r3_best7_s6.py` 4c98f12fc7fab9730b90d3bae2f7f014ce85bae35439fb688282d98bd167c59f, `fecc29af` pending: + mixed_parallel M256, O config6 M1
Provenance and screens: `dev/s2/`-`dev/s6/` READMEs; S1 in FINDINGS. Unchanged shapes differ by up to ~4.5% between runs:
promote from paired labs on identical inputs and many seeds, not from one scored run.
Judge prepare: S1 170.2 s, S2 173.6 s, S4 200.8 s of 300 s; each new native module adds cold compile time.

Cell handoffs (path, SHA-256; all in S6):
- LOP3 NT1 M1-32: `dev/nt1_branches/nt1_lop3_cell.py` 3f84a8a4e12b41bd7257712d8656d3f417f6a9e238f00e481b3be526da69cc40
- D32g M128: `../muse_moe_w4a16/dev/mixed_wgmma/m128/phase_mix/d32g_cell.py` a7afb63d9cb17c920412c8f3003640bf956599ab6caa874a9631f290798dce82
- H_H M128: `../muse_moe_w4a16/dev/mixed_wgmma/m128/phase_mix/hh_cell.py` 970663d44d993f8a029cd4289e2a9716bd2bf5897098799a6e0605e0abf21f0f
- mixed_parallel M256: accepted payload of `8483c0c6`, cb20858c5de9bfaca16ee289529cc2f46554ed575e32a7ce16dcccd31b1cd179 (`dev/s6/build.py` extracts it)
- O config6 M1: `../muse_bf16_proj/dev/gemv_fixed/projection.py` c42e9fcc02a6ce57652507e763e1c5dbd67fc459b32a14684f379ac2ffe9d9fe
- Not in S1-S6: verify-1 sliding single launch `dev/verify1_sliding_pick.py` 46b940320048047cace11d16807e160c11cdd84f364ff902cd04bacf81ee263e,
  19.55 vs FA3 path 20.16 us at partials=0 (`a1679f9c`)

In flight: S5 `e2f45792`, S6 `fecc29af`; labs M128 mix sweep `4c469bf9` (proj), M256 mix `ec44b322` (MoE), dense bulk ring `87b2667f` (proj).
Next: S7 = the best measured S plus each landed lab win and the verify-1 sliding cell; rerun `dev/levers/build.py` when S5/S6 land.
Withdrawn: S4-S6 (`ad858b58`, `e2f45792`, `fecc29af`) carry D32g/H_H's 16-bit scale words, a dequantized copy
of the scales (CONSTRAINTS 5); S7 = S6 with H_H on uint8 scales (MoE lane). Clean best: S2 1.4742x `4e212ec2`.
Parked: megakernel at +1.74% estimated (`dev/megakernel/README.md`). Glue recipes lost: `80b70f7f` (bug), `9eeb39ba`.
