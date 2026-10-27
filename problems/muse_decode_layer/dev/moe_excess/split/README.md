# P3b split: S22 large-M load path and branch serialization

**Measured J `c22acbbfdcd94fe1`, I `c67e0a1ecb984506`.** Tables: [result/split.md](result/split.md),
`result/summary.json`. FINDINGS "P3b S22 large-M split" has the decision.

- The ladder's load term (L0 - STREAM, 45.1 / 80.4 us at M128 / M256) was mostly L0's own sink tax
  (38.5 / 56.1). The true loads-only term is 6.6 / 24.2 us (six-load R1.013): weight rereads 2.5 / 14.2,
  activation copies 2.5 / 7.3, per-thread issue 0.9 / 2.7. Forcing any residency or ring depth costs
  time, and L2 bulk prefetch adds 16-28 us.
- Largest: branch serialization, the joint phase above max(heavy alone, unique stream): 53.2 / 80.0 us
  (R1.067 [1.057, 1.078]). Heavy at 2 CTAs/SM fills the register file (2 x 256 x 128) and small at 2/SM
  fills shared memory (2 x 94 KiB), so branches do not co-reside; small CTAs enter W13 at a median
  33 / 54 us. Heavy alone runs 93.8 / 164.4 us against a 26.9 / 61.1 us padded BF16 tensor roof.

## Lab (exact S22 `4b8c803d` source)

`build.py` clones the three S22 large-M branches (heavy `chunk::kernel<4,F>`, medium
`mwbr::gemm<16,F,7/6,1,8,16>`, small `pmz::phase`; small W2 from `MT_PMZ_CUDA`) into one translation unit
with excluded-work modes, and writes the payloads `out/split_{J,I,I2}.py`. Each W13 / W2 phase runs alone
in S22's stream structure (heavy on a side stream, medium on a light stream, small on the parent) from
frozen routing metadata. Eight seeds (`manifest.json`), M128 and M256, four palindromic cold rounds,
every round kept in 32 ns units; full-math arms pass TASK before and after timing.

| Set | Arms |
|---|---|
| J | BASE, AA, L0 (ladder loads, sinked), L0N (loads only, no sink), NOX (no activation), REUSE (each expert read once), ISSUE (16 B per thread), RES1 / RES2S3 / RES3 (forced residency), PFN / PF (L2 prefetch, loads only / full math), STREAM, HISTN (CTA traces) |
| I | Each branch alone: BASE, L0N, STREAM; joint BASE, AA, L0N, STREAM |
| I2 | Heavy alone at 2 and 1 CTAs/SM, small alone, medium alone, medium+small, joint with heavy at 1 CTA/SM, full-math CTA traces of BASE and R1 |

`read.py <J> [<I> [<I2>]]` checks identities, seeds, TASK counts and rounds, and reports paired
contrasts with t7 95% intervals, six-load transfer ratios on S19 walls 321.09 / 425.66 us, heavy reread
and tensor roofs from the per-seed expert counts, branch serialization and CTA timelines.

## Production candidates

`prod.py` patches the literal S22 `P3B_CUDA` / `MT_PMZ_CUDA` sources (only v16 / v32 run them) and writes
single-file candidates (`PF.py`, `RESH1.py`, `CO.py`, `XW4.py`, `candidates.json`). `prod_lab.py <arms>` builds
a whole-load cohort on 64 fresh seeds x 16 symmetric rounds (P4 `ec754e5` protocol, INC control, A:A,
every arm TASK and nearby); `prod_screen.py <tag>` rehearses it locally; `prod_read.py <receipt> <tag>`
reads it with `tools/promotion.py`.

| Arm | Mechanism |
|---|---|
| PF | In-kernel L2 bulk prefetch ahead of each branch's ring: loses, R 0.9857 [0.9851,0.9863] (`8e23096f`; J predicted 0.9863) |
| RESH1 | Heavy ring reserved at half the SM, so one small or three medium CTAs co-reside with one heavy CTA |
| CO | An expert's 64-row chunks on adjacent CTAs, so its weight rereads share L2 |
| XW4 | Four warpgroups share each heavy X stage (one 512-thread CTA per SM) |

Queue and receipts: `queue.json`.
