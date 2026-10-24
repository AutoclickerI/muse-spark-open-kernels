# MoE focus
Serves: G1, G2.

G1/G2: the decode-layer score (R-20261025-1). Own M128 MoE.
Incumbent is NT1 on v9's lossless recoded scale bank:283.6us versus raw322.2us
on identical routes (58ae2e2cf8df45ef); one-read floor~106us. The old344us bar
and raw-only bank restriction are stale. Free lane owns M256 and W2/final latency.

Sent M128 raw/recoded fused-WGMMA comparison on MoE: send def9bff54e82c1b0,
judge471f6d6216804dda; detached log confirms accepted. Payload:
`prof/lab/moe_m128_scale_bank.py`; frozen hashes/protocol in
`dev/mixed_wgmma/m128/scale_bank/manifest.json`. No decode dispatch promoted.

Recoded producer reads the same9 bytes/block: E2M1 codes plus uint8 exponent/
mantissa scale; sign/zero folded into codes in prepare. Four byte permutations
replace four shared scale-table reads; no CTA E4M3 conversion/table remains.
K64 register-bank repair retained. Configs3/4/5, grid528, paired against actual
board NT1-recoded;12 public-router weight seeds x3 fresh inputs, six reversed/
rotated cold128MiB graph orders, NT1 A:A samples, independent FP32 gates.

Local: all36 fixtures/216 forced producer graphs pass; maxL2<=0.004132,
row<=0.004851; all4064 finite E4M3/E2M1 combinations exact. Wrapper5 tests,
6 scored graph shapes and6 static checks pass. Native12 kernels: no spills/
C7513; all20 K64 paths retain disjoint A banks. H200 ranking is unverified.
Recoded-only best7 drop-in passes21 graph gates with signed/zero/subnormal
scales and changing routes; H200 winner alone gets path+SHA handoff.

Prior K64 repair send0ea82c48f2255889 still has no receipt at this turn's one
read. It is independent of the scale-bank lab; no waiting/polling was added.
Next resume: read471f6d6216804dda once, require every numeric gate and complete
seed means, then hand the decode lane a winning M128 cell or reject it.
