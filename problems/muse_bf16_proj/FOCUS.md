# Decode projection/glue fusion (P7)

Solo, GPU0 only, no commits. Send labs only; Jihoon sends decode candidates.
Never poll/wait for judge; check receipts once per turn, then build or end.
Stock CUTLASS recipe set closed by45f58e7009c440eb: all144/key lose at M128/256.
Best20-pair new/control ratios M256 qkv1.306,pool1.254,unpool1.082,o1.183;
M1281.322/1.244/1.142/1.186. Close these recipes, not the family.
Old custom TMA v2 92e78027 also loses1.21–1.95x; v1 remains invalid.
H200 Lt596b2a5d: M128 qkv21.10->20.22,unpool24.00->22.96us, others parity.

Active: O/unpooler GEMV post-norm+residual+next-norm epilogues; pooler/router
hi+lo GEMV with pre-expert norm and sigmoid/biased top8/renormalization.
BF16 boundaries preserved; nonspinning last-CTA reduction resets counters in-kernel.
M1/8/16/32/128/256,300 configurations,96 unique sm90 specializations.
Lab uses exact cand_r3_best812f5f36 projection/glue code and one shared sync reset;
synthetic norm/gates when the proj host lacks them. Cold1GiB/fresh3/20 paired graphs.
Only a checked combined-chain winner enables runtime; no per-stage shortcut.

Sent priority0: prof/lab/verify_projection_glue.py, SHA34143ac6134efe1f…;
host4b4182cb1d0ff580…; waiting request468733, no assigned judge ID at local check.
Drop-in dev/glue/projection_glue.py SHA c763555f99081588…; full hashes in NEXT_CHECKS.
API/merge/read commands in dev/glue/README.md; selected/unknown shapes retain incumbent.
sm120:300 configs/900 fresh graphs; all96 compile sm90. Host9/9+30/30+5 static;
layer13/13+12/12+11 static,5.740s cached prepare. All19 block definitions match.
Every combined chain loses locally; closest M1 tail-only is1.019x slower.
H200 timing/cold prepare and whole-layer/PDL effects remain unverified.

#45015 belongs to serving lane. No prepared expanded NVFP4 bank added.
MoE belongs to its lane; old grouped path is withdrawn. Dense V11 remains a separate job.
