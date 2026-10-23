# Projection and DFlash compute handoff

Serves G1/G2; read [GOAL](../../docs/GOAL.md), [CONSTRAINTS](../../docs/CONSTRAINTS.md).
Proj alone cannot move the fitted overall rank; its kernels serve decode and serving.
Solo, GPU0 only. Jihoon/orchestrator sends and commits; this thread did neither.

Ready next: **`cand_dflash_diag.py`**, frozen **`9b4fae2f…`**; full SHA in NEXT_CHECKS.
Scored source/dispatch remains byte-identical f3786341611f4f1f (`cand_fused_select.py`).
Prepare separately measures TP8 DFlash dense shapes and stock-versus-fused glue:
cold128MiB, fresh inputs, graphs, 20 pairs, exact stock guard, reject/log errors.
225s diagnostic deadline leaves 15s below the task's 240s limit; compact results print last.
Local **sm_120**: 9/9 official tests, 30/30 graph shapes, 45.85s prepare;
all48 diagnostic comparisons complete with no case rejection or budget skip.
Serving module: `layers/muse_dflash.py`; unchanged released BF16 drafter weights.
Tuned 2D dense kernels, exact two-tap/group16 convolution, selector input fusion, unary.
7 unit tests: 464 exact graph comparisons +464 stock repeats, 14 forced Hopper ABI cases.
16 pointwise SM90 variants and the dense SM90a utility compile; execution unverified.

[Full inventory](../../docs/reference/muse_dflash.md): M=8C, prediction P=7C;
QKV1280, o K1024, gate/up7168, down K3584, conv coefficients2048, head25256/rank.
Old562us/234us opportunity is stale: FC split already wired, coefficient GEMMs omitted.
New structural model saving is **0.006ms/step at C1/4/16/32**, plus unmeasured dense/glue.
Strided selector hidden stays stock bmm; flattening changes BF16 results.
Serving lane wires only H200 winners and checks full-model agreement, memory and TTFT.

Receipt `3afea80c527c4969` failed prepare: missing `_AllModule.old_hopper`; no score.
Do not reuse that source's results as a passing portfolio. The orchestrator owns its fix.
Decode `cand_r3_proj.py` was accepted/committed24d577c; decode lane owns subsequent work.
[dev/NEXT_CHECKS.md](dev/NEXT_CHECKS.md) owns ready hashes, validation and receipt follow-up.
