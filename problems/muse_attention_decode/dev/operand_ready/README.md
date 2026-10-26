# P5 operand readiness audit

The proposed shared-`full[slot]` split does **not** exist on S15's active
sliding paths. No barrier arm or H200 payload is built. This drops the
specific reviewed implementation claim; it does not close attention
rescheduling generally.

Exact source: [S15](../../../muse_decode_layer/cand_r3_best7_s15.py), SHA256
`602548834f7481747701469b2a85154dde214bea28076c10fefce2c46b5240cb`.
Run `python3 problems/muse_attention_decode/dev/operand_ready/audit.py` to
rebuild [audit.json](audit.json) using AST/static source checks. Extracted
CUDA is inspected without executing or changing the submission.

| Item | Evidence | Decision |
|---|---|---|
| Decode1 sliding | `_CLUSTER_ROWS=((1,1,True),)`; `cs_CUDA` SHA3815beab, `sliding_main` lines619-625: `cp_wait<1>`, `__syncthreads`, V transpose, then QK in the same warpgroup. No `full` ring. | Drop splitting an existing shared-full barrier. Serial transpose/QK reordering remains a different experiment. |
| Decode16/32 sliding | `CUDA_SRC` SHAb63e42f6, loop lines427-439: `cp_wait<STAGES-2>`, `__syncthreads`, then `compute_tile`. No `full` ring. | Drop splitting an existing shared-full barrier. A new producer/consumer pipeline remains open. |
| Verify1/16/32 sliding, including claimed verify32 VQ8 | `_VQ8_ROWS` at S15 line7733 is only `(16,8,False)` and `(32,8,False)`. `_best7_attention` line7006 checks these exact keys; all verify sliding settings choose `layer`, and `_attn_port_variant` line5043 calls `_attention_fa3_public` line6709 on Hopper. | Drop the VQ8-sliding claim. FA3 internals are not established by this audit. |
| Full verify16/32 VQ8 | `VQ8_CUDA` SHA9a23ca9a: producer lines443-450 completes copies, transposes V, then publishes `full[slot]`; consumer `step` line628 waits on `full[(rr+1)%ST]` before QK. Prologue waits line744. | Confirmed on full only; P5b owns its separate K/V-ready arms. |

CUDA line numbers above refer to the extracted literal; source assignment
lines and all barrier occurrences are retained in `audit.json`. Python
references refer directly to S15. The audit also checks that prepare returns
the static VQ8 table and no other assignment enables sliding VQ8.

Native cross-check: S15 timeline receipt `7eaa262a30ec483e`, with the same
source hash, records `sliding_main` at d1S, `attn_main` at d16S/d32S, and
`_qk_prep_kernel`, `flash::prepare_v`, two CUTLASS attention kernels and
`_norm_gate_kernel` at every verify sliding load. Receipt45ab5a9beecd407b resolves the abbreviated `flash::prepare_v` name
as `prepare_varlen_num_blocks_kernel`: this is scheduling metadata, not V
transpose. The earlier interpretation as V preparation was wrong. The
VQ8-sliding dispatch claim remains dropped; the FA3 cost is now measured
in `../window_rebuild/diagnostic_analysis.json`.
The retained native dispatch excerpt is in `audit.json`; the timeline's
full evidence belongs to `muse_decode_layer/dev/s15_timeline/`.

No normal/forced barrier-arm screen or remote send applies after these
claims are dropped. The d16/d32 point-winner integration screens are separate
and recorded in [sliding_round2](../sliding_round2/README.md).
