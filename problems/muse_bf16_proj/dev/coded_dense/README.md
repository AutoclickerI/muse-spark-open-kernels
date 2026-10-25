# P10 dense coding lane

Integration handoff: `patch_coded_s12.py` **8dc761b0** applies to
`problems/muse_decode_layer/cand_r3_best7_s11_final.py` **acda6c62**.
The independently generated candidate is `s11_check/s11_coded.py` **c5750b4b**.
S12 integration owns `problems/muse_decode_layer/dev/s12/`; this lane does not edit it.

The handoff freezes the measured v3 body, not the experimental cleanup in `coded.py`.
It routes full qkv at T=8 and pool_router at T=1/8. The ROWS table also contains full qkv T=1,
which S11 final's gate split bypasses in the actual decode-1 layer. Preserve that split until
whole-layer pairing establishes whether full coded qkv or coded qkv3 plus split gate is better.

Fixed the old pointer-keyed global encoded bank: packed weights now live in
`state['coded_dense']`, with projection dispatch by role. `check_cand.py` clones both these tensors
and the corresponding BF16 weights before running the layer, exercising changed weight addresses.

Precision audit (CONSTRAINTS 5): `coded` is uint8 (12 bits/weight); `base` int32 and `scales` FP32
encode one exponent-window origin per 16 rows; each scale is exactly 2**(base-112).
`eoff` int32 indexes escape lists; `esc` int32 pairs contain row/column addresses and the original
16-bit BF16 patterns (no widened numerical weights). Original BF16 weights remain for other rows.
Prepare checks bit-exact reconstruction; the audit asserts the tensor inventory, metadata values,
and bit-exact reconstruction again after cloning. There are no prepare-expanded BF16/FP32 weights
inside the coded bank. `coded_state.json` records all shapes/dtypes/bytes.

Verification: cleanup `test_coded.py --seeds 3` passed both row/k layouts, v3/v4, integer-exact GEMVs,
all 65,536 BF16 patterns, +/-zero, denormals, extreme scales, finite escapes and Inf/NaN propagation.
Exact candidate **c5750b4b** passes normal and forced judge-image screens: each has 13/13 correctness
tests, 12/12 graph-timed shapes and 23/23 static checks. Both clone encoded banks and BF16 weights and
check the full precision inventory. `handoff.json` pins source hashes and completion; `s11_check/normal/`
and `s11_check/forced/` hold the current audit/call/precision evidence. Earlier concurrent attempts hit
GPU-memory contention and stayed local; the final serial reruns passed without changing the checks.
The integration lane froze the earlier `ccb512c3` patch; its dispatch has the same state-owned bank
with a redundant presence guard. The final **8dc761b0** patch removes that guard and the frozen module's
inapplicable emitter; integration can consume this final patch directly.

H200 evidence:

| Job | Paired result |
|---|---|
| `35c875e6` (inherited v1) | Wins only at T=1: pool_router -1.84 us, qkv -0.29 us vs balanced; loses T>=16 |
| `320884442a6f4624` (inherited frozen v3) | qkv 10.74/11.20 vs balanced 11.66/11.78 us at T=1/8 (0.921/0.951); pool_router 18.75/19.14 vs 21.12/21.33 (0.888/0.897). T32 qkv 14.21 vs 12.46 (1.140), pool_router 25.74 vs 22.38 (1.150). o and slice lose at T1/8 |
| `3239a2ea8b4c4800` | Completed: `cleanup_lab.py`, v3/v4 x both layouts, frozen v3 and balanced controls, 3 input seeds x 12 cold rounds, A:A. All cells correct; cleanup does not close T>=16. Best T16 qkv/pool 12.44/22.37 vs bal 11.86/21.06; T32 14.11/25.66 vs 12.54/22.47. k-layout T1 qkv 10.87 vs frozen 10.85; T8 qkv 11.49 vs 11.36. V4 never wins. o and slice improve over old coding but still lose to balanced |
| `9aff77cc54bf471b` | Pending: `staging_lab.py` merged by the lab tool with attention's `attn_partial_port`; three seeds x24 rounds, CB2/4/8 and half/k16 activation fragments. No result yet |
| `c446fdab439b44e5` | Pending: `cpall_lab.py` on `muse_prefill_layer`, synthetic dense weights; all-thread cp.async with one CTA barrier vs bulk staging, 3 seeds x24 paired rounds + A:A/balanced. Proj quota exhausted before submission, so only this lane's unsubmitted proj waiter was stopped and rerouted |

`cleanup_sm90.json` pins the cleaned source and ptxas resource counts: all 24 GEMV instantiations and
the decoder compile for sm_90a without spills; RT1/PIPE2 v3 uses 56/64/75 registers at NT1/2/4.
Local image cleanup lab passed every cell (largest relative L2 2.9e-6); timings are sm_120 under
other lanes' load, so no promotion is inferred. `build_cleanup.py` regenerates the payload.

The kernel already decodes each weight once across all T; T>=16 work targets activation-register
lifetimes, staging/occupancy and layout. `build_stream.py` builds half-k32 and k16 activation-fragment
load experiments while retaining one decode per weight. Integer decode remains SHF + 4 LOP3 + 4 PRMT
per eight weights (1.125 instructions/weight, prior SASS audit), below the stream's issue budget.

Next: handoff screens and cleanup receipt are complete. The fragment-lifetime/chunk-size lab and one-barrier lab passed all 14 local-image cells and are queued;
`build_cpall.py` adds all-thread cp.async staging with one CTA barrier, targeting the fixed o/slice costs.
`test_summary.json` pins local test outcomes and payload hashes; `cpall_sm90.json` confirms the new inline-asm arm compiles without spills on Hopper too. Staging shares a proj-host run with attention; cpall uses the prefill host after proj quota ran out. Promote only through integration's identical-input multi-seed whole-layer ratios.
Then o/slice and entropy coding (2.545 exponent bits, ideal 10.545 bits/weight). `DENSER.md` records why naive 11-bit codes would barely save bytes and multiply serial escape work, plus a sparse-bitplane design to measure.
