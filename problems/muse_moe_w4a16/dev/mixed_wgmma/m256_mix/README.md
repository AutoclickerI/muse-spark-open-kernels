# M256 expert and phase composition

Serves the decode-layer score. Current measured winner is8483c0c6 `mixed_parallel`,
344.5us mean versus same-run C1g2112366.2 and masked450.2. It is integrated in
`problems/muse_decode_layer/cand_r3_best7_s6.py`, SHA256
`4c98f12fc7fab9730b90d3bae2f7f014ce85bae35439fb688282d98bd167c59f`.
Its vector W13 converter wins over quad346.1us, despite quad's lower SASS count.
Whole-layer H200 composition remains unmeasured.

The new port uses recoded NT1 for experts with counts<=8/16/32 and packed WGMMA
N32/N64 above each threshold, independently in W13 and W2. Six W13 choices cross
six W2 choices plus full C1 W2:42 compositions. Exact C1g2112, exact mixed_parallel,
and threshold16/grid1056 make45 arms. W13 folds in the8483 vector converter;
W2 uses the C1 distributed vector finalizer. Heavy producer/consumer decode-once
work remains with the MoE lane; this experiment does not duplicate it.

All retained scale banks are uint8 exponent/mantissa codes, with sign/zero folded
into packed E2M1. The H_H helper's widened int16 scales are not copied. Both the
NT1 layout and WGMMA layout hold exactly4.5bits per weight including group scales;
dequantization and scale conversion are inside timed kernels, FP32 global scales
in epilogues. The GPU builds disjoint heavy-task prefixes and zeroes NT1 norm
partials; no routed value controls host branches or allocation.

Protocol:12 public weight/router seeds x3fresh fixtures; four cold128MiB AB/BA
orders against EACH C1 and mixed control, A:A for both; finite/L2<=1%/row<=5%
before/after. The oracle consumes actual supplied route IDs/scores, accepting
valid router cutoff flips without enforcing identical IDs across implementations.
Fixed-variant means, medians and paired ratios are reported, with phase means
over all12seeds. No partial seed or failed arm can support promotion.

Local4518 numerical/capture checks pass: threshold edges0..256, empty and all32
expert routes, signed/zero/subnormal scales, separate gate/up globals, poisoned
ACT/Y/SS, scratch-owner churn, local_start64 and nearby M128/192/257. All21 native
SM90a entries have no spills or live-A overwrite;286720 B128 addresses match.
The mixed control's packed banks and eight captured outputs match the literal
accepted8483 payload bit-for-bit. Full lab39.4s within78s,3271-character report.
Host and direct static screens, deadline checks and final hashes are recorded
in [admission.json](admission.json). Local timing does not rank H200.

Payload `prof/lab/moe_m256_mix.py`, SHA256
`974a56676cf6604fb5840913095f425cab3d9050658be886c5dbf35a7d48e93e`.
The one MoE-host snapshot, send7c2160b9dbc119ad, is held for2026-10-25 09:01 KST.
The immediate detached log confirms no judge request before release; current state is
in admission.json. No new expert-mix winner is claimed before the paired H200 means.

`unmeasured_cell.py` is a fixed threshold16/N32 review artifact, not a promoted
M256 winner. Its actual prepare/expert API passes15 fresh-route graph gates and
three static checks. `build_cell.py` freezes the eventual measured recipe without
a runtime portfolio; FOCUS records only the measured S6 winner until replacement.

Reproduce in tools/img: `build.py`, `check.py`, `native.py`, `cold_run.py`,
`check_control.py`, `build_cell.py`, `check_cell.py`; assemble the host with
`tools/h200_lab.py build` and screen through `tools/local_eval.py`.
