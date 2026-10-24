# Expert-wide M128 MoE

Serves G1/G2: improve the decode-layer score. This experiment is complete;
**all six recipes lose** in H200 **640d67e7e56d42e5**, send0a5efab4bc3396f8.
NT1-recoded290.83 us; RS64g528378.67, RS128g528425.03, RS64g2112361.16;
SS64g528750.96, SS128g528785.54, SS128g1056678.86. All36/36 numeric gates
pass, maxL2/row0.00413/0.00469; cold A:A abs mean1.784 us. Best RS graph
route/W13+act/W2+SS/finish medians8.0/235.4/124.6/12.6 us (seed0/fixture0).
`h200_result.json` records the table. The provisional drop-in remains rejected.

D32g from the instruction lab **dc3fea0a20a349bf** is the new incumbent,
272.95 vs NT1rec290.07 us. Its exact cell is handed to decode; the next
[expert/phase-composition lab](../phase_mix/README.md) is accepted as
**770809cf159f41e2**. It retains the best one-read producer as a composable
control. Removing logical weight rereads alone did not overcome this producer's
transport/address/synchronization costs; no DRAM counter isolates a causal
binding limit.
The measured final-phase defect takes priority over an instruction-bound
hypothesis. Recoded M128 cfg3/4/5 passed numerically but took
1649.96/1738.80/1851.23 us, versus NT1's291.70 us; recoding gains only about1%.
The free lane's H200 `97be11d4631b44d6` isolated the serialized last-warp combine:
W2 goes866.4 ->161.6 us after moving finalization into one vector CTA per token.
That finalizer's GPU body is copied byte-for-byte into every path here; its hash
and equality check are in `manifest.json`. All three paths write Y and partial
sum-of-squares, then a separate captured kernel computes norms and weighted sums.
No completion-ticket or last-warp combine remains in these W2 producers.

The old RS128 producer already read each packed expert tile once at M128. Its
mainloop, scale reader and scheduling are preserved in the RS control; only the
W2 finalizer changes. This tests the measured defect before instruction tuning.
It retains repaired K64 register-bank rotation, plus a K128 control and grids
528/2112. The bank stays packed E2M1 plus exact recoded uint8 scales,9 bytes/block.

The new SS path assigns one expert-wide token group to each channel-tile CTA.
It loads each K64 Q/SF tile once, dequantizes once inside the timed kernel, and
writes a shared64x64 BF16 A tile. Every WGMMA token subgroup consumes that same
shared tile before it can be overwritten. N limits64 and128 are tested. Public
shape selects token capacity128/256/384, so M128 and hot129/257 experts remain
one packed-weight group each. Inactive rows are zero-filled. No expanded expert
bank survives prepare, and no input-derived state survives a call.

The SS shared A/B layouts use canonical Hopper B128 descriptors. There is a CTA
barrier and async-proxy fence after shared stores; all WGMMA groups retire before
the next K64 tile overwrites A or B. Native SASS has the shared/shared descriptor
pair (one gdesc group, with no register-A operand), and wait0 at the reuse edge.
The portable SM120 branch reads the same layouts with immediate MMA. W13 stores
only up rows in shared memory; gate values round to BF16 in registers before
SwiGLU, preserving both original BF16 activation boundaries.

Correctness beyond128 exposed a real partial-write defect: a128-thread CTA
reduced SS only for rows0-127 of larger expert groups. Poisoning showed ACT/Y
fully finite but all64 partial sums of row128 unwritten. The writer now strides
over the complete token extent. `partial_write_diagnosis.json` preserves the
failure, and the129/257-token edge gates poison ACT/Y/SS before production and
require every active entry to become finite. No tests or tolerances were weakened.

The H200 lab compares actual NT1-recoded with six fixed recipes: repaired RS
K64/K128, larger RS grid, shared-A N64/N128 and a larger N128 grid. It uses12
weight/router seeds x3 identical real-route fixtures; six reversed/rotated cold
128MiB CUDA-graph orders, duplicate NT1 A:A samples, and independent finite/FP32
gates before and after timing. It reports route/W13+SwiGLU/W2+SS/finalizer phase
medians. The logical packed-tile factor is1.000 for expert-wide paths; this is an
affine address/scheduling guarantee, not a measured DRAM counter or proof that
rereads bind NT1. L2 can save traffic while the decoder still repeats work.

Local evidence:36 fixtures,216 producer +36 NT1 graph checks;79 signed/zero/
subnormal-scale and route-transition checks, hot counts up to257 and local_start64.
Max fixture L2/row0.004129/0.004847. All19 sm90a GEMMs have no spills/C7513;
10 RS K64 paths remain safe. The SS scatter writes all4096 A values exactly
once, and20,480 canonical B128 addresses match. The wrapper passes5 official
tests,6 scored graph shapes and6 static checks; the payload completes in23.6 s
within its78 s local allowance. H200 now confirms numeric
execution and the latency rejection; whole-layer speedup is not claimed.

Reproduce the sent lab:

```bash
python3 problems/muse_moe_w4a16/dev/mixed_wgmma/m128/one_read/build.py
python3 tools/h200_lab.py build prof/lab/moe_m128_no_reread.py --host muse_moe_w4a16 -o problems/muse_moe_w4a16/lab/lab_moe_m128_no_reread.py
tools/img python problems/muse_moe_w4a16/dev/mixed_wgmma/m128/one_read/native.py
CUDA_VISIBLE_DEVICES=0 tools/img env MAX_JOBS=2 python problems/muse_moe_w4a16/dev/mixed_wgmma/m128/one_read/check.py
CUDA_VISIBLE_DEVICES=0 tools/img env MAX_JOBS=2 python problems/muse_moe_w4a16/dev/mixed_wgmma/m128/one_read/check_edges.py
```

`build_dropin.py --backend <winner> --config <id> --grid <grid>` can emit a
fixed recipe. The provisional RS3g2112 cell passes8 local wrapper gates but
is rejected by the complete H200 result above. No decode dispatch uses it.
