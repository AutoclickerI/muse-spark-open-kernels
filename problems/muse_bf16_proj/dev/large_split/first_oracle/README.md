# Failed initial oracle, not performance evidence

`017b445ab14f4ec6` timed out with exit -9 after 614.96 seconds and retained no oracle rows. `results.json` therefore marks it unmeasured. The exact blocked instruction is not observed.

The original source had two diagnosed hazards: combined shared-X/math-free rewrites omitted the weight checksum, allowing load elimination, and removing WGMMA also removed the warpgroup rendezvous that prevented a producer warp lapping an mbarrier phase. The repaired batch retains all weight loads, adds the CTA handoff, matches occupancy and traps on a wait exceeding two million cycles. It completes in `9ab81a3f`.

These original `native_validation.json` and `lab_validation.json` files preserve the limited checks that passed. Compilation and portable arithmetic did not validate the synthetic twins' barrier semantics or prove every load survived. In particular, the combined-X/math-free SASS has only four static global loads instead of the repaired minimum 36. They are superseded by the current parent-directory resource and native H200 receipts, not grounds to claim an oracle result.
