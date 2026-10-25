# What the four-arm lab identifies

H200 `360810b110a44d36`, payload `eac899e7`, exact S11 `acda6c62` consumer69.
All times below are means of separate median-of-three CUPTI node samples over16 routed fixtures.
Unprofiled paired event ratios are in README; subtracting mean node samples is descriptive, not an
independent per-seed causal estimator with a confidence interval.

| M | A-B dequant contrast us | B-C residual us | C-best D us | A-best D us |
|---|---:|---:|---:|---:|
| 8 | 4.930 | 12.370 | -1.642 | 15.658 |
| 16 | 6.562 | 15.630 | 3.546 | 25.738 |
| 32 | 19.479 | 15.932 | 16.060 | 51.471 |

A/B keeps loads, MMA count and synchronization, but changes the register dependency graph and compiler
schedule with a bounded wrong-math reinterpret. It estimates exposed decoder cost for this kernel; it is
not a precise decoder-only instruction issue counter. B/C also removes activation reads, dependent MMA,
expert readiness, norm/output/finalizer work. C/D changes repeated8-token-block weight requests to unique
expert ranges, address/order, stage geometry and shared reservation. C/D therefore cannot all be called a
pure memory-pipeline latency. No transaction-level HBM bandwidth or stall counter was collected.

What is established: original matched weight loads reach4.192TB/s atM8. The exact arithmetic kernel is
slower even there, so expanding the original raw queue alone cannot explain/fix every shape. AtM32,
removing exposed dequant is insufficient and the repeated-request schedule also matters. Separate fixed-work
ONCE/REDO (`e4ad355e`) measures14.75us of re-decode exposure atM32, while ONCE still loses36.75us toS11.
Dropping computation by capping token count is not an estimate of decode cost.

# Independent raw queue

The first full-K RS implementation eliminates re-decode and K-split reduction but loses1.52–2.29x
(`00f4adc3`). Fusing the exact activation permutation and increasing copies toK128/K256 improves the
best ratio to1.37/1.32/1.11 atM8/16/32 (`b3f1c4f`). It still misses4TB/s. S11's K-split also shortens the
per-CTA dependency chain; this implementation changes that geometry, so decode-once is not free.

`build_wg_queue.py` retains the corrected K128 adaptive RS math but changes transport:

- One raw N64/K128 unit is4096 code bytes +512 raw scale bytes =4608B, still NVFP4 with byte-exact scales.
  Prepare only permutes the byte layout; both halves have inverse equality checks.
- A single-thread TMA issues one contiguous4608B unit, versus per-thread code/scale cp.async copies.
  A raw ring of4/8/12 units is independent of3 activation stages of8192B each. The activation stage count
  is set by async WGMMA lifetimes, not by the desired weight prefetch depth.
- The issued weight queue can cover18/36/54KiB perCTA. H200 smem limits imply3/3/2CTAs perSM, giving
  nominal54/108/108KiB queued raw weights perSM. These are storage/issue upper bounds; completed loads
  and tails mean they are not measurements of actual outstanding DRAM bytes.
- Dynamic shared atM32:45,992/64,456/82,920B. Hopper static compile151–153regs, no local stack/spills,
  no serialization warning. Actual residency is queried on the H200 and caps the grid; no oversubscribed
  grid barrier is allowed.
- Activation slots use k%3. After two K64 WGMMA groups in iterationk, wait_group1 has retired iterationk-1.
  Thus k+2 may overwrite the same slot as k-1. Register A fragments alternate over two constant buffers.
  Raw-stage reuse follows a CTA barrier after all raw loads. Each tile drains WGMMA and cp.async before
  reinitializing its TMA barriers. Initialization and its release fence run on the issuing thread before
  the CTA barrier; an initial distributed-init version failed locally and was corrected before sending.

A win would justify a whole-layer paired follow-up and a patch onto frozenS11. A loss would show that this
raw-queue/TMA change does not recover the full-K math geometry's cost; it would not justify claiming the
original kernel is HBM-bound. The next structural question would separate producer decoding/copies from
consumer MMA with an expanded on-chip operand queue, charging its extra shared storage and warp occupancy.
