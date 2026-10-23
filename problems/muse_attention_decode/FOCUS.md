# Attention verify

Serves: G2, G3.

The active source is `cand_attn_portfolio.py`. It preserves the scored q_len=1
kernel and tunes, per q_len=8 scored shape, the decode-gather math family, 27
FA3 settings, the 16 producer/MMA-WGMMA variants, and eight raw-FP8 P/V
variants. The new FP8-PV family is validated on the local sm_120 stand-in
against the existing group path: L2 0.00231 at B1 sliding and 0.00450 at B16
full. Only shape-keyed choices survive `prepare()`.

The previously queued job `4148288` contains the earlier portfolio hash. This
working-tree change adds the decode-gather math family, so the orchestrator
must requeue the new frozen hash in `dev/NEXT_CHECKS.md`.

The H200 evidence identifies the gap. `cand_xqa_diag` check `32c5b4b287c24c38`
measured FA3 versus the best custom paths at full verify B16/B32 as 206.56 vs
675.30 us and 380.22 vs 1,182.91 us. The K128 check measured 207.36 vs
616.74 us and 391.36 vs 1,173.81 us. Those correspond to 327.68/655.36 MB of
raw FP8 KV and only about 0.53/0.56 TB/s for the custom path, versus the
standalone random-gather probe's 2.558 TB/s.

The mechanism measurements are now in the portfolio receipt:

| Mechanism | Evidence | Confirmation packed into `prepare()` |
|---|---|---|
| KV reuse across eight draft rows and eight GQA heads | The custom path's raw-byte bandwidth is far below the gather probe despite one KV head per CTA | `PORTFOLIO_MECH` logs raw KV bytes, the 64 logical query rows per KV page, FA3/custom time and effective TB/s |
| TMA versus page-row `cp.async` | The installed SM90 cubin has repeated `LDGSTS.E.*` tensor-memory loads; XQA uses `cp.async` row gathers | `PORTFOLIO_MECH_BINARY` records the cubin signatures; the shape timings are paired with the same cold-L2 input |
| Split count and SM fill | FA3's cubin contains 64/96/128/192 tile families and persistent schedulers; prior XQA sweeps changed little without that scheduler | `PORTFOLIO_MECH_SPLITS` prints every FA3 `(num_splits, pack_gqa)` time per shape |
| Tensor PV format | FA3's SM90 SASS contains `HGMMA.64x128/176x16.F32.BF16`; FP8-PV remains a legal hypothesis to test against our FP16-PV path | The portfolio times raw-FP8 P/V variants beside FP16-PV and logs the winner |

The local sm_120 screen passes all 16 correctness tests and 12 graph timing
shapes. The forced decode-gather math path also passes reference validation at
B1 sliding, B16 full, and B32 full. Its sm_120 benchmark uses the q1 gather as
the byte-equivalent control: at B16/full it reaches 1.103 TB/s versus 1.441
TB/s (76.5%), and at B32/full 1.301 versus 1.488 TB/s (87.4%). The sm_120
four-stage footprint is 100,352 B, just above that GPU's opt-in limit, so its
stage-3 L=8 launch is a portable stand-in; the H200 path launches the exact
four-stage q1 gather. H200 numbers remain unverified until the orchestrator
requeues and collects the updated portfolio receipt. The target is a custom
path above FA3 at full verify B16/B32; no path has been promoted yet.

The detailed receipt/source audit is preserved in `dev/fa3_mechanism.md`.

The companion ready source is `cand_attn_fa3_struct.py`. It adds an eight-stage
persistent producer/consumer ring beside the existing paths: 128 producer
threads issue LDGSTS-backed `cp.async` gathers behind mbarriers, the consumer
warpgroup runs WGMMA QK/PV, and the final CTA performs the FP32 split merge and
fused gate epilogue. Its sm_120 stand-in is finite, exactly agrees with the
five-stage group path (L2 0.0 for B1 sliding and B16 full), and the full local
screen passes.

Frozen hashes and all ready send candidates are in `dev/NEXT_CHECKS.md`.
Only the orchestrator sends or commits.

`cand_attn_portfolio_v2.py` is the next local candidate and leaves the queued
portfolio untouched. Its clock64 trace shows all four consumer warps with the
same stall pattern: at B32/full the median per-tile wait/issue/MMA spans are
50/238/3,827 cycles, so MMA occupies about 91% of the phase and no single warp
is a straggler. Nsight Systems independently attributes 62.4% of the traced
GPU time to the verify kernel (eight 590 us instances). The v2 portfolio adds
four-stage Hopper overlap and four two-consumer-group split-count settings.
The two-group setting is measured at 1.280 TB/s at B32/full versus 1.306 TB/s
for the one-group math path; it remains below the q_len=1 control (1.495 TB/s)
while preserving correctness. H200 timing is unverified.
