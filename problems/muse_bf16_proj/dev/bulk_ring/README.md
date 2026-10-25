# Persistent bulk-ring dense projections (P7)

One real-weight **proj-host** batch: send key `ef2048bf49b3fdf4` (`sent.json` pins the
snapshot), judge job `87b2667f5dac43a6`. No H200 result or dispatch promotion is
claimed before its receipt.

`bandwidth.json` and FINDINGS contain all16 best7 role/M rows against a3.6TB/s
streaming control. M32 pool+router has the largest opportunity ceiling:
75,497,472 weight bytes /33.69us =2.241TB/s, versus20.97us at3.6TB/s; gap
12.72us/layer,788.5us over62 layers. QKV leads at M1/M8; pool+router at M16/M32.
These are useful weight bytes, excluding X/output/partial traffic. Raw graph
times contain launch/replay overhead, so the gap is an upper bound on available
time, not an estimate of recoverable bandwidth alone.

The prior95ab383d observer measured47-101 entry cycles and10-11 PDL-wait cycles.
This kernel targets outstanding bytes and shared-memory operand transport.
At132 SMs,3.6TB/s times an assumed1us DRAM latency requires~27,273 bytes/SM in
flight. The sweep provides32-128KiB of weight ring per CTA,4/8 stages of8/16/32KiB,
and1/2 CTAs per SM. Explicit K splits1/2/4/8 and row tiles32/64/128 vary the
work distribution and final partial count. The1us latency is a design estimate,
not a measured H200 latency.

`projection.py` is self-contained and exposes `BulkRingProjection(w,M,config)`.
It returns `(partials[S,M,N],S)`, the decode consumer ABI. Prepare it outside
capture, once per concurrent call site. The bank preserves original BF16 bits:
`[N/R,R,split,K/split/KC,KC]` becomes
`[split,N/R,K/split/KC,R,KC]`, with an odd-row64B XOR inside each shared tile.
The XOR is self-inverse and separates adjacent rows' LDS.128 bank groups.
There is no conversion of the BF16 bank or router hi/lo rows to a coarser format.

Each CTA owns `[floor(total_tiles*id/grid),floor(total_tiles*(id+1)/grid))`.
It can cross row and explicit-split boundaries. Every CTA streams the same
number of K tiles, differing by at most one. A row's overlapping CTA intervals
write separate FP32 partial slots; its last owner initializes unused slots in
the same launch. The existing QK prep, boundary or route consumer folds them.
No separate zero/reduction kernel, atomic counters or global spin loop is used.
The grid contains exactly SM-count or twice SM-count CTAs, with all M tokens
inside each CTA. Hopper shared-memory reservations cap residency at1/2 CTAs/SM,
so there is no second grid wave. Different final epilogues can still create a
completion tail; the receipt reports SM coverage, equal-work counts and tail.

Thread0 fills the ring using
[`cp.async.bulk` and transaction-count mbarriers](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk).
The initial immutable-weight transfers precede the PDL activation wait.
Each consumed slot is refilled after the CTA's reads finish. The async proxy
fence publishes barrier initialization. MMA uses original BF16 weights and
activations with FP32 accumulators. Performance launches contain no timestamps;
an extra launch records globaltimer start/end and work distribution.

The batch freezes best7_scale42f543a0 and its recorded dense choices from
1b80ffc0. It uses the proj host's actual QKV, O, pooler, unpooler and FP32 router.
The controlled segment uses identity norms, fixed alpha1/beta0.5 and zero
router bias because that host has no learned decode boundary tensors. Those
fixture values are declared and identical across arms. Input banks are seeded
725108; no random learned weight bank is created. This segment isolates the
projection change and does not establish full-layer agreement.

All16 role/M pairs compare21 configurations against best7, both producer-only
and producer+consumer. A:A controls cover both endpoints. Every finalist gets24
AB/BA pairs over three shared input banks; every replay is preceded by a256MiB
L2 flush outside its events. Every configuration passes fresh FP32 raw-dot and
router1e-4 gates and an exact-column probe. All32-bit partial slots are written.
The O/boundary/pool segment compares O, P and OP replacements at every M; M1
also compares the unchanged O/config6 cell and O6+P. Routing stability is checked
outside the measured segment and printed as flips. Each graph has its own
capture stream so resetting another graph cannot invalidate its cuBLAS workspace.

Local admission uses the actual Hopper132/264-grid and residency-prepare branch.
This sm120 GPU has100KiB shared/SM, so explicit driver stand-ins cap the reserve
and five oversized configurations use shorter native rings, retaining their
packing, split ownership and consumer ABI. Native4/8-stage pipelines execute on
all fitting shapes; the full oversized Hopper variants compile as sm90a.
`native_validation.json`, `validation.json` and `compile_sm90a.json` state exactly
which checks ran. Local timing does not establish H200 speed.

Receipt lines: `R Mrole cCONFIG B=producer/combined N=producer/combined`, paired
ratio/seeds, producer/combined AA ratios, grid/partial-count/ring bytes, observed
SM coverage, tile-count range, completion tail, then21 producer/combined coarse
ratios. `O6` reports the exact existing cell. `S` reports segment times, paired
ratio, route-token flips and all three seed ratios. Compact lines are returned
last, under the host's9,000-character lab cap.

```bash
python problems/muse_bf16_proj/dev/bulk_ring/build.py
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_bf16_proj/dev/bulk_ring/validate.py
tools/img python problems/muse_bf16_proj/dev/bulk_ring/audit.py
python tools/h200_lab.py build prof/lab/dense_bulk_ring.py --host muse_bf16_proj -o problems/muse_bf16_proj/lab/lab_dense_bulk_ring.py
CUDA_VISIBLE_DEVICES=0 tools/img python problems/muse_bf16_proj/dev/bulk_ring/screen.py
python tools/h200_lab.py send prof/lab/dense_bulk_ring.py --host muse_bf16_proj --priority 1.2 --not-before 2026-10-25T09:02:00+09:00 --note "P7 equal-work bulk ring vs best7_scale, real dense weights, O6 control, cold ABBA and AA"
```

After a receipt, run `read_receipt.py JOB`. It only reads an already-landed
local file. Candidate cells must pass a live whole-layer route/output check
before the decode lane changes dispatch. Add a `('bulk',config)` projection
branch returning
`state['best7_bulk_ops'][(M,role,config)](x)` directly; instantiate that cell in
prepare. Keep the tuple and partial count intact through the existing consumer.
The existing tensor-returning `('stream',...)` ABI is different.

Local final result:756 native checks/252 graphs,16 roles x21 configs,4 segments,
9 host tests and3 static checks pass;33 sm90a specializations have no spills.
The forced Hopper prepare branch caught a shared-specialization limit bug:
setting a smaller reserve lowered a previously prepared cell's launch limit.
Prepare now fixes that limit at the device opt-in maximum; each launch retains
its own80/120KiB residency reservation. `local_failures.json` preserves it.
Local O/OP segment arms change2/2/4 route-token selections at M8/16/32 over
three inputs; pool-only and M1 arms have zero flips. These are recorded
limitations requiring live-layer admission, not hidden numerical successes.
