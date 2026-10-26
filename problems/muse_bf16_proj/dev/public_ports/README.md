# P10 public projection ports

Rank the newly permitted public projection mechanisms against **S19's actual site
dispatcher**, including its selected lossless coded banks. Stage ranking is not a
whole-layer promotion. P4 must apply the selected patch to its current S21/S22
composition and pass held-out paired six-load G0.

The donor is [operatorstack at 6d9b7765](https://github.com/operatorstack/open-kernel-challenge-solutions/tree/6d9b77656b6c295e421be9e1424b0f8f99299b0a).
`review.json` records the external, separate clone, immutable source and license
hashes, and each copied kernel's reviewed source range. No donor installation,
build script, test, driver, prepare, dispatcher or import fallback was executed or
copied. `NOTICES.txt` contains both MIT notices; the operatorstack MIT notice also
stays in the adapted kernel source. NathenFernandes v18 uses the previously
reviewed MIT86aff75 source already vendored in our tree.

`review.py` extracts reviewed pure kernels; `build_operator.py` adds separate PDL
twins and preserves FP32 output for the fused pool/router. `build.py` combines the
frozen S19 source, reviewed kernels, exact coded reader and ABI adapter.
`build_lab.py` packs all36 dense site/shape cells into one batch, rather than
spending a host check per kernel. O is raw donor, P adds PDL, C uses exact coded
weights in swap-AB WGMMA, N covers released NF v18 dispatches (already-shipped
ones are explicitly marked). Missing published NF shapes are unmeasured. Inactive
gate/QKV cells are diagnostics, never nominees. Pool/router keeps both FP32 router
planes and pays its real route neighbour; O/unpooler pay native norm/residual
consumers. Active split-gate timings pay the sigmoid/apply kernel and get actual
published whole-layer TASK checks.

The new exact format stores a12-bit payload plus one32-bit escape prefix per128
weights and rare BF16 escape values. Four base banks share QKV/gate and unpooler
slices. GPU packing includes a bitwise roundtrip; no exponent escape limit or
lossy conversion is allowed. A compact-load decoder avoids repeated word loads
that otherwise create inadmissible software-pipeline buffers. The actual decoder
passes every65,536 BF16 bit pattern, including NaNs, infinities and signed zeros.
Prepare bank timings are recorded separately from baseline preparation.

Local evidence: `base_validation.json` (official13 TASK/12 graphs, forced S19),
`local_validation.json` (full adapter dispatch rehearsal with explicit Hopper
stand-ins), `decoder_validation.json` (actual Triton decoder), and
`native_validation.json` (49 actual sm90 cross-compiles; shared-memory admission).
Local stand-ins establish glue/checker coverage, not H200 arithmetic or speed.

Remote ranking:8 fresh input seeds,4 symmetric forward/reverse rounds per arm,
all individual rounds retained losslessly. Every arm gets published projection
TASK random and one-column probes, strict original FP32 router checks, and active
gate layer TASK. `read.py RECEIPT` validates the packet/source identity and reports
candidate/S19 time ratios with95% Student-t intervals on independent seed means,
using the larger primary/A:A log standard error. Empty-graph calibration is
recorded separately and is not subtracted from primary times. Failed/unsupported
arms are unmeasured, never silently converted into losses. See `PORTS.md`,
`ratios.csv`, and `results.json` after native completion.

Submit through the normal h200_lab sender, with the existing P10 process-local
14k reservation (2k suffix guard and full16k receipt preserved):

```sh
python3 problems/muse_bf16_proj/dev/native_boundary/send.py send \
  problems/muse_bf16_proj/dev/public_ports/out/public_projection.py \
  --host muse_bf16_proj --lab-only --import-budget 840 \
  --lane P10 --decision "port public proj kernels per site" --note "..."
```

The first ranking receipt is `c74d2a26699743d0`: all36 cells complete,4408 TASK
checks. [PORTS.md](PORTS.md) contains every per-arm ratio/CI; `results.json` retains
every individual round. Nine points have CI-upper<1 and six other positive means
remain exploratory points. All15 go to the same whole-layer batch; stage CIs are
not a promotion veto. QKV3T16 has an unexplained1975.712us baseline round and a
wide CI; it is not an established gain. Pool/router T32 PDL is.950465
[.944382,.956588], T16 PDL1.0438[1.0237,1.0642]. The new coded WGMMA reader loses
badly and is not selected. See `coded_resources.json` for its representative
255-register/1016-byte-stack sm90 resource usage; this is supporting evidence,
not an exclusive physical timing split.

The measured H200 prepare is137.812s baseline +57.585s for all new banks/NF
variants. The four exact GPU code banks total.028504s and raw tiled banks<.0004s.
Selected S21 preparation including compilation/warmup is not measured yet.

`patch.py BASE.py OUTPUT.py POINTS.json` applies exact optional point dispatches
onto the S21/S22 ABI, with MIT notice in `port_body.py.inc`. `points.json` has
all15; `preferred_points.json` ranks nine first, without excluding other points
from whole-layer evaluation. Base preparation warms its original dispatch before
the new bank is constructed; the selected dispatch is installed for all actual
calls. `P4_HANDOFF.json` contains source identity, complete ratios and caveats.
T1 gate must be rejudged because S21 already has the earlier coded sliding gate.
`shipping_integrity.json` proves the whitespace-clean shipped patch has exactly
the same Python AST, including all embedded source literals, as the screened patch.

The d16/d32 build in `pool.py.inc` keeps S19's original exact12 NF reader and
tries WK4/8, ST1/2, and first-codeword loads before the PDL activation wait.
`pool_tma.py.inc` tests the donor's tiled TMA with BK128/256, ST2/3 and early
weights, ordinary variants reduce shared allocation from164–196KB to61–98KB; corrected
early variants use additional first-tile storage, with exact sizes in
`pool_native_validation.json`. sm90 PTX proves
that only early arms issue their first TMA before the dependency wait.
`whole_validation.json` passes848layer+640strict projection/router TASK in the
local full-batch rehearsal; new TMA body uses declared stand-ins, actual NF
arithmetic executes. `pool_native_validation.json` records all8 sm90 compiles.
`pool_nf_native_validation.json` adds12 actual NVCC sm90a compiles,72–128regs,
zero spills, and the intended PTX load/wait order. Native runtime effect remains
unmeasured. The exact patch passes the official normal/forced local screens,
13TASK/12graphs/6static each (`candidate_validation.json`).

Native whole-layer selection is **unmeasured**: all five hosts have0scored and
0checks. `queue.json` preserves corrected sender3966799 (`5a8ae27c`), replacing3347492 before the real
10-27 09:01KST reset. It freezes S21eacc087b, tests15points individually/portfolio
plus10pool variants on8fresh seeds and8symmetric retained rounds. The exact
queued payload is archived in `runs/sent/5a8ae27c764a6f9b/`; no duplicate is queued.
After its receipt, run `whole_read.py RECEIPT` and send surviving points to P4's
held-out six-load G0. No new G0 promotion, native pool-prefetch result or selected
prepare is claimed.


The night continuation is [night/README.md](night/README.md): exact site sizing,
seven strongest coded DSM/TMA candidates, seven leave-outs, and a no-extra-bank
coded T32 pool alternative. One-shot dispatcher4088520 waits for the corrected
parent receipt, runs `whole_read.py`, hands in-graph points/pool schedules to P4,
drops losers, and sends the admitted next lab through `h200_lab --lane P10`.
All changes to parent3347492 were a necessary correctness repair: its early-TMA
variants hoisted activation loads before GDC. A peeled W tile and opaque post-wait
address dependency now keep every activation TMA after the wait; all eight sm90
schedules verify that property. Arms/seeds/rounds remain unchanged; old snapshot
never submitted, no quota spent. See `night/parent_repair.json`.
